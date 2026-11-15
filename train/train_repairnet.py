"""Train RepairNet on any folder of videos, with loss masks produced by the real packetizer layout.

    python -m train.train_repairnet --data data/train --minutes 40

Masks: each frame is split into n packets (n ~ U[30,120], as at 640x360 JPEG tiles), tiles are
interleaved round-robin exactly like phantom.sender, and packets are dropped with a Bernoulli or
Gilbert-Elliott loss process at a random rate. With prob 0.15 the whole frame is lost
(pure extrapolation). So the network trains on exactly the hole patterns it will see at runtime.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom.channel import Bernoulli, GilbertElliott  # noqa: E402
from phantom.config import N_TILES, TILES_X, TILES_Y, TILE  # noqa: E402
from phantom.models.repairnet import RepairNet, count_params  # noqa: E402
from phantom.sender import layout  # noqa: E402
from phantom.video import list_videos, load_video, synthetic  # noqa: E402


def sample_tile_mask(rng: random.Random) -> np.ndarray:
    if rng.random() < 0.15:
        return np.ones(N_TILES, bool)
    n = rng.randint(30, 120)
    loss = rng.uniform(0.02, 0.5)
    if rng.random() < 0.5:
        lm = Bernoulli(loss, rng)
    else:
        lm = GilbertElliott.for_mean(loss, rng.uniform(1.5, 8), rng)
    lost_pkts = [lm.lost() for _ in range(n)]
    if not any(lost_pkts):
        lost_pkts[rng.randrange(n)] = True
    groups = layout(range(N_TILES), n, interleave=rng.random() < 0.8)  # same layout as the sender
    tile_pkt = np.zeros(N_TILES, int)
    for j, g in enumerate(groups):
        tile_pkt[g] = j
    return np.array([lost_pkts[p] for p in tile_pkt])


def tile_to_px(m: np.ndarray) -> np.ndarray:
    return np.kron(m.reshape(TILES_Y, TILES_X), np.ones((TILE, TILE), bool))


class Clips:
    def __init__(self, videos: list[list[np.ndarray]], seed: int = 0):
        self.videos = [v for v in videos if len(v) >= 3]   # triplets (3 frames) count as tiny videos
        self.rng = random.Random(seed)

    def sample(self, batch: int, crop_tiles: int = 0):
        """crop_tiles > 0: random tile-aligned crop of crop_tiles x crop_tiles tiles (faster training;
        the hole pattern still comes from the full-frame packet layout)."""
        p2s, p1s, cur, gts, ms = [], [], [], [], []
        for _ in range(batch):
            v = self.rng.choice(self.videos)
            stride = self.rng.choice([1, 1, 1, 2]) if len(v) >= 5 else 1
            t = self.rng.randrange(2 * stride, len(v))
            seq = [v[t - 2 * stride], v[t - stride], v[t]]
            if self.rng.random() < 0.3:
                seq = seq[::-1]
            if self.rng.random() < 0.5:
                seq = [s[:, ::-1] for s in seq]
            m = tile_to_px(sample_tile_mask(self.rng))
            if crop_tiles:
                for _try in range(10):
                    ty = self.rng.randint(0, TILES_Y - crop_tiles) * TILE
                    tx = self.rng.randint(0, TILES_X - crop_tiles) * TILE
                    s = crop_tiles * TILE
                    if m[ty:ty + s, tx:tx + s].any():
                        break
                seq = [x[ty:ty + s, tx:tx + s] for x in seq]
                m = m[ty:ty + s, tx:tx + s]
            partial = seq[2].copy()
            partial[m] = seq[1][m]              # receiver pre-fills lost tiles by copy
            p2s.append(seq[0]); p1s.append(seq[1]); cur.append(partial); gts.append(seq[2]); ms.append(m)

        def T(arr):
            a = np.stack([np.ascontiguousarray(x[..., ::-1]) for x in arr])
            return torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255.0

        mask = torch.from_numpy(np.stack(ms)[:, None].astype(np.float32))
        return T(p2s), T(p1s), T(cur), mask, T(gts)


def load_triplets(root: str, max_n: int) -> list[list[np.ndarray]]:
    """Load framegen-style triplet folders as 3-frame clips, resized to 640x360 (parallel PNG decode)."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    from phantom.tiles import resize_frame

    dirs = sorted(p for p in Path(root).iterdir() if (p / "im3.png").exists())
    random.Random(0).shuffle(dirs)
    dirs = dirs[:max_n]

    def one(d):
        return [resize_frame(cv2.imread(str(d / f"im{k}.png"))) for k in (1, 2, 3)]

    with ThreadPoolExecutor(12) as ex:
        return list(ex.map(one, dirs))


def masked_psnr(pred, gt, mask):
    se = ((pred - gt) ** 2 * mask).sum() / (mask.sum() * 3 + 1e-8)
    return (10 * torch.log10(1.0 / (se + 1e-10))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="weights/repairnet.pt")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop-tiles", type=int, default=5, help="0 = full frames")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--max-frames", type=int, default=1500, help="per video")
    ap.add_argument("--synthetic", action="store_true", help="smoke test without real videos")
    ap.add_argument("--triplets", default="", help="folder of <id>/im1.png,im2.png,im3.png (e.g. framegen game triplets)")
    ap.add_argument("--triplets-max", type=int, default=1500)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.synthetic:
        train_v = [synthetic(120, seed=s) for s in range(3)]
        val_v = [synthetic(60, seed=99)]
    else:
        vids = list_videos(args.data)
        if not vids:
            sys.exit(f"no videos in {args.data}")
        print("train videos:", [v.name for v in vids])
        clips = [load_video(v, max_frames=args.max_frames) for v in vids]
        # validation = last VAL_FRAMES of every training clip (+ some held-out triplets); test clips are
        # never touched during training or checkpoint selection
        VAL_FRAMES = 40
        train_v = [c[:-VAL_FRAMES] for c in clips]
        val_v = [c[-VAL_FRAMES:] for c in clips]
        if args.triplets:
            trip = load_triplets(args.triplets, args.triplets_max + 100)
            val_v += trip[:100]
            trip = trip[100:]
            print(f"+ {len(trip)} game triplets from {args.triplets} (+100 held out for validation)")
            # each triplet is one 'video'; repeat natural videos so both sources are sampled evenly
            reps = max(1, len(trip) // max(1, len(train_v)) // 4)
            train_v = train_v * reps + trip
    print("frames:", sum(map(len, train_v)), "train /", sum(map(len, val_v)), "val")

    train, val = Clips(train_v, 0), Clips(val_v, 1)
    val_batches = [val.sample(4) for _ in range(8)]   # full frames

    # prefetch training batches on a background thread so the GPU never waits for numpy
    import queue
    import threading
    q: queue.Queue = queue.Queue(maxsize=8)

    def producer():
        while True:
            q.put(train.sample(args.batch, crop_tiles=args.crop_tiles))

    for _ in range(2):
        threading.Thread(target=producer, daemon=True).start()

    torch.backends.cudnn.benchmark = True
    net = RepairNet(base=args.base).to(dev)
    print(f"RepairNet params: {count_params(net) / 1e6:.2f}M on {dev}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t_end = time.time() + args.minutes * 60
    step, best = 0, -1.0
    while time.time() < t_end:
        net.train()
        p2, p1, cur, m, gt = (x.to(dev, non_blocking=True) for x in q.get())
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            pred = net(p2, p1, cur, m)
        pred = pred.float()
        l1 = (torch.abs(pred - gt) * m).sum() / (m.sum() * 3 + 1e-8)
        # gradient loss keeps edges sharp inside the holes
        gx = lambda x: x[..., :, 1:] - x[..., :, :-1]
        gy = lambda x: x[..., 1:, :] - x[..., :-1, :]
        lg = (torch.abs(gx(pred) - gx(gt)) * m[..., :, 1:]).mean() + (torch.abs(gy(pred) - gy(gt)) * m[..., 1:, :]).mean()
        loss = l1 + 0.5 * lg
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % args.eval_every == 0:
            net.eval()
            ours, copy = [], []
            with torch.no_grad():
                for vb in val_batches:
                    p2, p1, cur, m, gt = (x.to(dev) for x in vb)
                    ours.append(masked_psnr(net(p2, p1, cur, m), gt, m))
                    copy.append(masked_psnr(cur, gt, m))
            o, c = float(np.mean(ours)), float(np.mean(copy))
            left = (t_end - time.time()) / 60
            print(f"step {step:6d} loss {loss.item():.4f} | val masked PSNR ours {o:.2f} dB vs copy {c:.2f} dB "
                  f"(+{o - c:.2f}) | {left:.1f} min left", flush=True)
            if o > best:
                best = o
                torch.save({"model": net.state_dict(), "base": args.base, "step": step,
                            "val_masked_psnr": o, "val_copy_psnr": c}, args.out)
    print(f"done. best val masked PSNR {best:.2f} dB -> {args.out}")


if __name__ == "__main__":
    main()
