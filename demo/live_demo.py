"""Live side-by-side demo over real UDP sockets on this machine.

    python -m demo.live_demo --src data/test/clip.mp4          # a video file (loops)
    python -m demo.live_demo --src 0                           # webcam

sender --UDP--> lossy relay (:5000) --UDP--> receiver (:5004) --feedback--> sender

Panels: original | what arrived (lost tiles in magenta) | baseline concealment | Phantom (RepairNet)
Keys:   + / -  loss +-5 %     b  random <-> bursty loss     c  cycle baseline (freeze/copy/flow)
        o      highlight synthesised tiles                  q  quit
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom import conceal as C  # noqa: E402
from phantom import metrics as M  # noqa: E402
from phantom.live.endpoints import LiveReceiver, LiveSender  # noqa: E402
from phantom.live.relay import Relay  # noqa: E402
from phantom.receiver import assemble  # noqa: E402
from phantom.tiles import tile_mask_to_pixels  # noqa: E402
from phantom.video import load_video  # noqa: E402

MAGENTA = np.array([255, 0, 255], np.uint8)


def label(img, text, sub=None):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30 if not sub else 52), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
    return img


def net_side(src, delay_ms, cr, shared):
    """Child process: sender -> relay. Loss settings come from shared values the UI changes."""
    relay = Relay(5000, ("127.0.0.1", 5004), loss=shared["loss"].value, delay_ms=delay_ms).start()
    tx = LiveSender(src, ("127.0.0.1", 5000), cr_threshold=2.0 if cr else 0.0).start()
    cur = (relay.loss, relay.model)
    while not shared["stop"].is_set():
        want = (shared["loss"].value, "ge" if shared["bursty"].value else "bernoulli")
        if want != cur:
            relay.set_loss(want[0], want[1], 4.0)
            cur = want
        shared["bytes"].value = tx.bytes_sent
        shared["rep_loss"].value = tx.reported_loss
        time.sleep(0.05)
    tx.stop()
    relay.stop()


class RelayView:
    def __init__(self, s):
        self.s = s

    @property
    def loss(self):
        return self.s["loss"].value

    @property
    def model(self):
        return "ge" if self.s["bursty"].value else "bernoulli"

    def set_loss(self, loss, model=None, burst=None):
        self.s["loss"].value = max(0.0, min(0.9, loss))
        if model:
            self.s["bursty"].value = int(model == "ge")

    def stop(self):
        self.s["stop"].set()


class SenderView:
    def __init__(self, s):
        self.s = s

    @property
    def bytes_sent(self):
        return self.s["bytes"].value

    @property
    def reported_loss(self):
        return self.s["rep_loss"].value

    def stop(self):
        self.s["stop"].set()


class RefFrames:
    def __init__(self, frames):
        self.frames = frames

    def get(self, fid):
        return self.frames[fid % len(self.frames)] if self.frames else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="0")
    ap.add_argument("--weights", default="weights/repairnet.pt")
    ap.add_argument("--framegen", default="weights/framegen.pt")
    ap.add_argument("--loss", type=float, default=0.10)
    ap.add_argument("--delay", type=float, default=10.0)
    ap.add_argument("--deadline", type=float, default=100.0)
    ap.add_argument("--no-cr", action="store_true", help="disable conditional replenishment")
    args = ap.parse_args()

    # sender + lossy relay run in their own process (separate GIL, like a separate machine)
    import multiprocessing as mp
    shared = {"loss": mp.Value("d", args.loss), "bursty": mp.Value("b", 0), "bytes": mp.Value("q", 0),
              "rep_loss": mp.Value("d", 0.0), "stop": mp.Event()}
    netp = mp.Process(target=net_side, args=(args.src, args.delay, not args.no_cr, shared), daemon=True)
    netp.start()
    relay, tx = RelayView(shared), SenderView(shared)
    rx = LiveReceiver(5004, deadline_ms=args.deadline)
    # reference frames for the PSNR readout: the sender loops the same file from frame 0
    ref = load_video(args.src) if not str(args.src).isdigit() else []
    originals = RefFrames(ref)

    baselines = [C.Freeze(), C.CopyTile(), C.FlowExtrapolate()]
    bi = 0
    modes = []  # (concealer, panel name, extra latency in frames)
    if Path(args.weights).exists():
        modes.append((C.RepairNetConcealer(args.weights), "Phantom (RepairNet, +0 ms)", 0))
    if Path(args.framegen).exists():
        fb = modes[0][0] if modes else C.FlowExtrapolate()
        modes.append((C.FramegenInterp(args.framegen, fallback=fb), "Phantom-Interp (framegen, +33 ms)", 1))
    if not modes:
        print("[warn] no weights found -> Phantom panel uses flow extrapolation")
        modes.append((C.FlowExtrapolate(), "Phantom (flow fallback)", 0))
    mi = 0
    pending: list = []   # finalised frames waiting for their successor (interp mode)

    hist_b, hist_p = [], []
    ps_b, ps_p = deque(maxlen=60), deque(maxlen=60)
    overlay = True
    t_rate, b_rate = time.perf_counter(), 0
    mbps = 0.0
    win = "Phantom Frames"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 760)

    while True:
        pending.extend(rx.poll())
        while len(pending) > modes[mi][2]:
            phantom, pname, look = modes[mi]
            fid, st = pending.pop(0)
            a_b = assemble(st, fid, hist_b[-1] if hist_b else None)
            a_p = assemble(st, fid, hist_p[-1] if hist_p else None)
            rx.report_missing(a_p.missing)
            mask = tile_mask_to_pixels(a_p.missing)
            out_b = baselines[bi](a_b.image, mask, hist_b) if a_b.missing else a_b.image
            if not a_p.missing:
                out_p = a_p.image
            elif look and pending:
                nxt = assemble(pending[0][1], pending[0][0], a_p.image)
                out_p = phantom(a_p.image, mask, hist_p, nxt=nxt.image, nxt_missing_frac=len(nxt.missing) / 144)
            else:
                out_p = phantom(a_p.image, mask, hist_p)
            for h, o in ((hist_b, out_b), (hist_p, out_p)):
                h.append(o)
                if len(h) > 3:
                    h.pop(0)
            orig = originals.get(fid)
            if orig is None:          # webcam: no reference, show what arrived instead
                orig = a_p.image
            else:
                ps_b.append(M.psnr(out_b, orig))
                ps_p.append(M.psnr(out_p, orig))
            arrived = a_p.image.copy()
            arrived[mask] = MAGENTA
            show_p = out_p.copy()
            if overlay and mask.any():
                edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                show_p[edge] = (0, 255, 255)
            lost_pct = 100 * len(a_p.missing) / 144
            infer = np.mean(phantom.times_ms[-30:]) if getattr(phantom, "times_ms", None) else 0
            top = np.hstack([label(orig, "Original (sender)"),
                             label(arrived, f"What arrived: {lost_pct:.0f}% of tiles lost")])
            bot = np.hstack([label(out_b, f"Baseline: {baselines[bi].name}",
                                   f"PSNR {np.mean(ps_b):.1f} dB"),
                             label(show_p, pname, f"PSNR {np.mean(ps_p):.1f} dB   infer {infer:.1f} ms")])
            grid = np.vstack([top, bot])
            now = time.perf_counter()
            if now - t_rate > 1:
                mbps = (tx.bytes_sent - b_rate) * 8 / (now - t_rate) / 1e6
                t_rate, b_rate = now, tx.bytes_sent
            bar = np.zeros((40, grid.shape[1], 3), np.uint8)
            txt = (f"link loss {relay.loss:.0%} ({'bursty' if relay.model == 'ge' else 'random'})   "
                   f"receiver-reported loss {tx.reported_loss:.1%}   send rate {mbps:.1f} Mbit/s   "
                   f"buffer {args.deadline + 33 * look:.0f} ms, no retransmission   "
                   f"[+/-] loss [b] burst [c] baseline [m] mode [o] overlay [q] quit")
            cv2.putText(bar, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, np.vstack([grid, bar]))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k in (ord("+"), ord("=")):
            relay.set_loss(relay.loss + 0.05)
        elif k in (ord("-"), ord("_")):
            relay.set_loss(relay.loss - 0.05)
        elif k == ord("b"):
            relay.set_loss(relay.loss, "bernoulli" if relay.model == "ge" else "ge", 4.0)
        elif k == ord("c"):
            bi = (bi + 1) % len(baselines)
        elif k == ord("m"):
            mi = (mi + 1) % len(modes)
        elif k == ord("o"):
            overlay = not overlay
    tx.stop(); relay.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
