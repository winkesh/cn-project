"""Run the evaluation matrix and write results/results.csv.

    python -m eval.run_matrix --video data/test/clip.mp4 --frames 300
    python -m eval.run_matrix --synthetic --quick          # smoke test

Experiments
    E1 loss   : Bernoulli loss 0-30 %, one-way delay 25 ms, deadline 100 ms
    E2 burst  : Gilbert-Elliott, 5 % mean loss, mean burst 1-8 packets
    E3 rtt    : one-way delay 10-100 ms at 5 % loss (where does NACK stop working?)
    E4 ilv    : interleaving on/off under bursty loss
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom import conceal as C  # noqa: E402
from phantom.config import SchemeConfig  # noqa: E402
from phantom.sim import NetConfig, simulate  # noqa: E402
from phantom.video import list_videos, load_video, synthetic  # noqa: E402
from tools.fetch_clips import CONTENT  # noqa: E402


def content_of(clip: str) -> str:
    return "game" if clip.startswith("game_") else CONTENT.get(clip, "other")


def schemes(has_net: bool, has_fg: bool = False):
    s = [
        SchemeConfig("Freeze (no retx)", "none", "freeze"),
        SchemeConfig("Copy-tile (no retx)", "none", "copy"),
        SchemeConfig("Flow extrapolation", "none", "flow"),
        SchemeConfig("NACK + copy", "nack", "copy"),
        SchemeConfig("FEC k=5 + copy", "fec", "copy", fec_k=5),
        SchemeConfig("TCP", "tcp", "copy"),
    ]
    if has_net:
        s += [SchemeConfig("Phantom (RepairNet)", "none", "repairnet"),
              SchemeConfig("Phantom + NACK", "nack", "repairnet")]
    if has_fg:
        s += [SchemeConfig("Phantom-Interp (framegen, +33 ms)", "none", "framegen")]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="test clip (default: first video in data/test)")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--weights", default="weights/repairnet.pt")
    ap.add_argument("--framegen", default="weights/framegen.pt")
    ap.add_argument("--seeds", type=int, default=1, help="loss-pattern seeds per clip (clips add variety)")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--exp", default="loss,burst,rtt,ilv")
    ap.add_argument("--no-ssim", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        clips = {"synthetic": synthetic(args.frames)}
    else:
        paths = [Path(args.video)] if args.video else list_videos("data/test")
        if not paths:
            sys.exit("no test video: pass --video or put clips in data/test")
        clips = {p.stem: load_video(p, max_frames=args.frames) for p in paths}
    for c, f in clips.items():
        print(f"clip {c} ({content_of(c)}): {len(f)} frames")

    has_net = Path(args.weights).exists()
    has_fg = Path(args.framegen).exists()
    if not has_net:
        print(f"[warn] {args.weights} not found: RepairNet schemes skipped")
    if not has_fg:
        print(f"[warn] {args.framegen} not found: framegen scheme skipped")
    names = ["freeze", "copy", "flow"] + (["repairnet"] if has_net else []) + (["framegen"] if has_fg else [])
    concealers = {n: C.make(n, args.weights, args.framegen) for n in names}
    S = schemes(has_net, has_fg)
    seeds = range(1 if args.quick else args.seeds)
    exps = set(args.exp.split(","))

    runs = []  # (experiment, x, scheme, net)
    if "loss" in exps:
        for L in ([0.0, 0.05, 0.2] if args.quick else [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]):
            for sc in S:
                for sd in seeds:
                    runs.append(("loss", L * 100, sc, NetConfig(loss=L, delay_ms=25, seed=sd)))
    if "burst" in exps:
        for b in ([1, 4] if args.quick else [1, 2, 4, 8]):
            for sc in S:
                if sc.recovery == "tcp":
                    continue
                for sd in seeds:
                    runs.append(("burst", b, sc, NetConfig(loss=0.05, loss_model="ge", burst=b, seed=sd)))
    if "rtt" in exps:
        for d in ([10, 50] if args.quick else [10, 25, 40, 50, 75, 100]):
            for sc in S:
                if sc.conceal in ("freeze", "flow"):
                    continue
                for sd in seeds:
                    runs.append(("rtt", 2 * d, sc, NetConfig(loss=0.05, delay_ms=d, seed=sd)))
    if "ilv" in exps:
        for ilv in (True, False):
            for sc in S:
                if sc.recovery != "none" or sc.conceal == "freeze":
                    continue
                sc2 = SchemeConfig(sc.name + ("" if ilv else " [no interleave]"), sc.recovery, sc.conceal,
                                   interleave=ilv)
                for sd in seeds:
                    runs.append(("ilv", int(ilv), sc2, NetConfig(loss=0.05, loss_model="ge", burst=4, seed=sd)))

    runs = [(c, *r) for c in clips for r in runs]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, (clip, exp, x, sc, net) in enumerate(runs):
        cc = concealers[sc.conceal]
        if hasattr(cc, "times_ms"):
            cc.times_ms.clear()
        res, _ = simulate(clips[clip], sc, net, concealer=cc, compute_ssim=not args.no_ssim)
        res.update(experiment=exp, x=x, clip=clip, content=content_of(clip))
        rows.append(res)
        print(f"[{i + 1}/{len(runs)}] {clip[:10]:10s} {exp:5s} x={x:<5g} {sc.name:24s} PSNR {res['psnr_mean']:.2f} "
              f"(p5 {res['psnr_p5']:.2f})  frozen {res['frozen_pct']:.1f}%  lat95 {res['latency_p95']:.0f} ms  "
              f"ovh {res['overhead_pct']:.1f}%  [{time.time() - t0:.0f}s]", flush=True)
        if (i + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
