"""Standalone receiver window for a two-machine demo.

    python -m demo.receiver_view --port 5004
Shows: what arrived (lost tiles in magenta) | copy-tile baseline | Phantom (RepairNet).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom import conceal as C  # noqa: E402
from phantom.live.endpoints import LiveReceiver  # noqa: E402
from phantom.receiver import assemble  # noqa: E402
from phantom.tiles import tile_mask_to_pixels  # noqa: E402
from demo.live_demo import label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5004)
    ap.add_argument("--deadline", type=float, default=150.0)
    ap.add_argument("--weights", default="weights/repairnet.pt")
    a = ap.parse_args()
    rx = LiveReceiver(a.port, deadline_ms=a.deadline)
    base = C.CopyTile()
    ph = C.RepairNetConcealer(a.weights) if Path(a.weights).exists() else C.FlowExtrapolate()
    hb, hp = [], []
    print(f"listening on :{a.port}")
    while True:
        for fid, st in rx.poll():
            ab = assemble(st, fid, hb[-1] if hb else None)
            ap_ = assemble(st, fid, hp[-1] if hp else None)
            rx.report_missing(ap_.missing)
            m = tile_mask_to_pixels(ap_.missing)
            ob = base(ab.image, m, hb) if ab.missing else ab.image
            op = ph(ap_.image, m, hp) if ap_.missing else ap_.image
            for h, o in ((hb, ob), (hp, op)):
                h.append(o)
                if len(h) > 3:
                    h.pop(0)
            arr = ap_.image.copy()
            arr[m] = (255, 0, 255)
            cv2.imshow("Phantom Frames receiver", np.hstack([
                label(arr, f"Arrived ({len(ap_.missing)} tiles lost)"), label(ob, "Baseline: copy-tile"),
                label(op, "Phantom")]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
