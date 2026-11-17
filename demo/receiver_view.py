"""Receiver window for the two-machine demo.

Machine A (sender):    python -m phantom.live.relay --listen 5000 --to <B-ip>:5004 --loss 0.10
                       python -m phantom.live.sender --to 127.0.0.1:5000 --src <video file or 0 for webcam>
Machine B (receiver):  python -m demo.receiver_view --port 5004

Shows two panels from the same received packets: left = without Phantom, right = with Phantom.
Keys:  c  left side: no repair -> normal decoder (freeze) -> copy last tile
       a  show which tiles were lost (outline)       f  full screen       q  quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom import conceal as C  # noqa: E402
from phantom.config import HEIGHT, N_TILES, WIDTH  # noqa: E402
from phantom.live.endpoints import LiveReceiver  # noqa: E402
from phantom.receiver import assemble  # noqa: E402
from phantom.tiles import tile_mask_to_pixels  # noqa: E402

GREEN = (0, 185, 118)   # #76b900 in BGR


def tag(img, text, accent):
    """NVIDIA-style micro label: bold uppercase in a bordered black box."""
    img = img.copy()
    s = 0.5 if WIDTH <= 640 else 0.8
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    x0, y0, pad = 12, 12, 9
    cv2.rectangle(img, (x0, y0), (x0 + tw + 2 * pad, y0 + th + 2 * pad), (0, 0, 0), -1)
    cv2.rectangle(img, (x0, y0), (x0 + tw + 2 * pad, y0 + th + 2 * pad), GREEN if accent else (94, 94, 94),
                  2 if accent else 1)
    cv2.putText(img, text, (x0 + pad, y0 + pad + th), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5004)
    ap.add_argument("--deadline", type=float, default=150.0, help="playout buffer in ms after the first packet")
    ap.add_argument("--weights", default="weights/repairnet.pt")
    ap.add_argument("--model", default="repairnet", choices=["repairnet", "flow"],
                    help="flow: classic optical-flow repair for a receiver without a GPU")
    a = ap.parse_args()

    rx = LiveReceiver(a.port, deadline_ms=a.deadline)
    if a.model == "repairnet" and Path(a.weights).exists():
        ph, pname = C.RepairNetConcealer(a.weights), "REPAIRNET"
        if ph.device != "cuda":
            print("[warn] no CUDA GPU found: RepairNet on the CPU will be slow; try --model flow")
    else:
        ph, pname = C.FlowExtrapolate(), "OPTICAL FLOW"
    lefts = [(C.NoRecovery(), "NO REPAIR"), (C.Freeze(), "NORMAL DECODER"), (C.CopyTile(), "COPY LAST TILE")]
    li, outline, full = 0, False, False
    hb, hp = [], []
    win = "Phantom Frames receiver"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(2 * WIDTH, 1600), int(min(2 * WIDTH, 1600) * HEIGHT / (2 * WIDTH)) + 36)
    blank = np.zeros((HEIGHT + 36, 2 * WIDTH, 3), np.uint8)
    cv2.putText(blank, f"Waiting for packets on UDP port {a.port} ({WIDTH}x{HEIGHT})...", (30, HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.imshow(win, blank)
    print(f"receiver listening on UDP :{a.port} at {WIDTH}x{HEIGHT}, Phantom = {pname}   (q to quit)", flush=True)

    t_rep, n_frames, lost_tiles, warned = time.time(), 0, 0, False
    while True:
        for fid, st in rx.poll():
            ab = assemble(st, fid, hb[-1] if hb else None)
            ap_ = assemble(st, fid, hp[-1] if hp else None)
            rx.report_missing(ap_.missing)
            m = tile_mask_to_pixels(ap_.missing)
            conc, lname = lefts[li]
            ob = conc(ab.image, m, hb) if ab.missing else ab.image
            op = ph(ap_.image, m, hp) if ap_.missing else ap_.image
            for h, o in ((hb, ob), (hp, op)):
                h.append(o)
                if len(h) > 3:
                    h.pop(0)
            left = ob
            if outline and ap_.missing:
                left = ob.copy()
                edge = m & ~cv2.erode(m.view(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                left[edge] = (0, 0, 255)
            frame = np.hstack([tag(left, f"WITHOUT PHANTOM - {lname}", False), tag(op, f"WITH PHANTOM - {pname}", True)])
            bar = np.zeros((36, frame.shape[1], 3), np.uint8)
            n_frames += 1
            lost_tiles += len(ap_.missing)
            cv2.putText(bar, f"{len(ap_.missing)} of {N_TILES} tiles lost this frame   "
                             f"[c] left side  [a] outline lost tiles  [f] full screen  [q] quit",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow(win, np.vstack([frame, bar]))
        if time.time() - t_rep > 2:
            if n_frames:
                print(f"{n_frames / (time.time() - t_rep):.0f} fps, {lost_tiles / max(1, n_frames):.0f} tiles lost "
                      f"per frame on average", flush=True)
            if rx.bad and not n_frames and not warned:
                print(f"[warn] {rx.bad} packets could not be decoded: start sender and receiver with the same "
                      f"PHANTOM_RES (this receiver uses {HEIGHT}p)", flush=True)
                warned = True
            t_rep, n_frames, lost_tiles = time.time(), 0, 0
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c"):
            li = (li + 1) % len(lefts)
        elif k == ord("a"):
            outline = not outline
        elif k == ord("f"):
            full = not full
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_AUTOSIZE) < 0:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
