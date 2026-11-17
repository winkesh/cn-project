"""Receiver machine: one command.

    python -m phantom.receive                 # 10 % simulated loss, RepairNet
    python -m phantom.receive --loss 0.2      # other start loss; change it live with + / -

Waits for a sender (python -m phantom.send on the other machine finds it automatically), shows the stream
twice from the same packets: left without Phantom, right with Phantom. Packet loss is simulated as packets
arrive here, on top of whatever the real network loses.
Keys:  + / -  loss 5 %     b  random <-> bursty loss     c  left side mode     a  outline lost tiles
       f  full screen      q  quit
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import conceal as C
from .channel import make_loss
from .config import HEIGHT, N_TILES, WIDTH
from .live.endpoints import LiveReceiver, local_ips
from .receiver import assemble
from .tiles import tile_mask_to_pixels

GREEN = (0, 185, 118)   # #76b900 in BGR


def tag(img, text, accent):
    img = img.copy()
    s = 0.5 if WIDTH <= 640 else 0.8
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    x0, y0, pad = 12, 12, 9
    cv2.rectangle(img, (x0, y0), (x0 + tw + 2 * pad, y0 + th + 2 * pad), (0, 0, 0), -1)
    cv2.rectangle(img, (x0, y0), (x0 + tw + 2 * pad, y0 + th + 2 * pad), GREEN if accent else (94, 94, 94),
                  2 if accent else 1)
    cv2.putText(img, text, (x0 + pad, y0 + pad + th), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def screen(lines, w=2 * WIDTH, h=HEIGHT + 40):
    img = np.zeros((h, w, 3), np.uint8)
    y = h // 2 - 20 * len(lines)
    for i, (text, scale, col) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(img, text, ((w - tw) // 2, y + i * 44), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Phantom Frames receiver")
    ap.add_argument("--port", type=int, default=5004)
    ap.add_argument("--loss", type=float, default=0.10, help="simulated packet loss at the start (0-0.6)")
    ap.add_argument("--deadline", type=float, default=150.0, help="playout buffer in ms")
    ap.add_argument("--weights", default="weights/repairnet.pt")
    ap.add_argument("--model", default="repairnet", choices=["repairnet", "flow"],
                    help="flow: classic optical-flow repair for a machine without a GPU")
    a = ap.parse_args()

    try:
        rx = LiveReceiver(a.port, deadline_ms=a.deadline)
    except OSError as e:
        sys.exit(f"cannot listen on UDP port {a.port} ({e}). Is another receiver already running?")
    loss, bursty = min(0.6, max(0.0, a.loss)), False

    def set_loss():
        rx.loss_model = make_loss("ge" if bursty else "bernoulli", loss, 4.0, int(time.time()))
    set_loss()

    if a.model == "repairnet" and Path(a.weights).exists():
        ph, pname = C.RepairNetConcealer(a.weights), "REPAIRNET"
        if ph.device != "cuda":
            print("[warn] no CUDA GPU: RepairNet on the CPU will be slow; use --model flow")
    else:
        if a.model == "repairnet":
            print(f"[warn] {a.weights} not found (run from the project folder); using optical flow")
        ph, pname = C.FlowExtrapolate(), "OPTICAL FLOW"
    lefts = [(C.NoRecovery(), "NO REPAIR"), (C.Freeze(), "NORMAL DECODER"), (C.CopyTile(), "COPY LAST TILE")]
    li, outline, full = 0, False, False

    ips = local_ips()
    print("\nPhantom Frames receiver is ready.")
    print("On the sender machine run:   python -m phantom.send")
    if ips:
        print(f"If it cannot find this one:  python -m phantom.send --to {ips[0]}")
    print(f"(listening on UDP {a.port}; this machine: {', '.join(ips) or 'no network address found'})\n", flush=True)

    win = "Phantom Frames receiver"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(2 * WIDTH, 1600), int(min(2 * WIDTH, 1600) * (HEIGHT + 40) / (2 * WIDTH)))
    waiting = screen([("Waiting for the sender", 1.1, (255, 255, 255)),
                      ("On the other machine run:  python -m phantom.send", 0.75, (200, 200, 200)),
                      (f"or:  python -m phantom.send --to {ips[0] if ips else '<this IP>'}", 0.75, (150, 150, 150))])
    hb, hp = [], []
    last_frame_t = 0.0
    t_rep, n_frames = time.time(), 0
    while True:
        frames = rx.poll()
        for fid, st in frames:
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
            bar = np.zeros((40, frame.shape[1], 3), np.uint8)
            meas = rx.sim_dropped / max(1, rx.arrived)
            cv2.putText(bar, f"simulated loss {loss:.0%} ({'bursty' if bursty else 'random'})   "
                             f"{len(ap_.missing)}/{N_TILES} tiles lost   sender {rx.peer[0] if rx.peer else '-'}   "
                             f"[+/-] loss [b] burst [c] left [a] outline [f] full [q] quit",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow(win, np.vstack([frame, bar]))
            last_frame_t = time.time()
            n_frames += 1
        if not frames and time.time() - last_frame_t > 2:
            cv2.imshow(win, waiting)
            last_frame_t = time.time() - 1.5          # redraw the waiting screen ~every 0.5 s
        if time.time() - t_rep > 3:
            if n_frames:
                print(f"receiving {n_frames / (time.time() - t_rep):.0f} fps from {rx.peer[0] if rx.peer else '?'}, "
                      f"simulated loss {loss:.0%}", flush=True)
            elif rx.bad:
                print(f"[warn] {rx.bad} packets could not be decoded: run sender and receiver with the same "
                      f"PHANTOM_RES (this receiver: {HEIGHT}p)", flush=True)
            t_rep, n_frames = time.time(), 0
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in (ord("+"), ord("=")):
            loss = min(0.6, round(loss + 0.05, 2)); set_loss()
        elif k in (ord("-"), ord("_")):
            loss = max(0.0, round(loss - 0.05, 2)); set_loss()
        elif k == ord("b"):
            bursty = not bursty; set_loss()
        elif k == ord("c"):
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
