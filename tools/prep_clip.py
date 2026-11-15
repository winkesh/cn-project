"""Convert any local video into a 640x360 test/train clip.

    python -m tools.prep_clip --src ~/framegen/clips_laptop/forza.mp4 --out data/test/game_forza.avi \
        --stride 2 --start 600 --frames 300
Clips whose name starts with game_ are reported as content type "game".
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom.tiles import resize_frame  # noqa: E402
from tools.fetch_clips import writer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    cap = cv2.VideoCapture(str(Path(a.src).expanduser()))
    if not cap.isOpened():
        sys.exit(f"cannot open {a.src}")
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30) / a.stride
    for _ in range(a.start):          # decode up to the start (seeking in H.264 corrupts the first frames)
        if not cap.grab():
            break
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wr = writer(Path(a.out), fps)
    i = n = 0
    while n < a.frames:
        ok, img = cap.read()
        if not ok:
            break
        if i % a.stride == 0:
            wr.write(resize_frame(img))
            n += 1
        i += 1
    wr.release()
    print(f"{a.src} -> {a.out}: {n} frames at {fps:.0f} fps")


if __name__ == "__main__":
    main()
