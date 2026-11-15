"""Record a webcam clip for training / testing (talking-head video, like a real video call).

    python -m tools.record_clip --out data/train/me_01.mp4 --seconds 60
Press q to stop early. Move around, talk, gesture, change lighting a bit: variety helps training.
"""
import argparse
import time
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--cam", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, img = cap.read()
    if not ok:
        raise SystemExit("cannot read webcam")
    h, w = img.shape[:2]
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    t0 = time.time()
    n = 0
    while time.time() - t0 < a.seconds:
        ok, img = cap.read()
        if not ok:
            break
        wr.write(img)
        n += 1
        show = img.copy()
        cv2.putText(show, f"REC {time.time() - t0:4.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("recording (q to stop)", show)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    wr.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"saved {n} frames to {a.out}")


if __name__ == "__main__":
    main()
