"""Standalone live sender.   python -m phantom.live.sender --to 127.0.0.1:5000 --src 0"""
import argparse
import time

from .endpoints import LiveSender


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", default="127.0.0.1:5000")
    ap.add_argument("--src", default="0", help="video file or webcam index")
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--no-cr", action="store_true")
    a = ap.parse_args()
    host, port = a.to.split(":")
    tx = LiveSender(a.src, (host, int(port)), quality=a.quality, cr_threshold=0 if a.no_cr else 2.0).start()
    print(f"sending {a.src} to {a.to} (Ctrl+C to stop)", flush=True)
    last, t = 0, time.time()
    try:
        while True:
            time.sleep(2)
            now = time.time()
            print(f"frame {tx.pk.frame_id}  {(tx.bytes_sent - last) * 8 / (now - t) / 1e6:.2f} Mbit/s  "
                  f"receiver-reported loss {tx.reported_loss:.1%}", flush=True)
            last, t = tx.bytes_sent, now
    except KeyboardInterrupt:
        tx.stop()


if __name__ == "__main__":
    main()
