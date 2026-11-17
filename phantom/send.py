"""Sender machine: one command.

    python -m phantom.send                    # webcam, finds the receiver on this network by itself
    python -m phantom.send --src clip.mp4     # a video file (loops)
    python -m phantom.send --to 192.168.1.42  # skip discovery: send straight to that receiver

Start the web demo on the receiving machine first (python -m phantom.app); it listens for senders.
"""
from __future__ import annotations

import argparse
import ipaddress
import socket
import sys
import time

from .live.endpoints import ANNOUNCE, DISCOVER, LiveSender, local_ips


def discover(port: int, tries: int = 4, wait: float = 1.0) -> str | None:
    """Broadcast 'is there a receiver?' on every local network; return the first receiver's IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.settimeout(wait)
    targets = {"255.255.255.255"}
    for ip in local_ips():          # subnet broadcasts too (assume /24; covers home Wi-Fi and phone hotspots)
        targets.add(str(ipaddress.ip_network(f"{ip}/24", strict=False).broadcast_address))
    try:
        for _ in range(tries):
            for t in targets:
                try:
                    s.sendto(DISCOVER, (t, port))
                except OSError:
                    pass
            t_end = time.time() + wait
            while time.time() < t_end:
                try:
                    data, addr = s.recvfrom(64)
                except (socket.timeout, ConnectionResetError, OSError):
                    break
                if data == ANNOUNCE:
                    return addr[0]
    finally:
        s.close()
    return None


def main():
    ap = argparse.ArgumentParser(description="Phantom Frames sender")
    ap.add_argument("--to", help="receiver IP (or IP:port); found automatically when left out")
    ap.add_argument("--src", default="0", help="webcam index (0 = first camera) or a video file")
    ap.add_argument("--port", type=int, default=5004)
    ap.add_argument("--quality", type=int, default=75)
    a = ap.parse_args()

    if a.to:
        host, _, port = a.to.partition(":")
        dst = (socket.gethostbyname(host), int(port or a.port))
    else:
        print("Looking for a receiver on this network...", flush=True)
        ip = discover(a.port)
        if ip is None:
            sys.exit("No receiver found. Start the web demo (python -m phantom.app) on the receiving machine, or give its "
                     "address:  python -m phantom.send --to <receiver-ip>  (it prints its IP when it starts).")
        dst = (ip, a.port)
        print(f"Found receiver at {ip}", flush=True)

    try:
        tx = LiveSender(a.src, dst, quality=a.quality).start()
    except RuntimeError as e:
        sys.exit(f"{e}. For the webcam, close other apps using it (Teams, Zoom, Camera) and allow desktop apps "
                 f"to use the camera in Windows privacy settings; or try --src 1.")
    what = f"webcam {a.src}" if str(a.src).isdigit() else a.src
    print(f"Streaming {what} to {dst[0]}:{dst[1]}   (Ctrl+C to stop)", flush=True)
    last, t = 0, time.time()
    try:
        while True:
            time.sleep(2)
            now = time.time()
            rate = (tx.bytes_sent - last) * 8 / (now - t) / 1e6
            fb = (f"receiver sees {tx.reported_loss:.0%} packet loss" if now - tx.last_feedback < 2
                  else "no reply from the receiver (is it running? firewall?)")
            print(f"frame {tx.pk.frame_id:6d}   {rate:5.1f} Mbit/s   {fb}", flush=True)
            last, t = tx.bytes_sent, now
    except KeyboardInterrupt:
        tx.stop()


if __name__ == "__main__":
    main()
