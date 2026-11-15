"""Bidirectional UDP relay that emulates a lossy link (runs natively on Windows, no admin rights).

Media (sender -> receiver) gets loss / delay / jitter; feedback (receiver -> sender) is forwarded clean.

    python -m phantom.live.relay --listen 5000 --to 127.0.0.1:5004 --loss 0.1 --model ge --burst 4 --delay 25
"""
from __future__ import annotations

import argparse
import heapq
import socket
import threading
import time

from ..channel import make_loss


class Relay:
    def __init__(self, listen: int, to: tuple[str, int], loss=0.0, model="bernoulli", burst=3.0,
                 delay_ms=0.0, jitter_ms=0.0, seed=0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        self.sock.bind(("0.0.0.0", listen))
        self.sock.settimeout(0.002)
        self.to = to
        self.sender_addr = None
        self.delay_ms, self.jitter_ms = delay_ms, jitter_ms
        self.seed = seed
        self.set_loss(loss, model, burst)
        self.fwd = self.dropped = 0
        self._q: list = []
        self._stop = threading.Event()
        import random
        self._rng = random.Random(seed + 1)

    def set_loss(self, loss: float, model: str = None, burst: float = None):
        self.loss = max(0.0, min(0.95, loss))
        self.model = model or getattr(self, "model", "bernoulli")
        self.burst = burst or getattr(self, "burst", 3.0)
        self.seed += 1
        self.loss_model = make_loss(self.model, self.loss, self.burst, self.seed)

    def run(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(65535)
                now = time.perf_counter()
                if addr[0] == self.to[0] and addr[1] == self.to[1]:
                    if self.sender_addr:          # feedback path: clean
                        self.sock.sendto(data, self.sender_addr)
                else:
                    self.sender_addr = addr
                    if self.loss_model.lost():
                        self.dropped += 1
                    else:
                        j = self._rng.uniform(-self.jitter_ms, self.jitter_ms) if self.jitter_ms else 0
                        heapq.heappush(self._q, (now + (self.delay_ms + j) / 1000, id(data), data))
            except (socket.timeout, ConnectionResetError):
                pass
            now = time.perf_counter()
            while self._q and self._q[0][0] <= now:
                _, _, d = heapq.heappop(self._q)
                self.sock.sendto(d, self.to)
                self.fwd += 1

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=5000)
    ap.add_argument("--to", default="127.0.0.1:5004")
    ap.add_argument("--loss", type=float, default=0.05)
    ap.add_argument("--model", default="bernoulli", choices=["bernoulli", "ge"])
    ap.add_argument("--burst", type=float, default=3.0)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    a = ap.parse_args()
    host, port = a.to.split(":")
    r = Relay(a.listen, (host, int(port)), a.loss, a.model, a.burst, a.delay, a.jitter).start()
    print(f"relay :{a.listen} -> {a.to}  loss={a.loss:.0%} ({a.model}, burst {a.burst})  delay={a.delay} ms")
    try:
        while True:
            time.sleep(2)
            tot = r.fwd + r.dropped
            print(f"forwarded {r.fwd}  dropped {r.dropped}  ({100 * r.dropped / max(1, tot):.1f}%)")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
