"""Network emulation: loss models plus a one-way link with delay, jitter and an optional bottleneck.

Used both by the trace-driven simulator (phantom.sim) and the live UDP relay (phantom.live.relay),
so offline numbers and the live demo share exactly the same loss behaviour.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


class LossModel:
    def lost(self) -> bool:
        raise NotImplementedError


class NoLoss(LossModel):
    def lost(self) -> bool:
        return False


class Bernoulli(LossModel):
    def __init__(self, p: float, rng: random.Random):
        self.p, self.rng = p, rng

    def lost(self) -> bool:
        return self.rng.random() < self.p


class GilbertElliott(LossModel):
    """Two-state Markov loss (same parameterisation as netem `loss gemodel p r 1-h 1-k`).

    p: P(good -> bad), r: P(bad -> good), loss_bad: loss prob in bad state, loss_good: in good.
    Mean loss = (p*loss_bad + r*loss_good) / (p + r); mean burst length ~ 1/r when loss_bad=1.
    """

    def __init__(self, p: float, r: float, rng: random.Random, loss_bad: float = 1.0, loss_good: float = 0.0):
        self.p, self.r, self.rng = p, r, rng
        self.loss_bad, self.loss_good = loss_bad, loss_good
        self.bad = False

    def lost(self) -> bool:
        if self.bad:
            if self.rng.random() < self.r:
                self.bad = False
        elif self.rng.random() < self.p:
            self.bad = True
        return self.rng.random() < (self.loss_bad if self.bad else self.loss_good)

    @staticmethod
    def for_mean(mean_loss: float, burst_len: float, rng: random.Random) -> "GilbertElliott":
        """Pick p, r so that average loss = mean_loss and mean burst length = burst_len packets."""
        r = 1.0 / burst_len
        p = mean_loss * r / max(1e-9, 1.0 - mean_loss)
        return GilbertElliott(p, r, rng)


def make_loss(kind: str, loss: float, burst: float, seed: int) -> LossModel:
    rng = random.Random(seed)
    if loss <= 0:
        return NoLoss()
    if kind == "bernoulli":
        return Bernoulli(loss, rng)
    if kind == "ge":
        return GilbertElliott.for_mean(loss, burst, rng)
    raise ValueError(kind)


@dataclass
class Link:
    """One-way link. All times in milliseconds."""
    delay_ms: float = 25.0
    jitter_ms: float = 0.0
    rate_mbps: float = 0.0        # 0 = unlimited
    loss: LossModel = None
    seed: int = 0

    def __post_init__(self):
        self.loss = self.loss or NoLoss()
        self._rng = random.Random(self.seed + 7919)
        self._busy_until = 0.0
        self._last_arrival = 0.0

    def send(self, t_ms: float, size_bytes: int) -> float | None:
        """Return arrival time, or None if the packet is dropped."""
        if self.rate_mbps > 0:
            start = max(t_ms, self._busy_until)
            self._busy_until = start + size_bytes * 8 / (self.rate_mbps * 1000.0)
            t_ms = self._busy_until
        if self.loss.lost():
            return None
        j = self._rng.uniform(-self.jitter_ms, self.jitter_ms) if self.jitter_ms else 0.0
        arr = t_ms + self.delay_ms + j
        # keep FIFO order (netem default with jitter can reorder; we don't, to keep NACK logic simple)
        arr = max(arr, self._last_arrival)
        self._last_arrival = arr
        return arr
