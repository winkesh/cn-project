"""RepairNet: a small mask-conditioned U-Net that synthesises lost tiles.

Inputs (10 channels, all in [0,1]):
    prev2  (3)  receiver output two frames ago
    prev1  (3)  receiver output one frame ago
    partial(3)  current frame: received tiles, lost tiles pre-filled by copying prev1
    mask   (1)  1 where the tile was lost
Output: the repaired frame. Received pixels are always kept exactly (hard composite), so the
network only ever "invents" pixels that the network actually lost.

It does extrapolation (whole frame lost -> predict from motion in prev2/prev1) and region repair
(some tiles lost -> use the received neighbours as well) with one model, and adds zero latency
because it never waits for future frames.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GELU(),
    )


class RepairNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1 = block(10, c[0])
        self.e2 = block(c[0], c[1])
        self.e3 = block(c[1], c[2])
        self.mid = block(c[2], c[3])
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.d3 = block(c[2] * 2, c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.d2 = block(c[1] * 2, c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.d1 = block(c[0] * 2, c[0])
        self.out = nn.Conv2d(c[0], 3, 1)
        nn.init.zeros_(self.out.weight)   # start exactly at the copy-tile baseline
        nn.init.zeros_(self.out.bias)

    def forward(self, prev2, prev1, partial, mask):
        x = torch.cat([prev2, prev1, partial, mask], 1)
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        res = self.out(d1)[..., :h, :w]
        pred = (partial + res).clamp(0, 1)
        return mask * pred + (1 - mask) * partial


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
