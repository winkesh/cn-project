"""Frame interpolation network from our framegen project (IFNet/RIFE-style coarse-to-fine flow).

Vendored from framegen/train.py (model definition only) so Phantom Frames can load framegen checkpoints
without the framegen repo. Given frames i0 and i1 it predicts the frame halfway between them.
Trained on Vimeo-90K, fine-tuned on game footage and on scene cuts (ckpt_cut_w0.75_leaky_d4_same).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(img, flow):
    # backward warp: out(x) = img(x + flow(x)), flow in pixels; samples in fp32, returns img's dtype
    dtype, img, flow = img.dtype, img.float(), flow.float()
    _, _, H, W = img.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=img.device, dtype=torch.float32),
                            torch.arange(W, device=img.device, dtype=torch.float32), indexing="ij")
    x, y = gx + flow[:, 0], gy + flow[:, 1]
    grid = torch.stack((2 * x / (W - 1) - 1, 2 * y / (H - 1) - 1), -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True).to(dtype)


def resize(x, f):
    return x if f == 1 else F.interpolate(x, scale_factor=f, mode="bilinear", align_corners=False)


def conv(cin, cout, stride=1, act="prelu"):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.PReLU(cout) if act == "prelu" else nn.LeakyReLU(0.1))


class Block(nn.Module):
    # works at 1/4 of its input res; predicts flow t->0, t->1 (4ch) + blend-mask logit (1ch)
    def __init__(self, cin, c, act="prelu", depth=6):
        super().__init__()
        self.down = nn.Sequential(conv(cin, c // 2, 2, act), conv(c // 2, c, 2, act))
        self.body = nn.Sequential(*[conv(c, c, 1, act) for _ in range(depth)])
        self.up = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x):
        h = self.down(x)
        h = self.body(h) + h
        return resize(self.up(h), 2)


class Net(nn.Module):
    # 3 blocks, coarse (1/4 res) to fine (full res); each refines the previous flow. H, W % 16 == 0.
    scales = (4, 2, 1)

    def __init__(self, width=1.0, act="prelu", depth=6):
        super().__init__()
        ch = lambda c: int(c * width) // 16 * 16
        self.blocks = nn.ModuleList([Block(6, ch(192), act, depth), Block(17, ch(128), act, depth),
                                     Block(17, ch(96), act, depth)])

    def forward(self, i0, i1, scale=1.0):
        full0, full1 = i0, i1
        if scale != 1:
            i0, i1 = resize(i0, scale), resize(i1, scale)
        outs, flow, mask = self.coarse_to_fine(i0, i1)
        if scale != 1:
            flow, mask = resize(flow, 1 / scale) / scale, resize(mask, 1 / scale)
            m = torch.sigmoid(mask.float())
            outs.append(m * warp(full0, flow[:, :2]) + (1 - m) * warp(full1, flow[:, 2:]))
        return outs

    def coarse_to_fine(self, i0, i1):
        outs, flow, mask = [], None, None
        for s, blk in zip(self.scales, self.blocks):
            x = torch.cat((i0, i1), 1) if flow is None else torch.cat((i0, i1, w0, w1, flow / s, mask), 1)
            d = resize(blk(resize(x, 1 / s)), s)
            df, dm = d[:, :4] * s, d[:, 4:5]
            flow = df if flow is None else flow + df
            mask = dm if mask is None else mask + dm
            w0, w1 = warp(i0, flow[:, :2]), warp(i1, flow[:, 2:])
            m = torch.sigmoid(mask.float())
            outs.append(m * w0 + (1 - m) * w1)
        return outs, flow, mask


def load_net(path, device="cuda"):
    sd = torch.load(path, map_location=device)
    if "net" in sd and isinstance(sd["net"], dict):
        sd = sd["net"]
    act = "prelu" if "blocks.0.down.0.1.weight" in sd else "leaky"
    depth = len({k.split(".")[3] for k in sd if k.startswith("blocks.0.body.")})
    net = Net(sd["blocks.0.down.0.0.weight"].shape[0] * 2 / 192, act, depth).to(device)
    net.load_state_dict(sd)
    return net.eval()
