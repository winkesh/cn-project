"""Loss concealment strategies, from classic to learned.

Every concealer gets the same inputs:
    partial  HxWx3 uint8: received tiles pasted over the previous output (so lost tiles = copy)
    mask     HxW bool: True where tiles were lost
    history  list of previous receiver outputs (most recent last)
and returns the frame to display.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class Concealer:
    name = "base"

    def __call__(self, partial, mask, history):
        raise NotImplementedError


class NoRecovery(Concealer):
    """Nothing repairs the picture: lost tiles are shown black."""
    name = "none"

    def __call__(self, partial, mask, history):
        out = partial.copy()
        out[mask] = 0
        return out


class Freeze(Concealer):
    """What a standard decoder does with a corrupt frame: keep showing the last good one."""
    name = "freeze"

    def __call__(self, partial, mask, history):
        if mask.any() and history:
            return history[-1].copy()
        return partial


class CopyTile(Concealer):
    """Replace only the lost tiles with the co-located tiles of the previous frame."""
    name = "copy"

    def __call__(self, partial, mask, history):
        return partial


class Inpaint(Concealer):
    """OpenCV Telea inpainting (spatial only)."""
    name = "inpaint"

    def __call__(self, partial, mask, history):
        if not mask.any():
            return partial
        if mask.mean() > 0.6:  # nothing to inpaint from; fall back to copy
            return partial
        return cv2.inpaint(partial, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)


class FlowExtrapolate(Concealer):
    """Classic motion-compensated concealment: Farneback flow prev2->prev1, push prev1 one step forward."""
    name = "flow"

    def __call__(self, partial, mask, history):
        if not mask.any() or len(history) < 2:
            return partial
        p2 = cv2.cvtColor(history[-2], cv2.COLOR_BGR2GRAY)
        p1 = cv2.cvtColor(history[-1], cv2.COLOR_BGR2GRAY)
        # flow from p1 back to p2: p1(x) ~ p2(x + f(x)); assume constant motion so next(x) ~ p1(x + f(x))
        flow = cv2.calcOpticalFlowFarneback(p1, p2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = p1.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pred = cv2.remap(history[-1], gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        out = partial.copy()
        out[mask] = pred[mask]
        return out


class RepairNetConcealer(Concealer):
    name = "repairnet"

    def __init__(self, weights: str | Path, device: str | None = None, half: bool = True,
                 work_height: int | None = None):
        import torch
        from .models.repairnet import RepairNet

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half and self.device == "cuda"
        ck = torch.load(weights, map_location="cpu")
        self.net = RepairNet(base=ck.get("base", 32))
        self.net.load_state_dict(ck["model"])
        self.net.eval().to(self.device)
        if self.half:
            self.net.half()
        self.fmt = torch.contiguous_format
        if self.device == "cuda":               # channels-last + cuDNN autotune: ~25 % faster on RTX GPUs
            torch.backends.cudnn.benchmark = True
            self.fmt = torch.channels_last
            self.net = self.net.to(memory_format=self.fmt)
        self.times_ms: list[float] = []
        self.work_height = work_height            # e.g. 360: repair at 360p and upscale into larger frames

    def _t(self, img):
        # upload uint8 BGR, convert to RGB float on the GPU (cheaper than doing it in numpy)
        t = self.torch.from_numpy(np.ascontiguousarray(img)).to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).flip(0)[None]
        t = t.half() if self.half else t.float()
        return (t / 255.0).contiguous(memory_format=self.fmt)

    def __call__(self, partial, mask, history):
        if not mask.any():
            return partial
        torch = self.torch
        prev1 = history[-1] if history else partial
        prev2 = history[-2] if len(history) > 1 else prev1
        t0 = time.perf_counter()
        with torch.no_grad():
            m = torch.from_numpy(mask.view(np.uint8))[None, None].to(self.device)
            m = (m.half() if self.half else m.float()).contiguous(memory_format=self.fmt)
            p2, p1, cur = self._t(prev2), self._t(prev1), self._t(partial)
            h, w = cur.shape[-2:]
            if self.work_height and h > self.work_height:
                # Low-res repair: run the network on a downscaled copy, upscale only its prediction and paste
                # it into the lost tiles. Cost stays at the work resolution whatever the stream resolution;
                # received pixels keep full resolution.
                F = torch.nn.functional
                sh, sw = self.work_height, round(w * self.work_height / h / 8) * 8
                small = lambda t: F.interpolate(t, size=(sh, sw), mode="area").contiguous(memory_format=self.fmt)
                ms = (F.interpolate(m, size=(sh, sw), mode="area") > 0).to(m.dtype).contiguous(memory_format=self.fmt)
                pred = self.net(small(p2), small(p1), small(cur), ms)
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
                out = m * pred + (1 - m) * cur
            else:
                out = self.net(p2, p1, cur, m)
            out = (out[0].flip(0).float().clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        self.times_ms.append((time.perf_counter() - t0) * 1000)   # .cpu() above already synchronised
        return out


class FramegenInterp(Concealer):
    """Phantom-Interp: wait one extra frame, then fill lost tiles of frame t by interpolating between the
    previous output (t-1) and the partially received frame t+1 with our framegen network.
    Costs +1 frame (33 ms) of latency; falls back to a zero-latency concealer if t+1 is unusable."""
    name = "framegen"
    needs_next = True

    def __init__(self, weights: str | Path = "weights/framegen.pt", fallback: Concealer | None = None,
                 device: str | None = None):
        import torch
        from .models.framegen_net import load_net

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_net(str(weights), self.device)
        self.half = self.device == "cuda"
        if self.half:
            self.net.half()
        self.fallback = fallback or FlowExtrapolate()
        self.times_ms: list[float] = []

    def _t(self, img):
        t = self.torch.from_numpy(np.ascontiguousarray(img[..., ::-1])).permute(2, 0, 1)[None].to(self.device)
        t = t.half() if self.half else t.float()
        return t / 255.0

    def __call__(self, partial, mask, history, nxt=None, nxt_missing_frac: float = 1.0):
        if not mask.any():
            return partial
        if nxt is None or nxt_missing_frac > 0.5 or not history:
            return self.fallback(partial, mask, history)
        torch, F = self.torch, self.torch.nn.functional
        t0 = time.perf_counter()
        with torch.no_grad():
            a, b = self._t(history[-1]), self._t(nxt)
            h, w = a.shape[-2:]
            ph, pw = (-h) % 16, (-w) % 16
            if ph or pw:
                a, b = F.pad(a, (0, pw, 0, ph), mode="replicate"), F.pad(b, (0, pw, 0, ph), mode="replicate")
            mid = self.net(a, b)[-1][..., :h, :w]
            mid = (mid[0].float().clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).cpu().numpy()[..., ::-1]
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.times_ms.append((time.perf_counter() - t0) * 1000)
        out = partial.copy()
        out[mask] = mid[mask]
        return out


def make(name: str, weights: str | None = None, framegen_weights: str = "weights/framegen.pt") -> Concealer:
    if name == "none":
        return NoRecovery()
    if name == "freeze":
        return Freeze()
    if name == "copy":
        return CopyTile()
    if name == "inpaint":
        return Inpaint()
    if name == "flow":
        return FlowExtrapolate()
    if name == "repairnet":
        return RepairNetConcealer(weights or "weights/repairnet.pt")
    if name == "framegen":
        fb = RepairNetConcealer(weights) if weights and Path(weights).exists() else FlowExtrapolate()
        return FramegenInterp(framegen_weights, fallback=fb)
    raise ValueError(name)
