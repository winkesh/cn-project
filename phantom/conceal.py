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
        np.copyto(out, 0, where=mask[..., None])   # much faster than boolean indexing on large frames
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
    """RepairNet. backend: 'torch' (PyTorch FP16), 'trt' (TensorRT engine file), 'dml' (ONNX Runtime +
    DirectML: any DirectX 12 GPU). work_height: run the network at this height and paste the upscaled
    prediction into the lost tiles only (cost independent of stream resolution). TensorRT and DirectML use
    fixed-size models, so a larger frame automatically goes through this low-res path."""
    name = "repairnet"

    def __init__(self, weights: str | Path, device: str | None = None, half: bool = True,
                 work_height: int | None = None, backend: str = "torch", model_file: str | Path | None = None,
                 dml_device: int = 0):
        import torch
        from .models.repairnet import RepairNet

        self.torch = torch
        self.backend = backend
        self.times_ms: list[float] = []
        self.work_height = work_height
        self.runner = None
        if backend == "dml":                     # no CUDA or PyTorch GPU needed
            from .accel import DMLRunner
            self.runner = DMLRunner(model_file, dml_device)
            self.device = "dml"
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if backend == "trt":
            from .accel import TRTRunner
            self.runner = TRTRunner(model_file)
            self.device = "cuda"
        self.half = half and self.device == "cuda" and backend == "torch"
        if self.runner is not None:                 # TensorRT keeps FP32 I/O
            self.fmt = torch.contiguous_format
            self.net = None
            return
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

    precision = ""

    @property
    def label(self) -> str:
        b = {"torch": "PyTorch", "trt": "TensorRT", "dml": "DirectML"}[self.backend]
        b += f" {self.precision}" if self.precision else ""
        from .config import HEIGHT
        h = self.runner.h if self.runner is not None else self.work_height
        return b + (f", repairs at {h}p" if h and h < HEIGHT else "")

    def _call_dml(self, partial, mask, history):
        """NumPy/OpenCV path for ONNX Runtime DirectML (works on GPUs without CUDA)."""
        import cv2
        prev1 = history[-1] if history else partial
        prev2 = history[-2] if len(history) > 1 else prev1
        t0 = time.perf_counter()
        h, w = partial.shape[:2]
        rh, rw = self.runner.h, self.runner.w
        resize = (h, w) != (rh, rw)
        small = (lambda im: cv2.resize(im, (rw, rh), interpolation=cv2.INTER_AREA)) if resize else (lambda im: im)
        nchw = lambda im: np.ascontiguousarray(small(im)[..., ::-1].transpose(2, 0, 1)[None], np.float32) / 255.0
        ms = (cv2.resize(mask.astype(np.float32), (rw, rh), interpolation=cv2.INTER_AREA) > 0) if resize else mask
        y = self.runner(nchw(prev2), nchw(prev1), nchw(partial), ms.astype(np.float32)[None, None])
        pred = np.clip(y[0].transpose(1, 2, 0)[..., ::-1] * 255 + 0.5, 0, 255).astype(np.uint8)
        if resize:
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        out = partial.copy()
        out[mask] = pred[mask]
        self.times_ms.append((time.perf_counter() - t0) * 1000)
        return out

    def _t(self, img):
        # upload uint8 BGR, convert to RGB float on the GPU (cheaper than doing it in numpy)
        t = self.torch.from_numpy(np.ascontiguousarray(img)).to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).flip(0)[None]
        t = t.half() if self.half else t.float()
        return (t / 255.0).contiguous(memory_format=self.fmt)

    def _call_trt_small(self, partial, mask, history):
        """TensorRT on a frame larger than its engine: shrink on the CPU and move only engine-sized tensors
        to and from the GPU (a 720p frame is 4x the data of the 360p the engine needs)."""
        import cv2
        torch = self.torch
        prev1 = history[-1] if history else partial
        prev2 = history[-2] if len(history) > 1 else prev1
        t0 = time.perf_counter()
        h, w = partial.shape[:2]
        rh, rw = self.runner.h, self.runner.w
        small = lambda im: cv2.resize(im, (rw, rh), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(mask.view(np.uint8), (rw, rh), interpolation=cv2.INTER_AREA) > 0
        with torch.no_grad():
            m = torch.from_numpy(ms.view(np.uint8))[None, None].to(self.device).float()
            y = self.runner(self._t(small(prev2)), self._t(small(prev1)), self._t(small(partial)), m)
            pred = (y[0].flip(0).float().clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        out = partial.copy()
        np.copyto(out, pred, where=mask[..., None])
        self.times_ms.append((time.perf_counter() - t0) * 1000)
        return out

    def __call__(self, partial, mask, history):
        if not mask.any():
            return partial
        if self.backend == "dml":
            return self._call_dml(partial, mask, history)
        if self.runner is not None and partial.shape[:2] != (self.runner.h, self.runner.w):
            return self._call_trt_small(partial, mask, history)
        torch = self.torch
        prev1 = history[-1] if history else partial
        prev2 = history[-2] if len(history) > 1 else prev1
        t0 = time.perf_counter()
        run = self.runner or self.net
        with torch.no_grad():
            m = torch.from_numpy(mask.view(np.uint8))[None, None].to(self.device)
            m = (m.half() if self.half else m.float()).contiguous(memory_format=self.fmt)
            p2, p1, cur = self._t(prev2), self._t(prev1), self._t(partial)
            h, w = cur.shape[-2:]
            if self.runner is not None:              # fixed-size engine
                sh, sw = self.runner.h, self.runner.w
            elif self.work_height and h > self.work_height:
                sh, sw = self.work_height, round(w * self.work_height / h / 8) * 8
            else:
                sh, sw = h, w
            if (sh, sw) != (h, w):
                # Low-res repair: run the network on a downscaled copy, upscale only its prediction and paste
                # it into the lost tiles. Cost stays at the work resolution whatever the stream resolution;
                # received pixels keep full resolution.
                F = torch.nn.functional
                small = lambda t: F.interpolate(t, size=(sh, sw), mode="area").contiguous(memory_format=self.fmt)
                ms = (F.interpolate(m, size=(sh, sw), mode="area") > 0).to(m.dtype).contiguous(memory_format=self.fmt)
                pred = run(small(p2), small(p1), small(cur), ms).to(cur.dtype)
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
                out = m * pred + (1 - m) * cur
            else:
                out = run(p2, p1, cur, m)
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


def make_repairnet(weights: str = "weights/repairnet.pt", backend: str = "auto", repair_res: str = "auto",
                   budget_ms: float = 25.0) -> RepairNetConcealer:
    """Pick the fastest working RepairNet backend on this machine and decide on low-res repair.

    backend: auto | torch | trt | dml.  auto = TensorRT if a verified engine exists for this GPU (precision from
    results/accel.json), else PyTorch on CUDA, else DirectML on any DirectX 12 GPU, else PyTorch on the CPU.
    repair_res: auto | full | 360.  auto = full resolution if it fits budget_ms per frame, else repair at 360p.
    """
    import json

    import torch
    from . import accel
    from .config import HEIGHT, WIDTH

    cuda = torch.cuda.is_available()
    cfg = {}
    try:
        cfg = json.load(open("results/accel.json"))
    except (OSError, ValueError):
        pass
    rec = cfg.get("recommended", "fp16")
    order = (rec, "int8fp16", "fp16", "int8")
    engines = ([accel.engine_path(360, 640, p) for p in order] +                                 # built here
               [accel.engine_path(360, 640, p, portable=True) for p in order]) if cuda else []
    engines = [e for e in dict.fromkeys(engines) if e.exists()]
    engine = engines[0] if engines else None
    onnx_file = accel.onnx_path(360, 640, variant=cfg.get("dml_variant", "fp32"))
    if not onnx_file.exists():
        onnx_file = accel.onnx_path(360, 640)
    dml_ok = False
    try:
        import onnxruntime as ort
        dml_ok = "DmlExecutionProvider" in ort.get_available_providers() and onnx_file.exists()
    except ImportError:
        pass
    explicit = backend
    if backend == "auto":
        backend = "trt" if engine else "torch" if cuda else "dml" if dml_ok else "torch"
    if backend == "trt":
        if engine is None:
            raise FileNotFoundError("no TensorRT engine for this GPU: run python -m tools.build_accel")
        for e in engines:                         # first engine that loads (version / GPU / platform must fit)
            try:
                c = RepairNetConcealer(weights, backend="trt", model_file=e)
                c.precision = {"int8fp16": "INT8+FP16", "int8": "INT8"}.get(e.name.split("_")[2], "FP16")
                return c                          # 360p engine: larger streams repair at 360p automatically
            except Exception as err:
                print(f"[info] TensorRT engine {e.name} not usable here: {err}")
        if explicit == "trt":
            raise RuntimeError("no TensorRT engine loads on this machine")
        backend = "torch" if cuda else "dml" if dml_ok else "torch"
    if backend == "dml":
        c = RepairNetConcealer(weights, backend="dml", model_file=onnx_file, dml_device=cfg.get("dml_device", 0))
        c.precision = "FP16" if "fp16" in onnx_file.name else "FP32"
        return c
    work = None if repair_res == "full" else 360 if repair_res == "360" else None
    c = RepairNetConcealer(weights, work_height=work)
    if repair_res == "auto" and HEIGHT > 360:     # time full-resolution repair once on this GPU
        z = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        m = np.zeros((HEIGHT, WIDTH), bool)
        m[::80, ::80] = True
        for _ in range(4):
            c(z, m, [z, z])
        if np.median(c.times_ms[-3:]) > budget_ms:
            c.work_height = 360
        c.times_ms.clear()
    return c


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
