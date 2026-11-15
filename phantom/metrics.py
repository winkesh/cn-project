"""Quality and timing metrics."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return min(99.0, float(cv2.PSNR(a, b)))   # OpenCV: same formula, ~10x faster than numpy


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM on luma with the standard 11x11 Gaussian window (sigma 1.5), Wang et al. 2004.
    ~40x faster than skimage's RGB version; values match skimage's gaussian_weights=True on Y."""
    x = cv2.cvtColor(a, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    y = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mx, my = g(x), g(y)
    sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
    return float(s[5:-5, 5:-5].mean())


def ssim_skimage(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    d = (a.astype(np.float32) - b.astype(np.float32))[mask]
    mse = np.mean(d ** 2)
    return 99.0 if mse < 1e-10 else float(10 * np.log10(255.0 ** 2 / mse))


def stall_stats(frozen: list[bool], frame_ms: float, min_stall_ms: float = 200.0) -> tuple[int, float]:
    """Count runs of consecutive frozen frames that last >= min_stall_ms. Returns (count, total ms)."""
    count, total, run = 0, 0.0, 0
    for f in frozen + [False]:
        if f:
            run += 1
        else:
            if run * frame_ms >= min_stall_ms:
                count += 1
                total += run * frame_ms
            run = 0
    return count, total
