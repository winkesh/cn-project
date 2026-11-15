"""Video loading plus a synthetic test pattern for smoke tests."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import HEIGHT, WIDTH
from .tiles import resize_frame

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".y4m", ".m4v"}


def load_video(path: str | Path, max_frames: int | None = None, start: int = 0, stride: int = 1) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i >= start and (i - start) % stride == 0:
            frames.append(resize_frame(img))
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames


def list_videos(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_EXT)


def synthetic(n: int = 150, seed: int = 0) -> list[np.ndarray]:
    """Textured background with moving shapes and a panning camera."""
    rng = np.random.default_rng(seed)
    big = (rng.random((HEIGHT * 2, WIDTH * 2, 3)) * 255).astype(np.uint8)
    big = cv2.GaussianBlur(big, (0, 0), 6)
    big = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
    for _ in range(60):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(big, tuple(int(v) for v in rng.integers(0, WIDTH * 2, 2)), int(rng.integers(10, 60)), c, -1)
    frames = []
    for t in range(n):
        x0, y0 = int(2 * t) % WIDTH, int(t) % HEIGHT
        f = big[y0:y0 + HEIGHT, x0:x0 + WIDTH].copy()
        cx = int(WIDTH / 2 + 200 * np.sin(t / 15))
        cy = int(HEIGHT / 2 + 100 * np.cos(t / 11))
        cv2.rectangle(f, (cx - 40, cy - 30), (cx + 40, cy + 30), (30, 30, 230), -1)
        cv2.putText(f, f"{t:04d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        frames.append(f)
    return frames
