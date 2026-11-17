"""Tile grid, per-tile JPEG coding and loss masks."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .config import HEIGHT, N_TILES, TILE, TILES_X, TILES_Y, WIDTH


def tile_rect(tid: int) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1) of a tile."""
    ty, tx = divmod(tid, TILES_X)
    return ty * TILE, (ty + 1) * TILE, tx * TILE, (tx + 1) * TILE


def split(frame: np.ndarray) -> list[np.ndarray]:
    assert frame.shape[:2] == (HEIGHT, WIDTH), frame.shape
    t = frame.reshape(TILES_Y, TILE, TILES_X, TILE, 3).swapaxes(1, 2)
    return [t[i // TILES_X, i % TILES_X] for i in range(N_TILES)]


def encode_tile(tile: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def decode_tile_raw(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.shape != (TILE, TILE, 3):
        return None
    return img


@lru_cache(maxsize=60000)
def _decode_cached(data: bytes) -> np.ndarray | None:
    img = decode_tile_raw(data)
    if img is not None:
        img.setflags(write=False)
    return img


def decode_tile(data: bytes) -> np.ndarray | None:
    """JPEG-decode one tile (memoised: the simulator decodes the same tiles in many runs)."""
    return _decode_cached(bytes(data))


def paste(frame: np.ndarray, tid: int, tile: np.ndarray) -> None:
    y0, y1, x0, x1 = tile_rect(tid)
    frame[y0:y1, x0:x1] = tile


def tile_mask_to_pixels(missing: set[int] | np.ndarray) -> np.ndarray:
    """Boolean HxW mask, True where pixels are missing."""
    m = np.zeros((TILES_Y, TILES_X), bool)
    for tid in missing:
        m[tid // TILES_X, tid % TILES_X] = True
    return np.kron(m, np.ones((TILE, TILE), bool))


def resize_frame(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def letterbox(img: np.ndarray) -> np.ndarray:
    """Fit any input into WIDTHxHEIGHT without distortion (black bars), e.g. a 4:3 webcam."""
    h, w = img.shape[:2]
    if (w, h) == (WIDTH, HEIGHT):
        return img
    s = min(WIDTH / w, HEIGHT / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    y, x = (HEIGHT - nh) // 2, (WIDTH - nw) // 2
    out[y:y + nh, x:x + nw] = small
    return out
