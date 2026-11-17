"""Shared constants for Phantom Frames.

Resolution is chosen with the PHANTOM_RES environment variable: 360 (default, 640x360, 144 tiles,
what the evaluation uses), 720 (1280x720, 576 tiles) or 1080 (1920x1080, 1296 tiles).
"""
import os
from dataclasses import dataclass

RES = int(os.environ.get("PHANTOM_RES", "360"))
WIDTH, HEIGHT = {360: (640, 360), 720: (1280, 720), 1080: (1920, 1080)}[RES]
TILE = 40                      # 144 tiles at 640x360, 576 at 1280x720, 1296 at 1920x1080
TILES_X = WIDTH // TILE
TILES_Y = HEIGHT // TILE
N_TILES = TILES_X * TILES_Y
FPS = 30
FRAME_MS = 1000.0 / FPS
MAX_PAYLOAD = 1200             # bytes of tile data per packet (fits a 1500 B MTU)
RTP_PT = 96                    # dynamic payload type
CLOCK_HZ = 90000               # RTP video clock


@dataclass
class SchemeConfig:
    """One point in the evaluation matrix."""
    name: str = "phantom"
    recovery: str = "none"      # none | nack | fec | tcp
    conceal: str = "repairnet"  # freeze | copy | inpaint | flow | repairnet
    deadline_ms: float = 100.0  # playout deadline after capture
    fec_k: int = 5              # one XOR parity per k media packets
    jpeg_quality: int = 75
    interleave: bool = True
