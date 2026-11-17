"""Receiver: reassemble packets into frames, recover with FEC, and report which tiles are missing."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import tiles as T
from .config import HEIGHT, N_TILES, WIDTH
from .packet import FLAG_PARITY, Packet, bitmap_ids, unpack_tiles
from .sender import xor_bytes


@dataclass
class FrameState:
    frame_id: int
    n_pkts: int = 0
    fec_k: int = 0
    media: dict = field(default_factory=dict)     # pkt_idx -> payload
    parity: dict = field(default_factory=dict)    # group idx -> payload
    skip: set = field(default_factory=set)
    first_arrival: float | None = None
    decoded: dict | None = None                   # tile_id -> decoded tile, filled by assemble()
    _n_decoded: int = -1

    def add(self, p: Packet, now: float = 0.0) -> None:
        self.n_pkts = p.n_pkts
        self.fec_k = p.fec_k
        if self.first_arrival is None:   # skip map is identical in every packet of a frame
            self.skip = bitmap_ids(p.skip_map)
        if self.first_arrival is None:
            self.first_arrival = now
        if p.flags & FLAG_PARITY:
            self.parity[p.pkt_idx] = p.payload
        else:
            self.media[p.pkt_idx] = p.payload

    def fec_recover(self) -> int:
        """Recover single losses per parity group. Returns number recovered."""
        if not self.fec_k:
            return 0
        rec = 0
        for gi, par in self.parity.items():
            idx = range(gi * self.fec_k, min((gi + 1) * self.fec_k, self.n_pkts))
            missing = [i for i in idx if i not in self.media]
            if len(missing) == 1:
                self.media[missing[0]] = xor_bytes([par] + [self.media[i] for i in idx if i in self.media])
                rec += 1
        return rec

    @property
    def complete(self) -> bool:
        return self.n_pkts > 0 and len(self.media) >= self.n_pkts


@dataclass
class Assembled:
    frame_id: int
    image: np.ndarray            # tiles that arrived pasted over the previous output
    missing: set                 # tile ids lost in the network
    any_packet: bool             # False when nothing at all arrived for this frame


def assemble(state: FrameState | None, frame_id: int, prev: np.ndarray | None, pool=None) -> Assembled:
    """Build the partial frame. Skipped (unchanged) tiles are copied from prev; lost tiles reported.
    pool: optional ThreadPoolExecutor to JPEG-decode tiles in parallel (live app; no memoisation)."""
    base = prev.copy() if prev is not None else np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    if state is None or not state.media and not state.parity:
        return Assembled(frame_id, base, set(range(N_TILES)), False)
    if state.decoded is None or len(state.media) != state._n_decoded:
        state.fec_recover()
        items = [(tid, data) for payload in state.media.values() for tid, data in unpack_tiles(payload)
                 if tid < N_TILES]
        if pool is not None:
            from .sender import chunked_map
            tiles = chunked_map(pool, lambda its: [T.decode_tile_raw(d) for _, d in its], items)
        else:
            tiles = (T.decode_tile(data) for _, data in items)
        state.decoded = {tid: tile for (tid, _), tile in zip(items, tiles) if tile is not None}
        state._n_decoded = len(state.media)
    for tid, tile in state.decoded.items():
        T.paste(base, tid, tile)
    missing = set(range(N_TILES)) - set(state.decoded) - state.skip
    return Assembled(frame_id, base, missing, True)
