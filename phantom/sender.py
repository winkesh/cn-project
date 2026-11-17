"""Sender: frame -> tiles -> JPEG -> interleaved RTP packets (+ optional XOR FEC parity)."""
from __future__ import annotations

import math
import random

import numpy as np

from . import tiles as T
from .config import CLOCK_HZ, FPS, MAX_PAYLOAD, N_TILES
from .packet import FLAG_PARITY, FLAG_RETX, Packet, bitmap_from, pack_tiles

DIR_OVERHEAD = 1  # tile-count byte
PER_TILE_DIR = 4


ILV_STRIDE = 37  # coprime with N_TILES (144 or 576): walking tiles with this stride visits every tile once


def scatter_order(tile_ids) -> list[int]:
    """Tile order used for interleaving: positions k*37 mod N_TILES, so tiles that end up in the same or in
    consecutive packets are far apart on screen (a burst of lost packets -> scattered single holes)."""
    s = set(tile_ids)
    return [t for t in ((k * ILV_STRIDE) % N_TILES for k in range(N_TILES)) if t in s]


def layout(tile_ids, n: int, interleave: bool) -> list[list[int]]:
    """Split tiles into n packets: contiguous raster chunks, or chunks of the scattered order."""
    order = scatter_order(tile_ids) if interleave else sorted(tile_ids)
    per = math.ceil(len(order) / n) if order else 1
    return [order[j * per:(j + 1) * per] for j in range(n)]


def chunked_map(pool, fn_batch, items, parts: int = 8) -> list:
    """Run fn_batch over `parts` slices of items in a thread pool; return the concatenated results.
    One task per slice instead of per item: per-task overhead dominated at 576 tiles per frame."""
    items = list(items)
    if not items:
        return []
    k = -(-len(items) // parts)
    out = []
    for r in pool.map(fn_batch, [items[i:i + k] for i in range(0, len(items), k)]):
        out.extend(r)
    return out


def xor_bytes(chunks: list[bytes]) -> bytes:
    n = max(len(c) for c in chunks)
    acc = np.zeros(n, np.uint8)
    for c in chunks:
        acc[: len(c)] ^= np.frombuffer(c, np.uint8)
    return acc.tobytes()


class Packetizer:
    def __init__(self, quality: int = 75, interleave: bool = True, fec_k: int = 0,
                 ssrc: int | None = None, cr_threshold: float = 0.0, refresh: int = 15):
        self.quality = quality
        self.interleave = interleave
        self.fec_k = fec_k
        self.ssrc = ssrc if ssrc is not None else random.getrandbits(32)
        self.seq = random.getrandbits(16)
        self.frame_id = 0
        # conditional replenishment: skip tiles that barely changed since last sent
        self.cr_threshold = cr_threshold
        self.refresh = refresh
        self._last_sent: list[np.ndarray | None] = [None] * N_TILES
        self._age = [0] * N_TILES
        self.force_refresh: set[int] = set()   # tiles the receiver reported lost
        self._history: dict[int, Packet] = {}  # seq -> packet, for NACK retransmission
        self.pool = None                       # optional ThreadPoolExecutor for tile encoding

    def _choose_tiles(self, tiles: list[np.ndarray]) -> tuple[list[int], list[int]]:
        if self.cr_threshold <= 0:
            return list(range(N_TILES)), []
        send, skip = [], []
        for i, t in enumerate(tiles):
            last = self._last_sent[i]
            changed = last is None or np.abs(t.astype(np.int16) - last).mean() > self.cr_threshold
            if changed or self._age[i] >= self.refresh or i in self.force_refresh:
                send.append(i)
                self._last_sent[i] = t.astype(np.int16)
                self._age[i] = 0
            else:
                skip.append(i)
                self._age[i] += 1
        self.force_refresh.clear()
        return send, skip

    def packetize(self, frame: np.ndarray, send_us: int = 0) -> list[Packet]:
        tiles = T.split(frame)
        send_ids, skip_ids = self._choose_tiles(tiles)
        if self.pool is not None:     # cv2 releases the GIL: encode tiles in 8 parallel batches
            enc = lambda ids: [T.encode_tile(tiles[i], self.quality) for i in ids]
            coded = dict(zip(send_ids, chunked_map(self.pool, enc, send_ids)))
        else:
            coded = {i: T.encode_tile(tiles[i], self.quality) for i in send_ids}
        # Smallest packet count n such that splitting the (scattered or raster) tile order into n equal-count
        # chunks keeps every packet within MAX_PAYLOAD. Prefix sums make each try O(n) instead of O(tiles):
        # same packets as trying layout() for n = 1, 2, ... but ~100x faster at 720p (576 tiles).
        order = scatter_order(send_ids) if self.interleave else sorted(send_ids)
        L = len(order)
        csum = np.concatenate([[0], np.cumsum([len(coded[i]) + PER_TILE_DIR for i in order])])
        n = max(1, math.ceil(csum[-1] / (MAX_PAYLOAD - DIR_OVERHEAD)))
        while L:
            per = math.ceil(L / n)
            starts = np.arange(0, L, per)
            ends = np.minimum(starts + per, L)
            if DIR_OVERHEAD + (csum[ends] - csum[starts]).max() <= MAX_PAYLOAD or n >= L:  # 1 tile/packet floor
                break
            n += 1
        groups = [order[s:e] for s, e in zip(starts, ends)] if L else [[]]   # [[]]: a frame with nothing
        n = len(groups)                                                      # to send still carries its skip map
        skip_map = bitmap_from(skip_ids)
        ts = int(self.frame_id * CLOCK_HZ / FPS)
        pkts = []
        for j, g in enumerate(groups):
            pkts.append(Packet(seq=self.seq, timestamp=ts, ssrc=self.ssrc, frame_id=self.frame_id,
                               pkt_idx=j, n_pkts=n, payload=pack_tiles([(i, coded[i]) for i in g]),
                               marker=(j == n - 1 and not self.fec_k), fec_k=self.fec_k,
                               send_us=send_us, skip_map=skip_map))
            self.seq = (self.seq + 1) & 0xFFFF
        if self.fec_k:
            for gi in range(math.ceil(n / self.fec_k)):
                members = pkts[gi * self.fec_k:(gi + 1) * self.fec_k]
                par = Packet(seq=self.seq, timestamp=ts, ssrc=self.ssrc, frame_id=self.frame_id,
                             pkt_idx=gi, n_pkts=n, payload=xor_bytes([m.payload for m in members]),
                             flags=FLAG_PARITY, fec_k=self.fec_k, send_us=send_us, skip_map=skip_map)
                self.seq = (self.seq + 1) & 0xFFFF
                pkts.append(par)
            pkts[-1].marker = True
        for p in pkts:
            self._history[p.seq] = p
        if len(self._history) > 4096:
            for k in list(self._history)[:-2048]:
                del self._history[k]
        self.frame_id += 1
        return pkts

    def retransmit(self, seq: int) -> Packet | None:
        p = self._history.get(seq)
        if p is None:
            return None
        return Packet(seq=p.seq, timestamp=p.timestamp, ssrc=p.ssrc, frame_id=p.frame_id,
                      pkt_idx=p.pkt_idx, n_pkts=p.n_pkts, payload=p.payload, marker=p.marker,
                      flags=p.flags | FLAG_RETX, fec_k=p.fec_k, send_us=p.send_us, skip_map=p.skip_map)
