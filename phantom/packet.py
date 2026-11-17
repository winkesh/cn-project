"""RTP-compatible packet format (RFC 3550 fixed header + RFC 8285 one-byte-style extension).

Layout
------
 RTP fixed header (12 B): V=2,P=0,X=1,CC=0 | M | PT | seq | timestamp | SSRC
 Extension header (4 B):  0xBEDE profile | length in 32-bit words
 Extension body (32 B at 360p, 88 B at 720p):
     frame_id   u32   unambiguous frame key
     pkt_idx    u16   index of this packet within the frame
     n_pkts     u16   media packets in this frame (720p frames need > 255)
     flags      u8    see FLAG_*
     fec_k      u8    FEC group size (0 = no FEC)
     send_us    u32   sender clock in microseconds (one-way delay measurement)
     skip_map   N_TILES bits: tiles intentionally not sent (unchanged)  [18 B at 360p, 72 B at 720p]
     pad        to a 32-bit boundary
 Payload: tile directory (n u8, then n x [tile_id u16, len u16]) + JPEG bytes
          or, for parity packets, the XOR of the group's padded payloads.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .config import N_TILES, RTP_PT

FLAG_PARITY = 0x01
FLAG_RETX = 0x02
FLAG_DROPPABLE = 0x04
FLAG_KEY = 0x08

_RTP = struct.Struct("!BBHII")
_EXT_HDR = struct.Struct("!HH")
SKIP_BYTES = (N_TILES + 7) // 8
_PAD = (-(14 + SKIP_BYTES)) % 4
_EXT = struct.Struct(f"!IHHBBI{SKIP_BYTES}s{_PAD}x")
EXT_WORDS = _EXT.size // 4
HEADER_SIZE = _RTP.size + _EXT_HDR.size + _EXT.size


@dataclass
class Packet:
    seq: int
    timestamp: int
    ssrc: int
    frame_id: int
    pkt_idx: int
    n_pkts: int
    payload: bytes
    marker: bool = False
    flags: int = 0
    fec_k: int = 0
    send_us: int = 0
    skip_map: bytes = bytes(SKIP_BYTES)
    # simulation-only bookkeeping (not serialised)
    meta: dict = field(default_factory=dict, compare=False, repr=False)

    def encode(self) -> bytes:
        b0 = (2 << 6) | (1 << 4)  # V=2, X=1
        b1 = (int(self.marker) << 7) | RTP_PT
        head = _RTP.pack(b0, b1, self.seq & 0xFFFF, self.timestamp & 0xFFFFFFFF, self.ssrc)
        ext = _EXT_HDR.pack(0xBEDE, EXT_WORDS) + _EXT.pack(
            self.frame_id, self.pkt_idx, self.n_pkts, self.flags, self.fec_k,
            self.send_us & 0xFFFFFFFF, self.skip_map)
        return head + ext + self.payload

    @classmethod
    def decode(cls, data: bytes) -> "Packet":
        b0, b1, seq, ts, ssrc = _RTP.unpack_from(data, 0)
        if b0 >> 6 != 2:
            raise ValueError("not an RTP v2 packet")
        off = _RTP.size
        profile, words = _EXT_HDR.unpack_from(data, off)
        if profile != 0xBEDE or words != EXT_WORDS:
            raise ValueError("unexpected header extension")
        off += _EXT_HDR.size
        fid, idx, n, flags, k, send_us, skip = _EXT.unpack_from(data, off)
        off += _EXT.size
        return cls(seq=seq, timestamp=ts, ssrc=ssrc, frame_id=fid, pkt_idx=idx, n_pkts=n,
                   payload=data[off:], marker=bool(b1 >> 7), flags=flags, fec_k=k,
                   send_us=send_us, skip_map=skip)

    @property
    def size(self) -> int:
        return HEADER_SIZE + len(self.payload)


def pack_tiles(tiles: list[tuple[int, bytes]]) -> bytes:
    head = bytearray([len(tiles)])
    for tid, data in tiles:
        head += struct.pack("!HH", tid, len(data))
    return bytes(head) + b"".join(d for _, d in tiles)


def unpack_tiles(payload: bytes) -> list[tuple[int, bytes]]:
    n = payload[0]
    off = 1
    dirs = []
    for _ in range(n):
        tid, ln = struct.unpack_from("!HH", payload, off)
        dirs.append((tid, ln))
        off += 4
    out = []
    for tid, ln in dirs:
        out.append((tid, payload[off:off + ln]))
        off += ln
    return out


def bitmap_from(ids, n_bits: int = SKIP_BYTES * 8) -> bytes:
    b = bytearray(n_bits // 8)
    for i in ids:
        b[i >> 3] |= 1 << (i & 7)
    return bytes(b)


def bitmap_ids(bm: bytes) -> set[int]:
    return {i for i in range(len(bm) * 8) if bm[i >> 3] >> (i & 7) & 1}
