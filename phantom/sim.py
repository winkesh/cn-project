"""Discrete-event simulator: runs the real packetizer / receiver / concealment code over an emulated link.

Same packet format and receiver logic as the live UDP mode, but deterministic and much faster
than real time, so the whole evaluation matrix runs in minutes.

Recovery schemes
    none : no retransmission, no FEC. Missing tiles are concealed at the playout deadline.  (Phantom)
    nack : RTP-style NACK. Receiver detects sequence gaps and asks for retransmission; retransmitted
           packets that arrive before the deadline are used, the rest are concealed.
    fec  : one XOR parity packet per k media packets (RFC 5109 style); recovers 1 loss per group.
    tcp  : reliable in-order delivery (fast retransmit after 1 RTT, then RTO >= 200 ms), no deadline;
           the player waits for each frame, so latency grows and the picture stalls instead.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import asdict, dataclass, replace

import numpy as np

from . import metrics as M
from .channel import Link, make_loss
from .config import FRAME_MS, N_TILES, SchemeConfig
from .packet import FLAG_PARITY, FLAG_RETX
from .receiver import FrameState, assemble
from .sender import Packetizer
from .tiles import tile_mask_to_pixels


@dataclass
class NetConfig:
    loss: float = 0.05
    loss_model: str = "bernoulli"   # bernoulli | ge
    burst: float = 3.0              # mean burst length (packets) for ge
    delay_ms: float = 25.0          # one-way
    jitter_ms: float = 2.0
    rate_mbps: float = 0.0
    seed: int = 0
    schedule: tuple | None = None   # ((t_ms, loss), ...): time-varying loss, overrides `loss`

    @property
    def rtt(self) -> float:
        return 2 * self.delay_ms


_ENC_CACHE: dict = {}


def _encode_all(frames, scheme: SchemeConfig, seed: int):
    """Packetize the clip once per (clip, quality, interleave, fec) and reuse it across runs."""
    fec = scheme.fec_k if scheme.recovery == "fec" else 0
    key = (id(frames), len(frames), scheme.jpeg_quality, scheme.interleave, fec)
    if key not in _ENC_CACHE:
        if len(_ENC_CACHE) > 6:
            _ENC_CACHE.clear()
        _ENC_CACHE[key] = _encode_uncached(frames, scheme, seed)
    return _ENC_CACHE[key]


def _encode_uncached(frames, scheme: SchemeConfig, seed: int):
    pk = Packetizer(quality=scheme.jpeg_quality, interleave=scheme.interleave,
                    fec_k=scheme.fec_k if scheme.recovery == "fec" else 0, ssrc=0x50484E54)
    pk.seq = seed & 0xFFFF
    out, g = [], 0
    for img in frames:
        pkts = pk.packetize(img)
        for p in pkts:
            p.meta["g"] = g
            g += 1
        out.append(pkts)
    return pk, out


def simulate(frames, scheme: SchemeConfig, net: NetConfig, concealer=None, keep_outputs: bool = False,
             compute_ssim: bool = True):
    from .conceal import CopyTile

    concealer = concealer or CopyTile()
    pk, per_frame = _encode_all(frames, scheme, net.seed)
    link = Link(delay_ms=net.delay_ms, jitter_ms=net.jitter_ms, rate_mbps=net.rate_mbps,
                loss=make_loss(net.loss_model, net.loss, net.burst, net.seed, net.schedule), seed=net.seed)
    by_g = {p.meta["g"]: (f, p) for f, pk_list in enumerate(per_frame) for p in pk_list}
    media_bytes = sum(p.size for pl in per_frame for p in pl if not p.flags & FLAG_PARITY)

    if scheme.recovery == "tcp":
        return _simulate_tcp(frames, per_frame, link, net, scheme, media_bytes, keep_outputs, compute_ssim)

    # look-ahead concealers (frame interpolation) hold each frame one extra frame interval so that
    # frame t+1 has reached its own normal deadline when frame t is shown
    lookahead = getattr(concealer, "needs_next", False)
    # Playout deadline = capture + one-way delay + jitter buffer (deadline_ms), like a real receiver that
    # anchors playout to packet arrival (see phantom.live.endpoints.LiveReceiver.playout_time).
    D = net.delay_ms + scheme.deadline_ms + (FRAME_MS if lookahead else 0.0)
    heap, ctr = [], itertools.count()

    def push(t, kind, data):
        heapq.heappush(heap, (t, next(ctr), kind, data))

    for f, pkts in enumerate(per_frame):
        tc = f * FRAME_MS
        for i, p in enumerate(pkts):
            push(tc + i * (FRAME_MS * 0.8) / len(pkts), "send", p)
        push(tc + D, "deadline", f)

    states: dict[int, FrameState] = {}
    done: set[int] = set()
    received: set[int] = set()
    nack_tries: dict[int, int] = {}
    highest = -1
    sent_bytes = retx_bytes = 0
    late = 0
    outputs, history = [], []
    rows = []

    while heap:
        t, _, kind, data = heapq.heappop(heap)
        if kind == "send":
            p = data
            sent_bytes += p.size
            if p.flags & FLAG_RETX:
                retx_bytes += p.size
            arr = link.send(t, p.size)
            if arr is not None:
                push(arr, "arrive", p)
        elif kind == "arrive":
            p = data
            g = p.meta["g"]
            f = p.frame_id
            if g in received:
                continue
            received.add(g)
            if scheme.recovery == "nack" and g > highest:
                for miss in range(highest + 1, g):
                    if miss not in received and miss not in nack_tries:
                        nack_tries[miss] = 1
                        push(t + net.delay_ms, "nack", miss)
                        push(t + net.rtt * 1.5 + 5, "recheck", miss)
                highest = g
            highest = max(highest, g)
            if f in done:
                late += 1
                continue
            states.setdefault(f, FrameState(f)).add(p, t)
        elif kind == "nack":
            f, orig = by_g[data]
            if f in done:
                continue
            retx = replace(orig, flags=orig.flags | FLAG_RETX, meta={"g": data})
            push(t, "send", retx)
        elif kind == "recheck":
            g = data
            f, _ = by_g[g]
            if g in received or f in done or nack_tries[g] >= 5:
                continue
            nack_tries[g] += 1
            push(t + net.delay_ms, "nack", g)
            push(t + net.rtt * 1.5 + 5, "recheck", g)
        elif kind == "deadline":
            f = data
            done.add(f)
            asm = assemble(states.pop(f, None), f, history[-1] if history else None)
            mask = tile_mask_to_pixels(asm.missing)
            if asm.missing and lookahead:
                nst = states.get(f + 1)
                nxt = assemble(nst, f + 1, asm.image) if nst is not None else None
                frac = len(nxt.missing) / N_TILES if nxt is not None else 1.0
                out = concealer(asm.image, mask, history, nxt=nxt.image if nxt else None, nxt_missing_frac=frac)
            elif asm.missing:
                out = concealer(asm.image, mask, history)
            else:
                out = asm.image
            frozen = bool(history) and np.array_equal(out, history[-1])
            gt = frames[f]
            rows.append({
                "frame": f,
                "missing_tiles": len(asm.missing),
                "psnr": M.psnr(out, gt),
                "ssim": M.ssim(out, gt) if compute_ssim else np.nan,
                "masked_psnr": M.masked_psnr(out, gt, mask),
                "frozen": frozen,
                "latency_ms": D,
            })
            history.append(out)
            if len(history) > 3:
                history.pop(0)
            if keep_outputs:
                outputs.append(out)

    summary = _summarise(rows, scheme, net, media_bytes, sent_bytes, retx_bytes, len(frames), concealer)
    if keep_outputs:
        summary["per_frame"] = rows   # per-frame stats for the demo video overlays
    return summary, outputs


def _simulate_tcp(frames, per_frame, link, net, scheme, media_bytes, keep_outputs, compute_ssim):
    from . import tiles as T

    deliver_prev = 0.0
    sent_bytes = retx_bytes = 0
    complete = []
    for f, pkts in enumerate(per_frame):
        tc = f * FRAME_MS
        last = 0.0
        for i, p in enumerate(pkts):
            t_send = tc + i * (FRAME_MS * 0.8) / len(pkts)
            rto = 200.0
            tries = 0
            while True:
                sent_bytes += p.size
                if tries:
                    retx_bytes += p.size
                arr = link.send(t_send, p.size)
                if arr is not None:
                    break
                # first loss -> fast retransmit after ~1 RTT; later losses -> RTO with backoff
                t_send += net.rtt + 2 if tries == 0 else rto
                if tries:
                    rto *= 2
                tries += 1
            deliver_prev = max(deliver_prev, arr)   # in-order delivery (head-of-line blocking)
            last = deliver_prev
        complete.append(last)

    rows, outputs = [], []
    disp_prev = -1e9
    for f, pkts in enumerate(per_frame):
        tc = f * FRAME_MS
        disp = max(tc + net.delay_ms + scheme.deadline_ms, complete[f], disp_prev + FRAME_MS)
        img = np.zeros_like(frames[f])
        for p in pkts:
            if p.flags & FLAG_PARITY:
                continue
            from .packet import unpack_tiles
            for tid, data in unpack_tiles(p.payload):
                T.paste(img, tid, T.decode_tile(data))
        gt = frames[f]
        stall = disp - disp_prev - FRAME_MS if f else 0.0
        rows.append({"frame": f, "missing_tiles": 0, "psnr": M.psnr(img, gt),
                     "ssim": M.ssim(img, gt) if compute_ssim else np.nan, "masked_psnr": None,
                     "frozen": False, "latency_ms": disp - tc, "stall_gap_ms": max(0.0, stall)})
        disp_prev = disp
        if keep_outputs:
            outputs.append(img)
    return _summarise(rows, scheme, net, media_bytes, sent_bytes, retx_bytes, len(frames), None), outputs


def _summarise(rows, scheme, net, media_bytes, sent_bytes, retx_bytes, n_frames, concealer):
    ps = np.array([r["psnr"] for r in rows])
    ss = np.array([r["ssim"] for r in rows], dtype=float)
    mp = [r["masked_psnr"] for r in rows if r["masked_psnr"] is not None]
    lat = np.array([r["latency_ms"] for r in rows])
    frozen = [r["frozen"] for r in rows]
    if "stall_gap_ms" in rows[0]:
        gaps = [r["stall_gap_ms"] for r in rows]
        stall_count = sum(1 for g in gaps if g >= 200 - FRAME_MS)
        stall_ms = float(sum(gaps))
    else:
        stall_count, stall_ms = M.stall_stats(frozen, FRAME_MS)
    dur_s = n_frames * FRAME_MS / 1000
    inf = getattr(concealer, "times_ms", None)
    return {
        "scheme": scheme.name, "recovery": scheme.recovery, "conceal": scheme.conceal,
        "deadline_ms": scheme.deadline_ms, "fec_k": scheme.fec_k if scheme.recovery == "fec" else 0,
        **{f"net_{k}": v for k, v in asdict(net).items()},
        "psnr_mean": float(ps.mean()), "psnr_p5": float(np.percentile(ps, 5)),
        "ssim_mean": float(np.nanmean(ss)) if not np.isnan(ss).all() else np.nan,
        "masked_psnr_mean": float(np.mean(mp)) if mp else np.nan,
        "incomplete_pct": 100.0 * np.mean([r["missing_tiles"] > 0 for r in rows]),
        "lost_tile_pct": 100.0 * np.mean([r["missing_tiles"] for r in rows]) / N_TILES,
        "frozen_pct": 100.0 * np.mean(frozen),
        "stall_count": stall_count, "stall_ms": stall_ms,
        "latency_p50": float(np.percentile(lat, 50)), "latency_p95": float(np.percentile(lat, 95)),
        "latency_max": float(lat.max()),
        "media_mbps": media_bytes * 8 / dur_s / 1e6, "sent_mbps": sent_bytes * 8 / dur_s / 1e6,
        "overhead_pct": 100.0 * (sent_bytes - media_bytes) / media_bytes,
        "retx_pct": 100.0 * retx_bytes / max(1, sent_bytes),
        "infer_ms_p95": float(np.percentile(inf, 95)) if inf else np.nan,
    }
