"""Core correctness tests: python -m pytest -q  (or python tests/test_core.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom.config import N_TILES, SchemeConfig  # noqa: E402
from phantom.packet import Packet  # noqa: E402
from phantom.receiver import FrameState, assemble  # noqa: E402
from phantom.sender import Packetizer  # noqa: E402
from phantom.sim import NetConfig, simulate  # noqa: E402
from phantom.video import synthetic  # noqa: E402


def test_packet_roundtrip():
    f = synthetic(1)[0]
    for p in Packetizer().packetize(f):
        q = Packet.decode(p.encode())
        assert (q.seq, q.frame_id, q.pkt_idx, q.n_pkts, q.payload) == (p.seq, p.frame_id, p.pkt_idx, p.n_pkts, p.payload)
        assert p.size <= 1200 + 48


def test_lossless_reassembly_is_exact_jpeg():
    f = synthetic(1)[0]
    st = FrameState(0)
    for p in Packetizer(quality=95).packetize(f):
        st.add(Packet.decode(p.encode()))
    a = assemble(st, 0, None)
    assert not a.missing
    assert np.abs(a.image.astype(int) - f).mean() < 3


def test_loss_mask_matches_dropped_packets():
    f = synthetic(1)[0]
    pkts = Packetizer().packetize(f)
    st = FrameState(0)
    dropped_tiles = set()
    from phantom.packet import unpack_tiles
    for i, p in enumerate(pkts):
        if i % 4 == 1:
            dropped_tiles |= {t for t, _ in unpack_tiles(p.payload)}
        else:
            st.add(p)
    assert assemble(st, 0, None).missing == dropped_tiles


def test_fec_recovers_single_loss():
    f = synthetic(1)[0]
    pkts = Packetizer(fec_k=4).packetize(f)
    st = FrameState(0)
    for p in pkts:
        if not (p.pkt_idx == 2 and not p.flags):
            st.add(p)
    assert assemble(st, 0, None).missing == set()


def _burst_holes(interleave):
    """Drop 4 consecutive packets; return the largest 4-connected hole (in tiles)."""
    from phantom.packet import unpack_tiles
    from phantom.config import TILES_X, TILES_Y
    import cv2
    pkts = Packetizer(interleave=interleave).packetize(synthetic(1)[0])
    lost = {t for p in pkts[10:14] for t, _ in unpack_tiles(p.payload)}
    g = np.zeros((TILES_Y, TILES_X), np.uint8)
    for t in lost:
        g[t // TILES_X, t % TILES_X] = 1
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=4)
    return max(stats[1:, cv2.CC_STAT_AREA]) if n > 1 else 0


def test_interleave_turns_bursts_into_scattered_holes():
    assert _burst_holes(False) >= 4
    assert _burst_holes(True) <= 2


def test_sim_runs_all_schemes():
    frames = synthetic(30)
    for rec in ["none", "nack", "fec", "tcp"]:
        r, _ = simulate(frames, SchemeConfig(rec, rec, "copy"), NetConfig(loss=0.1), compute_ssim=False)
        assert r["psnr_mean"] > 10
    r0, _ = simulate(frames, SchemeConfig("x", "none", "copy"), NetConfig(loss=0.0), compute_ssim=False)
    assert r0["incomplete_pct"] == 0


def test_long_path_delay_alone_does_not_cause_loss():
    """Deadline is measured from arrival (one-way delay + jitter buffer), so a long but clean path is fine."""
    r, _ = simulate(synthetic(20), SchemeConfig("a", "none", "copy"), NetConfig(loss=0.0, delay_ms=150),
                    compute_ssim=False)
    assert r["incomplete_pct"] == 0


def test_nack_fails_when_rtt_exceeds_buffer():
    frames = synthetic(60)
    near, _ = simulate(frames, SchemeConfig("b", "nack", "copy"), NetConfig(loss=0.1, delay_ms=10), compute_ssim=False)
    far, _ = simulate(frames, SchemeConfig("b", "nack", "copy"), NetConfig(loss=0.1, delay_ms=80), compute_ssim=False)
    assert near["incomplete_pct"] < 20 < far["incomplete_pct"]


def test_nack_helps_when_rtt_small():
    frames = synthetic(60)
    none, _ = simulate(frames, SchemeConfig("a", "none", "copy"), NetConfig(loss=0.1, delay_ms=10), compute_ssim=False)
    nack, _ = simulate(frames, SchemeConfig("b", "nack", "copy"), NetConfig(loss=0.1, delay_ms=10), compute_ssim=False)
    assert nack["incomplete_pct"] < none["incomplete_pct"]


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
