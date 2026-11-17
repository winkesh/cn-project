"""Live streaming engine behind the demo app.

source (file / URL / webcam) -> frames -> real Packetizer (RTP packets, interleaved JPEG tiles)
  -> lossy channel (same Bernoulli / Gilbert-Elliott models as the UDP relay)
  -> real receiver (FrameState + assemble)
or: another machine running `python -m phantom.send` -> real UDP packets on port 5004 -> loss simulated on
    arrival -> the same receiver path (the app answers the sender's discovery broadcast and switches to
    the network stream when one starts)
  -> left:  what the loss does (no repair | normal decoder freeze | copy last tile)
     right: Phantom regenerating the lost tiles (RepairNet, or framegen with one frame of look-ahead)
  -> composite [left | right] published to the video streams + per-frame stats.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import conceal as C
from ..channel import make_loss
from ..config import FPS, HEIGHT, N_TILES, WIDTH
from ..live.endpoints import LiveReceiver, local_ips
from ..packet import HEADER_SIZE
from ..receiver import FrameState, assemble
from ..sender import Packetizer
from ..tiles import tile_mask_to_pixels

TARGET_FPS = FPS


@dataclass
class Settings:
    loss: float = 0.10          # packet loss rate
    burst: float = 1.0          # mean burst length in packets (1 = independent/random loss)
    baseline: str = "none"      # left panel: none | freeze | copy
    model: str = "repairnet"    # right panel: repairnet | framegen
    left: str = "loss"          # loss | original
    outline: bool = False       # outline lost tiles on the left panel
    paused: bool = False


def letterbox(img: np.ndarray) -> np.ndarray:
    """Fit any input into WIDTHxHEIGHT without distortion (black bars)."""
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


class Source:
    def __init__(self, spec: str, kind: str, label: str):
        self.spec, self.kind, self.label = spec, kind, label
        self.cap = cv2.VideoCapture(int(spec) if kind == "webcam" else spec)
        if not self.cap.isOpened():
            raise IOError(f"could not open {label}")
        ok, first = self.cap.read()
        if not ok:
            raise IOError(f"{label} opened but no frames could be decoded")
        self.first = first
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 < fps < 241 else 30.0
        self.size = (first.shape[1], first.shape[0])
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if kind == "file" else 0
        # high-fps sources are decimated to ~30 fps (60 fps game capture -> every 2nd frame)
        self.stride = max(1, round(self.fps / TARGET_FPS))
        self.out_fps = self.fps / self.stride

    def read(self):
        if self.first is not None:
            f, self.first = self.first, None
            return f
        for _ in range(self.stride - 1):
            self.cap.grab()
        ok, img = self.cap.read()
        if not ok and self.kind in ("file", "url"):  # loop files / VOD URLs: reopen (live URLs reconnect)
            self.cap.release()
            self.cap = cv2.VideoCapture(self.spec)
            ok, img = self.cap.read()
        return img if ok else None

    def close(self):
        self.cap.release()


class NetSource:
    """A stream arriving over UDP from `python -m phantom.send` on another machine."""
    kind, n, out_fps = "network", 0, float(FPS)

    def __init__(self, peer_ip: str):
        self.label = f"Live from {peer_ip}"
        self.size = (WIDTH, HEIGHT)

    def close(self):
        pass


class Engine:
    def __init__(self, weights: str = "weights/repairnet.pt", framegen: str = "weights/framegen.pt",
                 udp_port: int = 5004, backend: str = "auto", repair_res: str = "auto"):
        self.settings = Settings()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(8)
        self.baselines = {"none": C.NoRecovery(), "freeze": C.Freeze(), "copy": C.CopyTile()}
        self.models: dict[str, C.Concealer] = {}
        if Path(weights).exists():
            self.models["repairnet"] = C.make_repairnet(weights, backend, repair_res)
        if Path(framegen).exists():
            fb = self.models.get("repairnet") or C.FlowExtrapolate()
            self.models["framegen"] = C.FramegenInterp(framegen, fallback=fb)
        if not self.models:
            self.models["flow"] = C.FlowExtrapolate()
            self.settings.model = "flow"
        self.source: Source | None = None
        self.source_error = ""
        self.frames: queue.Queue = queue.Queue(maxsize=3)    # reader -> sender
        self.net_q: queue.Queue = queue.Queue(maxsize=3)     # sender (after the lossy channel) -> receiver
        self.subscribers: list[queue.Queue] = []
        self.latest: np.ndarray | None = None
        self.stats: deque = deque(maxlen=300)
        self._gen = 0                               # bumps on every source change
        self._reset_pipeline()
        # receiving from another machine: always listening, answers discovery broadcasts
        self.udp_port, self.net_error, self.rx = udp_port, "", None
        self._net_ssrc, self._net_key, self._avg_pkts = None, None, 0.0
        self.ips = local_ips()
        try:
            self.rx = LiveReceiver(udp_port, deadline_ms=150)
        except OSError as e:
            self.net_error = f"cannot listen on UDP {udp_port}: {e}"
        # stages run concurrently, like a real sender and receiver: decode source | packetize + channel |
        # receive + repair + compose; plus the UDP listener for streams from another machine
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        if self.rx is not None:
            threading.Thread(target=self._net_loop, daemon=True).start()

    # ------------------------------------------------------------------ control
    def update(self, **kw):
        with self.lock:
            for k, v in kw.items():
                if hasattr(self.settings, k):
                    cur = getattr(self.settings, k)
                    setattr(self.settings, k, type(cur)(v) if not isinstance(cur, bool) else bool(v))
            s = self.settings
            s.loss = min(0.6, max(0.0, s.loss))
            s.burst = min(16.0, max(1.0, s.burst))
            if s.baseline not in self.baselines:
                s.baseline = "none"
            if s.model not in self.models:
                s.model = next(iter(self.models))
            self._loss_key = None                   # rebuild loss model on next frame
        return asdict(self.settings)

    def set_source(self, spec: str, kind: str, label: str):
        src = Source(spec, kind, label) if kind != "network" else NetSource(spec)   # IOError if unopenable
        with self.lock:
            old, self.source = self.source, src
            self._gen += 1
            self._reset_pipeline()
            self.source_error = ""
        for q in (self.frames, self.net_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        if old:
            old.close()
        return {"label": src.label, "kind": src.kind, "size": src.size, "fps": round(src.out_fps, 1),
                "frames": src.n}

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self.subscribers:
            self.subscribers.remove(q)

    # ------------------------------------------------------------------ pipeline
    def _reset_pipeline(self):
        self.pk = Packetizer(quality=80, interleave=True)
        self.pk.pool = self.pool
        self.hist_b: list = []
        self.hist_p: list = []
        self.pending = None
        self.fid = 0
        self._loss_key = None
        self.stats.clear()

    def _reader(self):
        while True:
            src = self.source
            if src is None or src.kind == "network" or self.settings.paused:
                time.sleep(0.05)
                continue
            gen = self._gen
            t0 = time.perf_counter()
            img = src.read()
            if img is None:
                self.source_error = f"{src.label}: stream ended or stalled"
                time.sleep(0.2)
                continue
            frame = letterbox(img)
            if gen != self._gen:
                continue
            try:
                self.frames.put((gen, frame), timeout=1.0)
            except queue.Full:
                continue
            if src.kind in ("file", "url"):         # play at the source's own frame rate (live streams
                                                    # already arrive at that rate, so this only waits if early)
                dt = 1.0 / src.out_fps - (time.perf_counter() - t0)
                if dt > 0:
                    time.sleep(dt)

    def _sender(self):
        while True:
            try:
                gen, frame = self.frames.get(timeout=0.5)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            try:
                cur = self._send(frame)
                self.net_q.put((gen, cur), timeout=2.0)
            except queue.Full:
                pass
            except Exception as e:                   # keep the app alive; show the error in the UI
                self.source_error = f"sender error: {e!r}"
                time.sleep(0.1)

    def _worker(self):
        while True:
            try:
                gen, cur = self.net_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            try:
                self._recv(cur)
            except Exception as e:
                self.source_error = f"receiver error: {e!r}"
                time.sleep(0.1)

    def _net_loop(self):
        """Receive real UDP packets from another machine and feed them into the receiver stage."""
        rx = self.rx
        while True:
            src = self.source
            on = src is not None and src.kind == "network"
            s = self.settings
            key = (s.loss, s.burst) if on else None
            if key != self._net_key:                  # the loss slider acts on arrival for network streams
                rx.loss_model = (make_loss("ge" if s.burst > 1.01 else "bernoulli", s.loss, s.burst,
                                           seed=int(time.time())) if on else None)
                self._net_key = key
            frames = rx.poll()
            if rx.ssrc is not None and rx.ssrc != self._net_ssrc and rx.peer:
                self._net_ssrc = rx.ssrc              # a sender just started: show its stream
                try:
                    self.set_source(rx.peer[0], "network", "")
                except Exception as e:
                    self.net_error = repr(e)
                continue
            if not on or s.paused:
                continue
            gen = self._gen
            for fid, st in frames:
                n = st.n_pkts if st else 0
                if n:
                    self._avg_pkts = 0.9 * self._avg_pkts + 0.1 * n if self._avg_pkts else n
                else:                                 # nothing at all arrived for this frame
                    n = round(self._avg_pkts)
                got = len(st.media) if st else 0
                nbytes = sum(len(p) + HEADER_SIZE for p in st.media.values()) if st else 0
                cur = dict(fid=fid, orig=None, st=st, pkts=n, lost=max(0, n - got), bytes=nbytes,
                           t0=time.perf_counter(), net=True)
                try:
                    self.net_q.put_nowait((gen, cur))
                except queue.Full:                    # receiver stage behind: drop rather than lag
                    pass

    def _channel(self, s: Settings):
        key = (s.loss, s.burst)
        if key != self._loss_key:
            kind = "ge" if s.burst > 1.01 else "bernoulli"
            self.loss_model = make_loss(kind, s.loss, s.burst, seed=int(time.time()))
            self._loss_key = key
        return self.loss_model

    def _snapshot(self) -> Settings:
        with self.lock:
            return Settings(**asdict(self.settings))

    def _process(self, orig: np.ndarray):
        """Single-threaded path (profiling / tests): send then receive."""
        self._recv(self._send(orig))

    def _send(self, orig: np.ndarray) -> dict:
        s = self._snapshot()
        t0 = time.perf_counter()
        # ---- sender: real packetizer
        pkts = self.pk.packetize(orig)
        # ---- network: lossy channel
        lm = self._channel(s)
        st = FrameState(self.fid)
        lost = 0
        for p in pkts:
            if lm.lost():
                lost += 1
            else:
                st.add(p)
        sent_bytes = sum(p.size for p in pkts)
        cur = dict(fid=self.fid, orig=orig, st=st, pkts=len(pkts), lost=lost, bytes=sent_bytes, t0=t0)
        self.fid += 1
        return cur

    def _recv(self, cur: dict):
        s = self._snapshot()
        # ---- receiver (framegen needs frame t+1 before it can repair frame t)
        if s.model == "framegen":
            prev, self.pending = self.pending, cur
            if prev is None:
                return
            self._receive(prev, s, nxt=cur)
        else:
            if self.pending is not None:            # switched away from framegen: flush
                self._receive(self.pending, s, nxt=None)
                self.pending = None
            self._receive(cur, s, nxt=None)

    def _receive(self, f: dict, s: Settings, nxt: dict | None):
        st, orig = f["st"], f["orig"]
        a_b = assemble(st, f["fid"], self.hist_b[-1] if self.hist_b else None, pool=self.pool)
        a_p = assemble(st, f["fid"], self.hist_p[-1] if self.hist_p else None, pool=self.pool)
        miss = a_p.missing
        if f.get("net"):
            self.rx.report_missing(miss)              # loss-aware refresh feedback to the remote sender
        mask = tile_mask_to_pixels(miss)
        out_b = self.baselines[s.baseline](a_b.image, mask, self.hist_b) if miss else a_b.image
        model = self.models[s.model]
        ti = time.perf_counter()
        if not miss:
            out_p = a_p.image
        elif s.model == "framegen" and nxt is not None:
            n_asm = assemble(nxt["st"], nxt["fid"], a_p.image, pool=self.pool)
            out_p = model(a_p.image, mask, self.hist_p, nxt=n_asm.image, nxt_missing_frac=len(n_asm.missing) / N_TILES)
        else:
            out_p = model(a_p.image, mask, self.hist_p)
        infer = (time.perf_counter() - ti) * 1000 if miss else 0.0
        for h, o in ((self.hist_b, out_b), (self.hist_p, out_p)):
            h.append(o)
            if len(h) > 3:
                h.pop(0)
        show_orig = s.left == "original" and orig is not None   # a live network stream has no original
        left = orig if show_orig else out_b
        if s.outline and miss and not show_orig:
            left = left.copy()
            edge = mask & ~cv2.erode(mask.view(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            left[edge] = (60, 60, 230)
        comp = np.hstack([left, out_p])
        self.latest = comp
        for q in list(self.subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put_nowait(comp)
        self.stats.append(dict(
            t=time.time(), fid=f["fid"], pkts=f["pkts"], lost=f["lost"], bytes=f["bytes"],
            tiles_lost=len(miss),
            psnr_b=None if orig is None else min(60.0, cv2.PSNR(out_b, orig)),
            psnr_p=None if orig is None else min(60.0, cv2.PSNR(out_p, orig)),
            infer_ms=infer, proc_ms=(time.perf_counter() - f["t0"]) * 1000))

    # ------------------------------------------------------------------ stats
    def state(self) -> dict:
        st = list(self.stats)
        recent = [r for r in st if r["t"] > time.time() - 1.5]
        win = st[-30:]
        agg = {}
        if win:
            sent = sum(r["pkts"] for r in win)
            dur = max(1e-3, win[-1]["t"] - win[0]["t"]) if len(win) > 1 else 1.0
            withloss = [r["infer_ms"] for r in win if r["tiles_lost"]]
            pb = [r["psnr_b"] for r in win if r["psnr_b"] is not None]
            pp = [r["psnr_p"] for r in win if r["psnr_p"] is not None]
            agg = dict(
                measured_loss=sum(r["lost"] for r in win) / max(1, sent),
                pkts_per_frame=sent / len(win),
                tiles_lost=win[-1]["tiles_lost"],
                tiles_lost_avg=float(np.mean([r["tiles_lost"] for r in win])),
                psnr_b=float(np.mean(pb)) if pb else None,      # None: live stream, no original to compare
                psnr_p=float(np.mean(pp)) if pp else None,
                infer_ms=float(np.median(withloss)) if withloss else 0.0,
                fps=(len(recent) - 1) / max(1e-3, recent[-1]["t"] - recent[0]["t"]) if len(recent) > 2 else 0.0,
                mbps=sum(r["bytes"] for r in win) * 8 / dur / 1e6 if len(win) > 1 else 0.0,
            )
        src = self.source
        rx = self.rx
        hist = [r for r in st[-240:] if r["psnr_p"] is not None]
        return dict(
            settings=asdict(self.settings),
            models=list(self.models), baselines=list(self.baselines),
            model_info={k: getattr(v, "label", "") for k, v in self.models.items()},
            source=None if src is None else dict(label=src.label, kind=src.kind, size=src.size,
                                                 fps=round(src.out_fps, 1)),
            error=self.source_error, res=[WIDTH, HEIGHT], tiles=N_TILES, agg=agg,
            hist_b=[round(r["psnr_b"], 2) for r in hist],
            hist_p=[round(r["psnr_p"], 2) for r in hist],
            hist_loss=[round(r["tiles_lost"] / N_TILES, 3) for r in st[-240:]],
            net=dict(listening=rx is not None, port=self.udp_port, ips=self.ips, error=self.net_error,
                     sender=rx.peer[0] if rx is not None and rx.peer else None,
                     active=rx is not None and time.time() - rx.last_rx < 2.0),
        )
