"""Live UDP sender and receiver (deadline-driven playout, no retransmission)."""
from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque

import cv2
import numpy as np

from ..config import FPS, FRAME_MS, N_TILES
from ..packet import Packet
from ..receiver import FrameState, assemble
from ..sender import Packetizer
from ..tiles import letterbox, tile_mask_to_pixels


class FrameSource:
    """Loops a video file, or reads a webcam if path is an int / digit string."""

    def __init__(self, src):
        self.src = src
        self.is_file = not str(src).isdigit()
        self.cap = cv2.VideoCapture(str(src) if self.is_file else int(src))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {'video ' + str(src) if self.is_file else 'webcam ' + str(src)}")
        if not self.is_file:          # most webcams default to 640x480; ask for 16:9 HD instead
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

    def read(self) -> np.ndarray:
        ok, img = self.cap.read()
        if not ok and self.is_file:   # loop: reopen rather than seek (seeking is unreliable in some codecs)
            self.cap.release()
            self.cap = cv2.VideoCapture(str(self.src))
            ok, img = self.cap.read()
        if not ok:
            raise RuntimeError("source ended" if self.is_file else "webcam stopped delivering frames")
        return letterbox(img)          # never stretch: a 4:3 camera gets side bars


class LiveSender:
    def __init__(self, src, dst: tuple[str, int], quality=75, cr_threshold=2.0, fps=FPS, originals=None):
        self.source = FrameSource(src)
        self.dst = dst
        self.fps = fps
        self.pk = Packetizer(quality=quality, interleave=True, cr_threshold=cr_threshold)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
        self.sock.settimeout(0.0005)
        self.originals = originals          # optional dict frame_id -> original (demo only)
        self.bytes_sent = 0
        self.reported_loss = 0.0
        self._stop = threading.Event()

    def _poll_feedback(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except (socket.timeout, BlockingIOError, ConnectionResetError, OSError):
                return
            try:
                fb = json.loads(data)
            except ValueError:
                continue
            self.reported_loss = fb.get("loss", 0.0)
            # loss-aware refresh: tiles the receiver lost are re-sent next frame even if unchanged
            self.pk.force_refresh.update(fb.get("lost_tiles", []))

    def run(self):
        t_next = time.perf_counter()
        t0 = t_next
        while not self._stop.is_set():
            img = self.source.read()
            fid = self.pk.frame_id
            if self.originals is not None:
                self.originals[fid] = img
                self.originals.pop(fid - 120, None)
            send_us = int((time.perf_counter() - t0) * 1e6)   # capture time on the sender clock
            pkts = self.pk.packetize(img, send_us=send_us)
            # pace evenly over half a frame interval, against an absolute schedule (no drift)
            t_start, gap = time.perf_counter(), (0.5 / self.fps) / max(1, len(pkts))
            for i, p in enumerate(pkts):
                while time.perf_counter() < t_start + i * gap:
                    time.sleep(0)   # yield the GIL: receiver/relay/display may share this process
                b = p.encode()
                self.sock.sendto(b, self.dst)
                self.bytes_sent += len(b)
            self._poll_feedback()
            t_next += 1.0 / self.fps
            dt = t_next - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                t_next = time.perf_counter()

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()


class LiveReceiver:
    """Collects packets, and at each frame's playout deadline hands (fid, partial, mask) to on_frame."""

    def __init__(self, port: int, deadline_ms=100.0, feedback_every_ms=100.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.002)
        self.D = deadline_ms / 1000
        self.fb_every = feedback_every_ms / 1000
        self.states: dict[int, FrameState] = {}
        self.t0 = None
        self.fid0 = None
        self.next_fid = None
        self.peer = None
        self.prev_out = None
        self.recv_pkts = 0
        self.expected_pkts = 0
        self._fb_t = 0.0
        self._recent_lost: deque = deque(maxlen=30)
        self._win_recv = self._win_exp = 0
        self.late = 0
        self.bad = 0                         # packets that could not be decoded
        self.cap_us: dict[int, int] = {}   # frame_id -> sender capture time (us)
        self.us0 = 0

    def playout_time(self, fid):
        """RTP-style playout: local arrival time of the first packet + (sender capture time of this frame
        - sender capture time of the first frame) + D. Anchoring to the sender's capture clock rather than
        to frame numbers means a sender running slightly off-nominal never drifts out of the buffer.
        A frame we have heard nothing from is due just before the next frame we have heard from."""
        if fid in self.cap_us:
            return self.t0 + (self.cap_us[fid] - self.us0) / 1e6 + self.D
        later = [f for f in self.cap_us if f > fid]
        if later:
            return self.playout_time(min(later)) - 1e-4
        return float("inf")

    def report_missing(self, missing: set):
        self._recent_lost.append(set(missing))

    def poll(self):
        """Receive for a few ms; return frames whose playout deadline passed as (fid, FrameState|None).

        The caller assembles each frame (phantom.receiver.assemble) against its own previous output
        and calls report_missing() so the loss-aware refresh feedback knows which tiles were lost."""
        try:
            while True:
                data, addr = self.sock.recvfrom(65535)
                try:
                    p = Packet.decode(data)
                except Exception:          # wrong format: most often the sender runs another PHANTOM_RES
                    self.bad += 1
                    continue
                now = time.perf_counter()
                self.peer = addr
                if self.t0 is None:
                    self.t0, self.fid0, self.next_fid, self.us0 = now, p.frame_id, p.frame_id, p.send_us
                if p.frame_id < self.next_fid:
                    self.late += 1
                    continue
                self.cap_us.setdefault(p.frame_id, p.send_us)
                st = self.states.setdefault(p.frame_id, FrameState(p.frame_id))
                st.add(p, now)
                self.recv_pkts += 1
        except (socket.timeout, BlockingIOError, ConnectionResetError, OSError):
            pass
        out = []
        if self.t0 is None:
            return out
        now = time.perf_counter()
        while now >= self.playout_time(self.next_fid):
            fid = self.next_fid
            st = self.states.pop(fid, None)
            self.cap_us.pop(fid, None)
            n = st.n_pkts if st else 0
            if st:
                self._win_exp += n
                self._win_recv += len(st.media)
            out.append((fid, st))
            self.next_fid += 1
        if now - self._fb_t > self.fb_every and self.peer:
            self._fb_t = now
            lost_tiles = sorted(set().union(*self._recent_lost)) if self._recent_lost else []
            self._recent_lost.clear()
            loss = 1 - self._win_recv / self._win_exp if self._win_exp else 0.0
            self._win_recv = self._win_exp = 0
            fb = {"loss": round(loss, 4), "lost_tiles": lost_tiles[:N_TILES]}
            try:
                self.sock.sendto(json.dumps(fb).encode(), self.peer)
            except OSError:
                pass
        return out
