"""Phantom Frames demo app: HTTP server (standard library only) + H.264 / MJPEG live streams.

    python -m phantom.app                                  # 360p on 127.0.0.1:8000
    PHANTOM_RES=720 python -m phantom.app                  # 720p (faster GPU)
    python -m phantom.app --host 0.0.0.0                   # reachable from other devices on the network

Then open http://<host>:<port>/ in a browser. No external services are required.
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from ..config import HEIGHT, RES, WIDTH
from ..video import VIDEO_EXT
from .engine import Engine

STATIC = Path(__file__).parent / "static"
UPLOADS = Path.home() / ".cache" / "phantom_uploads"
URL_SCHEMES = ("http", "https", "rtsp", "rtsps", "rtmp", "srt", "udp")
ENGINE: Engine | None = None
SAMPLES: dict[str, Path] = {}


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def find_samples() -> dict[str, Path]:
    roots = ([Path("data720/test")] if RES == 720 else []) + [Path("data/test")]
    for r in roots:
        if r.exists():
            vids = sorted(p for p in r.iterdir() if p.suffix.lower() in VIDEO_EXT)
            if vids:
                return {p.stem: p for p in vids}
    return {}


NICE = {"game_forza": "Racing game (Forza Horizon 5)", "game_mw3": "Shooter (CoD: MW3)",
        "park_joy": "Fast motion (park_joy)", "tears_of_steel": "Film (Tears of Steel)",
        "FourPeople": "Video call (FourPeople)", "shields": "Camera pan (shields)"}


def results_summary(path: str = "results/results.csv") -> dict:
    """Headline numbers from the offline evaluation (eval/run_matrix.py), for the results table."""
    import csv
    from statistics import mean

    p = Path(path)
    if not p.exists():
        return {"rows": []}
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    clips = sorted({r["clip"] for r in rows})

    def m(exp, x, scheme, col):
        v = [float(r[col]) for r in rows if r["experiment"] == exp and float(r["x"]) == x and r["scheme"] == scheme
             and r[col] not in ("", "nan")]
        return round(mean(v), 2) if v else None

    order = ["Phantom (RepairNet)", "Phantom + NACK", "Phantom-Interp (framegen, +33 ms)", "NACK + copy",
             "FEC k=5 + copy", "TCP", "Flow extrapolation", "Copy-tile (no retx)", "Freeze (no retx)"]
    out = []
    for s in order:
        if not any(r["scheme"] == s for r in rows):
            continue
        out.append(dict(scheme=s, phantom=s.startswith("Phantom"),
                        psnr_5=m("loss", 5, s, "psnr_mean"), psnr_20=m("loss", 20, s, "psnr_mean"),
                        rtt_20=m("rtt", 20, s, "psnr_mean"), rtt_200=m("rtt", 200, s, "psnr_mean"),
                        latency=m("loss", 5, s, "latency_p95"), overhead=m("loss", 5, s, "overhead_pct")))
    return {"rows": out, "clips": len(clips)}


def open_source(spec: str, kind: str, label: str, timeout: float = 20.0) -> dict:
    """Open in a worker thread: a bad URL can block inside FFmpeg for a long time."""
    box: dict = {}

    def run():
        try:
            box["ok"] = ENGINE.set_source(spec, kind, label)
        except Exception as e:
            box["err"] = str(e)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return {"error": f"timed out opening {label}"}
    return {"source": box["ok"]} if "ok" in box else {"error": box.get("err", "unknown error")}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "PhantomFrames/1.0"

    def log_message(self, fmt, *args):  # quiet
        pass

    # ---------------------------------------------------------------- helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # ---------------------------------------------------------------- routes
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            body = (STATIC / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/api/state":
            self._json(ENGINE.state())
        elif u.path == "/api/samples":
            self._json([{"id": k, "name": NICE.get(k, k)} for k in SAMPLES])
        elif u.path == "/api/results":
            self._json(results_summary())
        elif u.path == "/video.mp4":
            self._stream_h264(parse_qs(u.query).get("q", ["high"])[0])
        elif u.path == "/stream.mjpg":
            self._stream_mjpeg()
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/settings":
                self._json({"settings": ENGINE.update(**self._body_json())})
            elif u.path == "/api/source":
                b = self._body_json()
                if "sample" in b:
                    p = SAMPLES.get(b["sample"])
                    if not p:
                        return self._json({"error": "unknown sample"}, 400)
                    r = open_source(str(p), "file", NICE.get(p.stem, p.stem))
                elif "url" in b:
                    url = b["url"].strip()
                    scheme = urlparse(url).scheme.lower()
                    if scheme not in URL_SCHEMES:
                        return self._json({"error": "use an http(s), rtsp, rtmp, srt or udp stream URL"}, 400)
                    if re.search(r"(youtube\.com|youtu\.be)/", url):
                        return self._json({"error": "YouTube page links are not video streams; paste a direct "
                                                    ".mp4 / .m3u8 / rtsp:// URL"}, 400)
                    r = open_source(url, "url", url[:80])
                elif "webcam" in b:
                    r = open_source(str(int(b["webcam"])), "webcam", f"Webcam {int(b['webcam'])}")
                elif "network" in b:                 # back to the stream arriving from another machine
                    rx = ENGINE.rx
                    if rx is None or not rx.peer or time.time() - rx.last_rx > 2:
                        return self._json({"error": "no sender is streaming to this machine right now; run "
                                                    "python -m phantom.send on the other machine"}, 400)
                    r = open_source(rx.peer[0], "network", "")
                else:
                    return self._json({"error": "nothing to open"}, 400)
                self._json(r, 200 if "source" in r else 400)
            elif u.path == "/api/upload":
                self._upload()
            else:
                self._json({"error": "not found"}, 404)
        except (ValueError, json.JSONDecodeError) as e:
            self._json({"error": str(e)}, 400)

    def _upload(self):
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > 4 << 30:
            return self._json({"error": "file must be between 1 byte and 4 GB"}, 400)
        name = re.sub(r"[^A-Za-z0-9._-]", "_", self.headers.get("X-Filename", "upload.mp4"))[-80:] or "upload.mp4"
        UPLOADS.mkdir(parents=True, exist_ok=True)
        path = UPLOADS / f"{int(time.time())}_{name}"
        left = n
        with open(path, "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            return self._json({"error": "upload interrupted"}, 400)
        r = open_source(str(path), "file", self.headers.get("X-Filename", name)[:80])
        self._json(r, 200 if "source" in r else 400)

    # ---------------------------------------------------------------- streams
    def _frames(self, fps=30.0):
        """Yield the newest composite at a steady fps (repeats frames if the engine is slower)."""
        q = ENGINE.subscribe()
        last = ENGINE.latest if ENGINE.latest is not None else np.zeros((HEIGHT, 2 * WIDTH, 3), np.uint8)
        period, t_next = 1.0 / fps, time.perf_counter()
        try:
            while True:
                try:
                    while True:
                        last = q.get_nowait()
                except queue.Empty:
                    pass
                yield last
                t_next += period
                dt = t_next - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                else:
                    t_next = time.perf_counter()
        finally:
            ENGINE.unsubscribe(q)

    def _stream_h264(self, quality: str):
        w, h = 2 * WIDTH, HEIGHT
        if quality == "med" and RES == 720:
            scale, rate = ["-vf", "scale=1920:540"], "6M"
        else:
            scale, rate = [], ("10M" if RES == 720 else "4M")
        cmd = [ffmpeg_exe(), "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", "30", "-i", "-", *scale, "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
               "-profile:v", "high", "-level:v", "4.0", "-pix_fmt", "yuv420p", "-g", "30", "-bf", "0",
               "-b:v", rate, "-maxrate", rate, "-bufsize", "4M",
               "-f", "mp4", "-movflags", "empty_moov+default_base_moof+frag_keyframe", "-frag_duration", "100000", "-"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        stop = threading.Event()

        def feed():
            try:
                for frame in self._frames():
                    if stop.is_set():
                        break
                    proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        threading.Thread(target=feed, daemon=True).start()
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = proc.stdout.read1(1 << 16)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            stop.set()
            proc.kill()
            self.close_connection = True

    def _stream_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for frame in self._frames(fps=25):
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        self.close_connection = True


def main():
    global ENGINE, SAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on; 0.0.0.0 makes it reachable from other devices on the network")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--weights", default="weights/repairnet.pt")
    ap.add_argument("--framegen", default="weights/framegen.pt")
    ap.add_argument("--udp-port", type=int, default=5004,
                    help="UDP port for streams from another machine (python -m phantom.send)")
    a = ap.parse_args()
    ENGINE = Engine(a.weights, a.framegen, udp_port=a.udp_port)
    if ENGINE.rx is not None:
        print(f"listening for senders on UDP {a.udp_port}; on another machine run:  python -m phantom.send"
              f"   (or --to {ENGINE.ips[0] if ENGINE.ips else '<this IP>'})", flush=True)
    else:
        print(f"[warn] {ENGINE.net_error}", flush=True)
    SAMPLES = find_samples()
    if SAMPLES:
        first = next(iter(SAMPLES))
        print("default source:", open_source(str(SAMPLES[first]), "file", NICE.get(first, first)))
    print(f"Phantom Frames demo at {WIDTH}x{HEIGHT}: http://{a.host}:{a.port}/   models: {list(ENGINE.models)}",
          flush=True)
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
