"""Download standard video test sequences (Xiph.org derf collection, Blender Sintel trailer).

Raw .y4m files are streamed and converted on the fly to 640x360 MJPEG (q95) at ~30 fps, so the
~800 MB originals never touch the disk.

    python -m tools.fetch_clips             # all clips, 3 in parallel
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom.config import HEIGHT, WIDTH  # noqa: E402

XIPH = "https://media.xiph.org/video/derf/y4m/"
BLENDER = "https://download.blender.org/"
CLIPS = {  # name: (url, split, stride, content type, start frame)
    # people / video calls
    "Johnny": (XIPH + "Johnny_1280x720_60.y4m", "train", 2, "people", 0),
    "KristenAndSara": (XIPH + "KristenAndSara_1280x720_60.y4m", "train", 2, "people", 0),
    "vidyo1": (XIPH + "vidyo1_720p_60fps.y4m", "train", 2, "people", 0),
    "vidyo3": (XIPH + "vidyo3_720p_60fps.y4m", "train", 2, "people", 0),
    "vidyo4": (XIPH + "vidyo4_720p_60fps.y4m", "train", 2, "people", 0),
    "FourPeople": (XIPH + "FourPeople_1280x720_60.y4m", "test", 2, "people", 0),
    # motion / sports-like / scenery
    "parkrun": (XIPH + "720p50_parkrun_ter.y4m", "train", 2, "motion", 0),
    "mobcal": (XIPH + "720p50_mobcal_ter.y4m", "train", 2, "motion", 0),
    "ducks_take_off": (XIPH + "ducks_take_off_420_720p50.y4m", "train", 2, "motion", 0),
    "in_to_tree": (XIPH + "in_to_tree_420_720p50.y4m", "train", 2, "motion", 0),
    "old_town_cross": (XIPH + "old_town_cross_420_720p50.y4m", "train", 2, "motion", 0),
    "shields": (XIPH + "720p50_shields_ter.y4m", "test", 2, "motion", 0),
    "park_joy": (XIPH + "park_joy_420_720p50.y4m", "test", 2, "sports", 0),
    # film / animation
    "sintel_trailer": (BLENDER + "durian/trailer/sintel_trailer-480p.mp4", "train", 1, "film", 0),
    "tears_of_steel": (BLENDER + "demo/movies/ToS/tears_of_steel_720p.mov", "test", 1, "film", 2900),
}
CONTENT = {k: v[3] for k, v in CLIPS.items()}
RAW_DIR = Path.home() / ".cache" / "phantom_raw"   # outside OneDrive; each download in its own folder
UA = {"User-Agent": "Mozilla/5.0 (video-research-downloader)"}


def writer(path: Path, fps: float):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (WIDTH, HEIGHT))
    w.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
    return w


def _get_range(url: str, start: int, end: int, tries: int = 5) -> bytes:
    """HTTP range request with retries (survives Wi-Fi drops / DNS hiccups)."""
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == end - start:
                return data
        except Exception:
            if t == tries - 1:
                raise
        time.sleep(2 * (t + 1))
    raise IOError(f"short read {url} [{start}:{end}]")


def fetch_y4m(name, url, out: Path, stride: int, max_frames: int, chunk_frames: int = 40):
    head = _get_range(url, 0, 512)
    hlen = head.index(b"\n") + 1
    header = head[:hlen].decode().split()
    assert header[0] == "YUV4MPEG2", header
    p = {h[0]: h[1:] for h in header[1:]}
    W, H = int(p["W"]), int(p["H"])
    num, den = (int(x) for x in p.get("F", "30:1").split(":"))
    fsize = W * H * 3 // 2
    rec = 6 + fsize                                    # b"FRAME\n" + I420 payload
    assert head[hlen:hlen + 6] == b"FRAME\n", "frame headers with parameters not supported"
    total = int(urllib.request.urlopen(urllib.request.Request(url, headers=UA, method="HEAD"),
                                       timeout=60).headers["Content-Length"])
    n_src = (total - hlen) // rec
    wr = writer(out, num / den / stride)
    n, k = 0, 0
    while n < max_frames and k < n_src:
        k_end = min(n_src, k + chunk_frames)
        data = _get_range(url, hlen + k * rec, hlen + k_end * rec)
        for i in range(k, k_end):
            if i % stride or n >= max_frames:
                continue
            off = (i - k) * rec + 6
            yuv = np.frombuffer(data, np.uint8, fsize, off).reshape(H * 3 // 2, W)
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            wr.write(cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
            n += 1
        k = k_end
    wr.release()
    return n


def fetch_mp4(name, url, out: Path, max_frames: int, start: int = 0):
    raw_dir = RAW_DIR / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / Path(url).name
    if not raw.exists():
        tmp = raw.with_suffix(".part")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(raw)
    cap = cv2.VideoCapture(str(raw))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    wr = writer(out, fps)
    n = 0
    while n < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        if img.mean() < 20:      # trailer has long black/title sections: skip them
            continue
        h, w = img.shape[:2]
        th = int(w * HEIGHT / WIDTH)            # centre-crop to 16:9 then resize
        y0 = max(0, (h - th) // 2)
        wr.write(cv2.resize(img[y0:y0 + th], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
        n += 1
    wr.release()
    return n


def one(name, root: Path, max_frames: int):
    url, split, stride, _, start = CLIPS[name]
    out = root / split / f"{name}.avi"
    if out.exists():
        c = cv2.VideoCapture(str(out))
        have = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        c.release()
        if have >= min(max_frames, 240):
            return f"{name}: exists ({have} frames)"
        out.unlink()  # truncated by an earlier network drop: fetch again
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if url.endswith(".y4m"):
            n = fetch_y4m(name, url, out, stride, max_frames)
        else:
            n = fetch_mp4(name, url, out, max_frames, start)
        return f"{name}: {n} frames -> {out}"
    except Exception as e:  # keep the other downloads going
        return f"{name}: FAILED {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = [n for n in CLIPS if not a.only or n in a.only.split(",")]
    with ThreadPoolExecutor(3) as ex:
        for msg in ex.map(lambda n: one(n, Path(a.root), a.max_frames), names):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
