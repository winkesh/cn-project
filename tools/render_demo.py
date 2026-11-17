"""Render the submission demo video (1920x1080, H.264).

Every frame of "without" and "with Phantom" comes from the real pipeline: the same packetizer, lossy
channel, receiver and RepairNet as the live demo and the evaluation, run through the simulator at 720p
(PHANTOM_RES=720). Both sides of a comparison see exactly the same packets lost.

    PHANTOM_RES=720 python -m tools.render_demo --clips data720/test --out results/phantom_frames_demo.mp4
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom import conceal as C  # noqa: E402
from phantom.config import FPS, N_TILES, RES, SchemeConfig  # noqa: E402
from phantom.sim import NetConfig, simulate  # noqa: E402
from phantom.video import load_video  # noqa: E402

W, H = 1920, 1080
BG = (21, 17, 14)            # BGR, near-black blue
PANEL = (38, 32, 28)
WHITE = (245, 245, 245)
MUTED = (165, 160, 155)
RED = (77, 72, 229)          # Phantom accent  (#E5484D)
GREY = (140, 140, 140)       # baseline accent
GREEN = (120, 200, 90)

FONT_DIRS = ["/usr/share/fonts/truetype/lato", "/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts"]
FONT_FILES = {"black": ["Lato-Black.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"],
              "bold": ["Lato-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"],
              "regular": ["Lato-Regular.ttf", "DejaVuSans.ttf", "arial.ttf"],
              "light": ["Lato-Light.ttf", "DejaVuSans.ttf", "arial.ttf"]}
_fonts: dict = {}


def font(weight: str, size: int):
    key = (weight, size)
    if key not in _fonts:
        for name in FONT_FILES[weight]:
            for d in FONT_DIRS:
                p = Path(d) / name
                if p.exists():
                    _fonts[key] = ImageFont.truetype(str(p), size)
                    break
            if key in _fonts:
                break
        else:
            _fonts[key] = ImageFont.load_default()
    return _fonts[key]


class Text:
    """Collect text draws for one frame and render them in a single PIL pass."""

    def __init__(self):
        self.ops = []

    def __call__(self, s, xy, size=28, color=WHITE, weight="regular", anchor="la"):
        self.ops.append((s, xy, size, color, weight, anchor))

    def width(self, s, size=28, weight="regular"):
        return font(weight, size).getlength(s)

    def render(self, img):
        if not self.ops:
            return img
        pil = Image.fromarray(img[..., ::-1])
        d = ImageDraw.Draw(pil)
        for s, xy, size, color, weight, anchor in self.ops:
            d.text(xy, s, font=font(weight, size), fill=tuple(int(c) for c in color[::-1]), anchor=anchor)
        return np.ascontiguousarray(np.asarray(pil)[..., ::-1])


def canvas():
    return np.full((H, W, 3), BG, np.uint8)


def rrect(img, x0, y0, x1, y1, color, r=14, thickness=-1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1, cv2.LINE_AA)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)


def pill(img, t, x, y, s, size, fg, bg, weight="bold", pad=14, anchor_right=False):
    w = t.width(s, size, weight) + 2 * pad
    h = size + pad + 4
    if anchor_right:
        x = x - w
    rrect(img, x, y, x + w, y + h, bg, r=h // 2)
    t(s, (x + pad, y + h / 2), size, fg, weight, "lm")
    return w


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


class Writer:
    def __init__(self, path: str, crf: int = 18):
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            exe = "ffmpeg"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(
            [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        self.n = 0
        self.last = None

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())
        self.n += 1
        self.last = img

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def fade(img, a):
    """a=1 -> img, a=0 -> background."""
    if a >= 1:
        return img
    return cv2.addWeighted(img, a, np.full_like(img, BG), 1 - a, 0)


def card(wr, draw_fn, seconds, fade_s=0.5):
    n = int(seconds * FPS)
    base = None
    for i in range(n):
        if base is None:
            img, t = canvas(), Text()
            draw_fn(img, t)
            base = t.render(img)
        a = min(1.0, (i + 1) / (fade_s * FPS), (n - i) / (fade_s * FPS))
        wr.write(fade(base, ease(a)))


# ---------------------------------------------------------------------------------------------- cards

def title_card(img, t):
    t("PHANTOM FRAMES", (W / 2, 430), 120, WHITE, "black", "mm")
    t("Generative loss recovery for low-latency video streaming", (W / 2, 540), 40, MUTED, "light", "mm")
    cv2.line(img, (W // 2 - 90, 610), (W // 2 + 90, 610), RED, 4, cv2.LINE_AA)
    t("Computer Networks project", (W / 2, 670), 28, MUTED, "regular", "mm")


def problem_card(img, t):
    y = 300
    lines = [("Netflix and YouTube never show you a broken frame.", WHITE, "bold"),
             ("They hide packet loss behind TCP retransmission and 10–30 seconds of buffering.", MUTED, "regular"),
             ("", WHITE, "regular"),
             ("Live sports, cloud gaming and video calls can't wait that long.", WHITE, "bold"),
             ("With a 100 ms budget there is no time to retransmit — lost packets mean a broken picture.", MUTED,
              "regular"),
             ("", WHITE, "regular"),
             ("Phantom Frames: never retransmit. Let the receiver synthesise what the network lost.", RED, "bold")]
    for s, c, w in lines:
        if s:
            t(s, (W / 2, y), 40 if w == "bold" else 32, c, w, "mm")
        y += 70


def how_card(img, t):
    t("How it works", (W / 2, 170), 56, WHITE, "black", "mm")
    boxes = [("Video frame", "720p, 30 fps"), ("JPEG tiles", "576 × 40 px"), ("RTP / UDP", "tiles interleaved\nacross packets"),
             ("Lossy network", "random + bursty\nloss, no retx"), ("Receiver", "100 ms playout\ndeadline"),
             ("RepairNet", "fills only the\nlost tiles")]
    n, bw, bh, gap = len(boxes), 250, 170, 50
    x = (W - (n * bw + (n - 1) * gap)) / 2
    y = 400
    for i, (h1, h2) in enumerate(boxes):
        col = RED if i == n - 1 else (60, 52, 46)
        rrect(img, x, y, x + bw, y + bh, col, 18)
        t(h1, (x + bw / 2, y + 52), 32, WHITE, "bold", "mm")
        for j, ln in enumerate(h2.split("\n")):
            t(ln, (x + bw / 2, y + 100 + j * 30), 24, (230, 230, 230) if i == n - 1 else MUTED, "regular", "mm")
        if i < n - 1:
            ax = int(x + bw + 8)
            cv2.arrowedLine(img, (ax, y + bh // 2), (ax + gap - 16, y + bh // 2), MUTED, 3, cv2.LINE_AA, tipLength=0.35)
        if i == 3:  # little X marks: lost packets
            for k in range(3):
                cx, cy = int(x + 60 + k * 65), y - 40
                cv2.line(img, (cx - 12, cy - 12), (cx + 12, cy + 12), RED, 4, cv2.LINE_AA)
                cv2.line(img, (cx - 12, cy + 12), (cx + 12, cy - 12), RED, 4, cv2.LINE_AA)
        x += bw + gap
    t("Received pixels are kept bit-exact. The network adds no retransmission delay and no FEC overhead.",
      (W / 2, 720), 32, MUTED, "regular", "mm")
    t("RepairNet: 1.9 M-parameter mask-conditioned U-Net, uses only past frames → zero added latency.",
      (W / 2, 775), 32, MUTED, "regular", "mm")


def results_card_fn(csv: Path, fig: Path):
    import pandas as pd

    stats = []
    if csv.exists():
        d = pd.read_csv(csv)
        L = d[d.experiment == "loss"]
        g = lambda s, x, m: L[(L.scheme == s) & (L.x == x)][m].mean()
        stats = [
            (f"+{g('Phantom (RepairNet)', 10, 'psnr_mean') - g('Copy-tile (no retx)', 10, 'psnr_mean'):.1f} dB",
             "vs standard concealment", "at 10% packet loss"),
            ("0%", f"extra bandwidth", f"FEC needs +{g('FEC k=5 + copy', 5, 'overhead_pct'):.0f}%"),
            (f"{g('Phantom (RepairNet)', 5, 'latency_p95'):.0f} ms",
             "frame latency (P95)", f"TCP: {g('TCP', 5, 'latency_p95'):.0f} ms at 5% loss"),
        ]

    def draw(img, t):
        t("Results", (W / 2, 90), 56, WHITE, "black", "mm")
        t("6 held-out clips · people, motion, sports, film, games · same losses for every scheme",
          (W / 2, 150), 28, MUTED, "regular", "mm")
        if fig.exists():
            f = cv2.imread(str(fig))
            f = f[:, : f.shape[1] // 2]                      # quality-vs-RTT panel
            s = 640 / f.shape[0]
            f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            x0 = 90
            rrect(img, x0 - 10, 210, x0 + f.shape[1] + 10, 210 + f.shape[0] + 20, (255, 255, 255), 12)
            img[220:220 + f.shape[0], x0:x0 + f.shape[1]] = f
            t("Retransmission (NACK) collapses once RTT reaches the buffer; Phantom does not.",
              (x0 + f.shape[1] / 2, 900), 26, MUTED, "regular", "mm")
            sx = x0 + f.shape[1] + 90
        else:
            sx = 600
        y = 240
        for big, l1, l2 in stats:
            rrect(img, sx, y, W - 90, y + 190, PANEL, 18)
            t(big, (sx + 40, y + 95), 84, RED, "black", "lm")
            t(l1, (sx + 420, y + 75), 34, WHITE, "bold", "lm")
            t(l2, (sx + 420, y + 125), 28, MUTED, "regular", "lm")
            y += 220
    return draw


def end_card(img, t):
    t("PHANTOM FRAMES", (W / 2, 420), 96, WHITE, "black", "mm")
    t("github.com/winkesh/cn-project", (W / 2, 520), 36, MUTED, "regular", "mm")
    cv2.line(img, (W // 2 - 90, 580), (W // 2 + 90, 580), RED, 4, cv2.LINE_AA)
    t("Every comparison in this video was produced by the real packetizer, lossy channel, receiver and RepairNet",
      (W / 2, 660), 26, MUTED, "regular", "mm")
    t("(discrete-event network emulation, rendered offline at 720p). Both sides always lose exactly the same packets.",
      (W / 2, 700), 26, MUTED, "regular", "mm")


# ------------------------------------------------------------------------------------------- segments

def run_pair(frames, baseline: str, net: NetConfig, phantom):
    if net.schedule is None:   # first frame arrives clean (a stream starts from a good keyframe)
        net.schedule = ((0, 0.0), (30, 0.0), (31, net.loss), (1e9, net.loss))
    b, out_b = simulate(frames, SchemeConfig("b", "none", baseline), net, concealer=C.make(baseline),
                        keep_outputs=True, compute_ssim=False)
    phantom.times_ms.clear()
    p, out_p = simulate(frames, SchemeConfig("p", "none", "repairnet"), net, concealer=phantom,
                        keep_outputs=True, compute_ssim=False)
    return out_b, out_p, b["per_frame"], p["per_frame"], list(phantom.times_ms) or [0.0]


def intro(wr, frame0_seq, title, source):
    """Original clip full-screen for ~1.5 s with its caption."""
    n = len(frame0_seq)
    for i, f in enumerate(frame0_seq):
        img = cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC)
        t = Text()
        a = ease(min(1.0, (i + 1) / 10))
        img[H - 170:] = (img[H - 170:] * 0.4).astype(np.uint8)       # darker band behind the caption
        pill(img, t, 60, 50, "ORIGINAL", 26, WHITE, (40, 40, 40))
        t(title, (60, H - 110), 48, WHITE, "black", "lm")
        t(source, (60, H - 55), 28, MUTED, "regular", "lm")
        wr.write(fade(t.render(img), a if i < n - 6 else ease((n - i) / 6)))


def graph(img, t, x0, y0, x1, y1, ser_b, ser_p, loss, k, n, lo, hi, lbl_b):
    rrect(img, x0, y0, x1, y1, PANEL, 16)
    gx0, gx1, gy0, gy1 = x0 + 90, x1 - 30, y0 + 50, y1 - 45
    X = lambda i: gx0 + (gx1 - gx0) * i / max(1, n - 1)
    Y = lambda v: gy1 - (gy1 - gy0) * (min(hi, max(lo, v)) - lo) / (hi - lo)
    # loss bars (fraction of tiles lost), scaled to the bottom 40 % of the plot
    for i in range(k + 1):
        if loss[i] > 0:
            h = (gy1 - gy0) * 0.4 * min(1.0, loss[i] / 0.4)
            cv2.line(img, (int(X(i)), gy1), (int(X(i)), int(gy1 - h)), (55, 50, 95), 3)
    for v in np.arange(math.ceil(lo / 5) * 5, hi + 0.1, 5):
        cv2.line(img, (gx0, int(Y(v))), (gx1, int(Y(v))), (60, 55, 50), 1, cv2.LINE_AA)
        t(f"{v:.0f}", (gx0 - 14, Y(v)), 20, MUTED, "regular", "rm")
    t("PSNR (dB)", (x0 + 24, y0 + 26), 22, MUTED, "bold", "lm")
    lx = x0 + 200
    for col, name in ((GREY, lbl_b.title()), (RED, "Phantom Frames"), ((55, 50, 95), "tiles lost per frame")):
        cv2.line(img, (lx, y0 + 26), (lx + 34, y0 + 26), col, 5, cv2.LINE_AA)
        t(name, (lx + 46, y0 + 26), 20, MUTED, "regular", "lm")
        lx += 80 + int(t.width(name, 20))
    for ser, col in ((ser_b, GREY), (ser_p, RED)):
        pts = np.array([[X(i), Y(ser[i])] for i in range(k + 1)], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, col, 3, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in pts[-1]), 7, col, -1, cv2.LINE_AA)
    cv2.line(img, (int(X(k)), gy0), (int(X(k)), gy1), (90, 85, 80), 1)


def split_segment(wr, frames, out_b, out_p, rows_b, rows_p, inf_ms, title, sub, lbl_b, loss_txt, start=0):
    n = len(frames)
    pw, ph = 930, 523
    xs = (20, W - 20 - pw)
    py = 150
    ser_b = [r["psnr"] for r in rows_b]
    ser_p = [r["psnr"] for r in rows_p]
    loss = [r["missing_tiles"] / N_TILES for r in rows_p]
    lo = max(5.0, math.floor(min(ser_b[start:] + ser_p[start:]) / 5) * 5)
    hi = min(55.0, math.ceil(max(ser_b[start:] + ser_p[start:]) / 5) * 5 + 2)
    sb, sp = deque(maxlen=15), deque(maxlen=15)
    m = n - start
    for i in range(start, n):
        img, t = canvas(), Text()
        t(title, (40, 52), 46, WHITE, "black", "lm")
        t(sub, (40, 104), 26, MUTED, "regular", "lm")
        lt = loss_txt(i) if callable(loss_txt) else loss_txt
        pill(img, t, W - 40, 34, lt, 28, WHITE, RED, anchor_right=True)
        t("no retransmission · 100 ms playout deadline", (W - 40, 112), 24, MUTED, "regular", "rm")
        for x, out, col, name in ((xs[0], out_b[i], GREY, lbl_b), (xs[1], out_p[i], RED, "PHANTOM FRAMES")):
            rrect(img, x - 4, py - 4, x + pw + 4, py + ph + 4, col, 10)
            img[py:py + ph, x:x + pw] = cv2.resize(out, (pw, ph), interpolation=cv2.INTER_AREA)
            pill(img, t, x + 16, py + 16, name, 24, WHITE, col if col == RED else (60, 60, 60))
        sb.append(ser_b[i])
        sp.append(ser_p[i])
        miss = rows_p[i]["missing_tiles"]
        yb = py + ph + 22
        t(f"PSNR {np.mean(sb):.1f} dB", (xs[0], yb + 20), 36, WHITE, "bold", "lm")
        t(f"{miss} of {N_TILES} tiles lost this frame", (xs[0] + pw, yb + 20), 26, MUTED, "regular", "rm")
        t(f"PSNR {np.mean(sp):.1f} dB", (xs[1], yb + 20), 36, RED, "bold", "lm")
        rep = f"{miss} tiles synthesised · {np.median(inf_ms):.0f} ms" if miss else "nothing lost this frame"
        t(rep, (xs[1] + pw, yb + 20), 26, MUTED, "regular", "rm")
        graph(img, t, 20, 740, W - 20, H - 20, ser_b, ser_p, loss, i, n, lo, hi, lbl_b)
        out = t.render(img)
        k = i - start
        a = ease(min(1.0, (k + 1) / 12, (m - k) / 12))
        wr.write(fade(out, a))


def wipe_segment(wr, frames, out_b, out_p, rows_p, lbl_b, loss_txt, title, start=0):
    n = len(frames)
    m = n - start
    for i in range(start, n):
        k = i - start
        u = k / FPS
        xdiv = int(W / 2 + 0.36 * W * math.sin(2 * math.pi * u / 6.0))
        a_img = cv2.resize(out_b[i], (W, H), interpolation=cv2.INTER_CUBIC)
        b_img = cv2.resize(out_p[i], (W, H), interpolation=cv2.INTER_CUBIC)
        img = b_img.copy()
        img[:, :xdiv] = a_img[:, :xdiv]
        cv2.line(img, (xdiv, 0), (xdiv, H), WHITE, 4, cv2.LINE_AA)
        cv2.circle(img, (xdiv, H // 2), 30, WHITE, -1, cv2.LINE_AA)
        cv2.arrowedLine(img, (xdiv - 6, H // 2), (xdiv - 22, H // 2), (40, 40, 40), 3, cv2.LINE_AA, tipLength=0.6)
        cv2.arrowedLine(img, (xdiv + 6, H // 2), (xdiv + 22, H // 2), (40, 40, 40), 3, cv2.LINE_AA, tipLength=0.6)
        t = Text()
        pill(img, t, 40, 40, lbl_b, 28, WHITE, (50, 50, 50))
        pill(img, t, W - 40, 40, "PHANTOM FRAMES", 28, WHITE, RED, anchor_right=True)
        img[H - 120:] = (img[H - 120:] * 0.4).astype(np.uint8)       # darker band behind the caption
        t(title, (40, H - 60), 40, WHITE, "black", "lm")
        lt = loss_txt(i) if callable(loss_txt) else loss_txt
        t(f"{lt} · same packets lost on both sides · {rows_p[i]['missing_tiles']} tiles repaired this frame",
          (W - 40, H - 60), 26, MUTED, "regular", "rm")
        a = ease(min(1.0, (k + 1) / 12, (m - k) / 12))
        wr.write(fade(t.render(img), a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data720/test")
    ap.add_argument("--weights", default="weights/repairnet.pt")
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--fig", default="results/figs/e3_rtt.png")
    ap.add_argument("--out", default="results/phantom_frames_demo.mp4")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--quick", action="store_true", help="short preview: 90 frames per clip, no cards")
    a = ap.parse_args()
    if RES != 720:
        print("[warn] rendering at 360p; set PHANTOM_RES=720 for the submission video")
    clips = Path(a.clips)
    nf = 90 if a.quick else a.frames
    phantom = C.RepairNetConcealer(a.weights)
    wr = Writer(a.out)

    if not a.quick:
        card(wr, title_card, 4.0)
        card(wr, problem_card, 8.0)
        card(wr, how_card, 8.0)

    INTRO = 45
    segs = [
        dict(clip="game_forza", kind="split", baseline="none", lbl="NO REPAIR",
             net=NetConfig(loss=0.10, seed=1), loss_txt="10% PACKET LOSS",
             title="Racing game, 10% packet loss", sub="Left: a receiver that cannot retransmit and does nothing. "
                                                       "Right: Phantom Frames.",
             src="Forza Horizon 5 gameplay · 720p · 30 fps"),
        dict(clip="park_joy", kind="split", baseline="copy", lbl="STANDARD CONCEALMENT",
             net=NetConfig(loss=0.0, seed=2, schedule=((0, 0.0), (nf * 1000 / FPS * 0.9, 0.30))),
             loss_txt=None, title="Fast motion, loss ramping 0% → 30%",
             sub="Left: standard error concealment (re-use the last frame's tile). Right: Phantom Frames.",
             src="park_joy · SVT / Xiph.org test sequence"),
        dict(clip="game_mw3", kind="wipe", baseline="none", lbl="NO REPAIR",
             net=NetConfig(loss=0.15, loss_model="ge", burst=4, seed=3), loss_txt="15% bursty loss",
             title="Shooter, bursty loss", src="Call of Duty: Modern Warfare III gameplay · 720p"),
        dict(clip="tears_of_steel", kind="split", baseline="freeze", lbl="STANDARD DECODER (FREEZE)",
             net=NetConfig(loss=0.01, seed=4), loss_txt="1% PACKET LOSS",
             title="Film, only 1% packet loss", sub="Left: a normal decoder drops every damaged frame, so the "
                                                    "picture stutters. Right: Phantom Frames.",
             src="Tears of Steel · Blender Foundation · CC-BY"),
        dict(clip="FourPeople", kind="split", baseline="copy", lbl="STANDARD CONCEALMENT",
             net=NetConfig(loss=0.20, seed=5), loss_txt="20% PACKET LOSS",
             title="Video call, 20% packet loss", sub="Left: standard error concealment. Right: Phantom Frames.",
             src="FourPeople · Xiph.org video-conferencing test sequence"),
    ]
    for s in segs:
        path = next((p for p in clips.glob(s["clip"] + ".*")), None)
        if path is None:
            print(f"[skip] {s['clip']} not found in {clips}")
            continue
        frames = load_video(path, max_frames=nf)
        print(f"{s['clip']}: {len(frames)} frames, simulating...", flush=True)
        out_b, out_p, rb, rp, inf = run_pair(frames, s["baseline"], s["net"], phantom)
        if s["loss_txt"] is None:
            sched = s["net"].schedule
            s["loss_txt"] = lambda i, sched=sched: (
                f"{100 * np.interp(i * 1000 / FPS, [p[0] for p in sched], [p[1] for p in sched]):.0f}% PACKET LOSS")
        if not a.quick:
            intro(wr, frames[:INTRO], s["title"], s["src"])
        start = 0 if a.quick else INTRO
        if s["kind"] == "split":
            split_segment(wr, frames, out_b, out_p, rb, rp, inf, s["title"], s["sub"], s["lbl"], s["loss_txt"], start)
        else:
            wipe_segment(wr, frames, out_b, out_p, rp, s["lbl"], s["loss_txt"], s["title"], start)
        print(f"  PSNR without {np.mean([r['psnr'] for r in rb]):.2f}  with {np.mean([r['psnr'] for r in rp]):.2f}"
              f"  infer p95 {np.percentile(inf, 95):.1f} ms", flush=True)

    if not a.quick:
        card(wr, results_card_fn(Path(a.results), Path(a.fig)), 9.0)
        card(wr, end_card, 5.0)
    wr.close()
    print(f"wrote {a.out}: {wr.n} frames, {wr.n / FPS:.1f} s")


if __name__ == "__main__":
    main()
