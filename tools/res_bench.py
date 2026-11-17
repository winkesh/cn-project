"""How high can the demo go on this machine? Times each stage of the demo pipeline per resolution.

    python -m tools.res_bench                 # 360p, 720p, 1080p, each in its own process

The demo runs three stages at once (sender: tiles + JPEG + packets + loss; receiver: decode + repair + compose;
browser stream: H.264 encode of the side-by-side picture), so its frame rate is set by the slowest stage.
Content: a textured clip (worst case for JPEG) and a real game clip.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bench_one(n: int = 40) -> dict:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from phantom.app.engine import Engine
    from phantom.config import HEIGHT, RES, WIDTH
    from phantom.tiles import letterbox
    from phantom.video import load_video, synthetic

    e = Engine(udp_port=0)                       # no clash with a running demo
    e.source = None
    e.update(loss=0.10)
    rn = e.models.get("repairnet")
    clips = {"textured (worst case)": synthetic(n)}
    game = ROOT / "data" / "test" / "game_forza.avi"
    if game.exists():
        clips["game clip"] = [letterbox(f) for f in load_video(game, max_frames=n)]
    out = dict(res=RES, size=[WIDTH, HEIGHT], repair=rn.label if rn else "-", clips={})
    for name, frames in clips.items():
        for f in frames[:5]:                     # warm-up (cuDNN / TensorRT / thread pools)
            e._recv(e._send(f))
        ts, tr = [], []
        for f in frames:
            t0 = time.perf_counter(); cur = e._send(f); t1 = time.perf_counter()
            e._recv(cur); t2 = time.perf_counter()
            ts.append(t1 - t0); tr.append(t2 - t1)
        out["clips"][name] = dict(send_ms=round(1000 * float(np.median(ts)), 1),
                                  recv_ms=round(1000 * float(np.median(tr)), 1),
                                  pkts=round(float(np.mean([r["pkts"] for r in list(e.stats)[-n:]]))))
    # browser stream: H.264 encode of the side-by-side composite, same settings as the server
    import imageio_ffmpeg
    w, h = 2 * WIDTH, HEIGHT
    level = "5.1" if w * h > 8192 * 256 else "4.0"
    comp = e.latest if e.latest is not None else np.zeros((h, w, 3), np.uint8)
    p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                          "-tune", "zerolatency", "-profile:v", "high", "-level:v", level, "-pix_fmt", "yuv420p",
                          "-g", "30", "-bf", "0", "-f", "null", "-"], stdin=subprocess.PIPE)
    frames = [comp, np.ascontiguousarray(comp[:, ::-1])]   # alternate so the encoder cannot skip work
    t = time.perf_counter()
    for i in range(60):
        p.stdin.write(frames[i % 2].tobytes())
    p.stdin.close(); p.wait()
    out["encode_ms"] = round((time.perf_counter() - t) / 60 * 1000, 1)
    return out


def main():
    if os.environ.get("RES_BENCH_CHILD"):
        print("RESULT " + json.dumps(bench_one()), flush=True)
        return
    rows = []
    wanted = [int(x) for x in sys.argv[1:]] or [360, 720, 1080]
    for res in wanted:
        env = dict(os.environ, PHANTOM_RES=str(res), RES_BENCH_CHILD="1")
        r = subprocess.run([sys.executable, "-m", "tools.res_bench"], cwd=ROOT, env=env, capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(f"{res}p failed:\n{r.stderr[-800:]}")
            continue
        rows.append(json.loads(line[7:]))
        time.sleep(2)                            # let the GPU rest between runs
    print(f"\n{'res':>6s} {'repair':28s} {'content':22s} {'pkts':>5s} {'sender':>8s} {'receiver':>9s} "
          f"{'H.264':>7s} {'=> fps':>7s}")
    for r in rows:
        for name, c in r["clips"].items():
            slowest = max(c["send_ms"], c["recv_ms"], r["encode_ms"])
            print(f"{str(r['res']) + 'p':>6s} {r['repair']:28s} {name:22s} {c['pkts']:5d} {c['send_ms']:6.1f}ms "
                  f"{c['recv_ms']:7.1f}ms {r['encode_ms']:5.1f}ms {min(1000 / slowest, 999):6.0f}")
    Path(ROOT / "results").mkdir(exist_ok=True)
    json.dump(rows, open(ROOT / "results" / "res_bench.json", "w"), indent=2)
    print("\nfps = 1000 / slowest stage (the stages run in parallel); capped by the source at 30 fps in the demo")


if __name__ == "__main__":
    main()
