"""Light check of RepairNet on ONNX Runtime + DirectML (any DirectX 12 GPU on Windows).

    python -m tools.dml_check

For each DirectML device (0, 1, ...) and each ONNX variant (FP32, FP16): a short burst of inferences on real
frames with packetizer-style loss masks. Accuracy is measured against ONNX Runtime on the CPU (no CUDA used);
which device is the NVIDIA GPU is found by watching nvidia-smi while each one runs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom import accel  # noqa: E402
from phantom.config import HEIGHT, WIDTH  # noqa: E402
from tools.build_accel import masked_psnr, samples  # noqa: E402


def nvidia_util(stop, out):
    while not stop.is_set():
        try:
            u = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            out.append(int(u.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.2)


def main():
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test")
    ap.add_argument("--n", type=int, default=6, help="samples for the accuracy check")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--out", default="results/dml.json")
    a = ap.parse_args()
    h, w = HEIGHT, WIDTH
    files = {v: accel.onnx_path(h, w, variant=v) for v in ("fp32", "fp16")}
    files = {v: f for v, f in files.items() if f.exists()}
    if not files:
        sys.exit("no ONNX model: run python -m tools.build_accel --stage export (or copy weights/*.onnx)")
    print(f"providers: {ort.get_available_providers()}   models: {[f.name for f in files.values()]}")

    test = samples(a.test, a.n, seed=12345, max_frames=40)
    feeds = [{k: s[k] for k in accel.INPUTS} for s in test]
    cpu = ort.InferenceSession(str(files.get("fp32", next(iter(files.values())))), providers=["CPUExecutionProvider"])
    ref = [cpu.run([accel.OUTPUT], f)[0] for f in feeds]
    copy_q = np.mean([masked_psnr(s["partial"], s["gt"], s["mask"]) for s in test])
    ref_q = np.mean([masked_psnr(r, s["gt"], s["mask"]) for r, s in zip(ref, test)])
    print(f"{w}x{h}: copy-tile {copy_q:.2f} dB, RepairNet on CPU (reference) {ref_q:.2f} dB in the lost regions\n")

    rows = {}
    for dev in range(3):
        for v, f in files.items():
            try:
                r = accel.DMLRunner(f, dev)
            except Exception as e:
                if v == "fp32":
                    print(f"device {dev}: not available ({str(e)[:80]})")
                break
            outs = [r(*[fd[k] for k in accel.INPUTS]) for fd in feeds]
            q = np.mean([masked_psnr(o, s["gt"], s["mask"]) for o, s in zip(outs, test)])
            agree = np.mean([masked_psnr(o, rr, s["mask"]) for o, rr, s in zip(outs, ref, test)])
            stop, util = threading.Event(), []
            th = threading.Thread(target=nvidia_util, args=(stop, util), daemon=True)
            th.start()
            t = time.perf_counter()
            for i in range(a.iters):
                r(*[feeds[i % len(feeds)][k] for k in accel.INPUTS])
            ms = (time.perf_counter() - t) / a.iters * 1000
            time.sleep(0.4)
            stop.set(); th.join()
            gpu = "NVIDIA RTX 3050" if util and max(util) >= 20 else "integrated GPU (NVIDIA idle)"
            name = f"DirectML device {dev} {v.upper()}"
            rows[name] = dict(device=dev, variant=v, gpu=gpu, ms=round(ms, 1), masked_psnr=round(float(q), 2),
                              agree_db=round(float(agree), 1), nvidia_util_max=max(util) if util else None)
            print(f"{name:26s} {gpu:30s} {ms:7.1f} ms/frame   lost-region {q:5.2f} dB   "
                  f"agreement with CPU {agree:5.1f} dB", flush=True)
            del r
            time.sleep(1.0)                      # let the GPU rest between bursts
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(res=[w, h], copy_tile_db=round(float(copy_q), 2), cpu_reference_db=round(float(ref_q), 2),
                   runs=rows), open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
