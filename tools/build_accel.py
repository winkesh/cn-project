"""Export RepairNet to ONNX, build TensorRT FP16 + INT8 engines, and verify accuracy and speed of every backend.

    python -m tools.build_accel              # 360p engines (also used for low-res repair of 720p/1080p)
    python -m tools.build_accel --skip-build # only re-run the comparison

INT8 calibration uses training clips; accuracy is measured on held-out test clips, both with loss masks laid
out exactly as the packetizer does. An engine is only worth using if its repaired tiles stay close to the
PyTorch reference (framegen's TensorRT engines lost 3-7 dB, so this is checked, not assumed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom import accel  # noqa: E402
from phantom.config import HEIGHT, WIDTH  # noqa: E402
from phantom.video import list_videos, load_video  # noqa: E402
from train.train_repairnet import Clips  # noqa: E402


def samples(folder: str, n: int, seed: int, max_frames: int = 150):
    vids = [load_video(v, max_frames=max_frames) for v in list_videos(folder)]
    clips = Clips([v for v in vids if v], seed=seed)
    out = []
    for _ in range(n):
        p2, p1, cur, m, gt = clips.sample(1)          # RGB float [0,1], NCHW
        out.append({"prev2": p2.numpy(), "prev1": p1.numpy(), "partial": cur.numpy(), "mask": m.numpy(),
                    "gt": gt.numpy()})
    return out


def masked_psnr(pred, gt, m):
    se = ((pred - gt) ** 2 * m).sum() / max(1.0, m.sum() * 3)
    return float(10 * np.log10(1.0 / max(se, 1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/repairnet.pt")
    ap.add_argument("--calib", default="data/train", help="clips for INT8 calibration")
    ap.add_argument("--test", default="data/test", help="held-out clips for the accuracy check")
    ap.add_argument("--n-calib", type=int, default=48)
    ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--stage", default="all", choices=["all", "export", "engines", "verify"],
                    help="run one step at a time (gentler on a laptop GPU)")
    ap.add_argument("--opt-level", type=int, default=2, help="TensorRT builder optimization level (0-5)")
    ap.add_argument("--dml", action="store_true", help="also test ONNX Runtime DirectML on each GPU")
    ap.add_argument("--portable", action="store_true",
                    help="engines that run on any Ampere-or-newer NVIDIA GPU (build on one machine, use on another)")
    ap.add_argument("--max-frames", type=int, default=80, help="frames loaded per clip")
    ap.add_argument("--precisions", default="fp16,int8fp16",
                    help="engines to build: fp16, int8fp16 (INT8 convs + FP16 rest), int8 (INT8 + FP32 rest)")
    ap.add_argument("--out", default="results/accel.json")
    a = ap.parse_args()
    import torch

    h, w = HEIGHT, WIDTH
    onnx = {v: accel.onnx_path(h, w, variant=v) for v in ("fp32", "fp16", "int8", "int8fp16")}
    eng = {p: accel.engine_path(h, w, p, portable=a.portable) for p in ("fp16", "int8", "int8fp16")}
    if not a.skip_build and a.stage in ("all", "export"):
        print(f"exporting ONNX {w}x{h}: FP32 and FP16", flush=True)
        accel.export_onnx(a.weights, h, w, onnx["fp32"])
        accel.export_onnx(a.weights, h, w, onnx["fp16"], half=True)
        print(f"INT8: calibrating on {a.n_calib} samples from {a.calib} and inserting quantize/dequantize nodes",
              flush=True)
        cal = samples(a.calib, a.n_calib, seed=11, max_frames=a.max_frames)
        accel.quantize_int8(onnx["fp32"], onnx["int8"], [{k: v for k, v in s.items() if k != "gt"} for s in cal])
        del cal
        accel.int8_to_fp16(onnx["int8"], onnx["int8fp16"])     # INT8 convolutions, FP16 everywhere else
    if not a.skip_build and a.stage in ("all", "engines"):
        for p in a.precisions.split(","):
            print(f"building TensorRT {p.upper()} engine (optimization level {a.opt_level}) ...", flush=True)
            accel.build_engine(onnx[p], eng[p], a.opt_level, portable=a.portable)
            print(f"  -> {eng[p]}", flush=True)
    if a.stage in ("export", "engines"):
        return

    test = samples(a.test, a.n_test, seed=12345, max_frames=a.max_frames)
    print(f"verifying on {len(test)} held-out samples from {a.test}", flush=True)

    # ---- backends
    from phantom.models.repairnet import RepairNet
    ck = torch.load(a.weights, map_location="cpu")
    net = RepairNet(ck.get("base", 32)); net.load_state_dict(ck["model"])
    net = net.cuda().eval().half().to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    def run_torch(s):
        t = [torch.from_numpy(s[k]).cuda().half().contiguous(memory_format=torch.channels_last) for k in accel.INPUTS]
        with torch.no_grad():
            return net(*t).float()

    backends = {"PyTorch FP16": (run_torch, "torch")}
    trt_names = {"fp16": "TensorRT FP16", "int8": "TensorRT INT8+FP32", "int8fp16": "TensorRT INT8+FP16"}
    for p, name in trt_names.items():
        if eng[p].exists():
            r = accel.TRTRunner(eng[p])
            backends[name] = (lambda s, r=r: r(*[torch.from_numpy(s[k]).cuda() for k in accel.INPUTS]), "torch")
    dml = {}
    try:
        import onnxruntime as ort
        if a.dml and "DmlExecutionProvider" in ort.get_available_providers():
            for dev in (0, 1):
                for v in ("fp32", "fp16"):
                    try:
                        r = accel.DMLRunner(onnx[v], dev)
                        name = f"DirectML GPU{dev} {v.upper()}"
                        backends[name] = (lambda s, r=r: r(*[s[k] for k in accel.INPUTS]), "numpy")
                        dml[name] = (dev, v)
                    except Exception as e:
                        print(f"DirectML device {dev} {v}: {str(e)[:120]}")
    except ImportError:
        print("onnxruntime-directml not installed: skipping DirectML")

    def to_np(y, kind):
        return y.float().cpu().numpy() if kind == "torch" else y.astype(np.float32)

    ref = [to_np(backends["PyTorch FP16"][0](s), "torch") for s in test]
    rows = {}
    copy_q = np.mean([masked_psnr(s["partial"], s["gt"], s["mask"]) for s in test])
    for name, (fn, kind) in backends.items():
        outs = [to_np(fn(s), kind).copy() for s in test]
        q = np.mean([masked_psnr(o, s["gt"], s["mask"]) for o, s in zip(outs, test)])
        dev = np.mean([masked_psnr(o, r, s["mask"]) for o, r, s in zip(outs, ref, test)])
        # speed: same input repeatedly, synchronised
        s0 = test[0]
        for _ in range(10):
            fn(s0)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(60):
            y = fn(s0)
        if kind == "torch":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 60 * 1000
        rows[name] = dict(masked_psnr=round(float(q), 2), vs_pytorch_db=round(float(dev), 1), ms=round(ms, 2))
    print(f"\n{w}x{h}, {len(test)} held-out samples   (copy-tile baseline: {copy_q:.2f} dB in lost regions)")
    print(f"{'backend':16s} {'lost-region PSNR':>17s} {'agreement w/ PyTorch':>21s} {'time/frame':>11s}")
    for k, r in rows.items():
        agree = "reference" if k == "PyTorch FP16" else f"{r['vs_pytorch_db']:.1f} dB"
        print(f"{k:16s} {r['masked_psnr']:14.2f} dB {agree:>21s} {r['ms']:8.2f} ms")
    # recommend INT8 only if it is faster AND its repairs stay within 0.3 dB of PyTorch in the lost regions
    ref_q = rows["PyTorch FP16"]["masked_psnr"]
    # the fastest TensorRT engine whose repairs stay within 0.3 dB of PyTorch in the lost regions
    ok = {p: rows[n] for p, n in trt_names.items() if n in rows and rows[n]["masked_psnr"] >= ref_q - 0.3}
    rec = min(ok, key=lambda p: ok[p]["ms"]) if ok else "fp16"
    good_dml = {k: rows[k] for k in dml if rows[k]["masked_psnr"] >= ref_q - 0.3}
    best_dml = min(good_dml, key=lambda k: good_dml[k]["ms"]) if good_dml else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    json.dump(dict(gpu=gpu, res=[w, h], copy_tile_db=round(float(copy_q), 2), backends=rows, recommended=rec,
                   dml_device=dml[best_dml][0] if best_dml else 0, dml_variant=dml[best_dml][1] if best_dml else "fp32"),
              open(a.out, "w"), indent=2)
    print(f"\nTensorRT precision to use: {rec.upper()}" + (f"   fastest accurate DirectML: {best_dml}" if best_dml else ""))
    print(f"wrote {a.out}  (GPU: {gpu})")


if __name__ == "__main__":
    main()
