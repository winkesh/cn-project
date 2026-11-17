"""Faster RepairNet backends: ONNX export, TensorRT FP16 / INT8 engines, ONNX Runtime DirectML.

    python -m tools.build_accel                     # export ONNX + build TensorRT engines + verify accuracy

Backends (RepairNetConcealer(backend=...)):
    torch  PyTorch FP16 on CUDA (reference)
    trt    TensorRT engine (FP16 or INT8) on an NVIDIA GPU, built for this exact GPU
    dml    ONNX Runtime + DirectML: any DirectX 12 GPU on Windows (NVIDIA, AMD, Intel integrated) or NPU
Engines are built for fixed sizes; a 360p engine also serves 720p / 1080p streams through low-res repair.
TensorRT 11 builds strongly typed networks, so precision lives in the ONNX file: an FP16 export for FP16, and
explicit INT8 quantize/dequantize nodes (calibrated on real frames) for INT8.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

INPUTS = ("prev2", "prev1", "partial", "mask")
OUTPUT = "repaired"


def onnx_path(h: int, w: int, root: str = "weights", variant: str = "fp32") -> Path:
    """variant: fp32 | fp16 | int8 (FP32 graph with quantize/dequantize nodes around the convolutions) |
    int8fp16 (the same INT8 convolutions, FP16 everywhere else: the fast one)."""
    return Path(root) / (f"repairnet_{h}x{w}.onnx" if variant == "fp32" else f"repairnet_{h}x{w}_{variant}.onnx")


def engine_path(h: int, w: int, precision: str, root: str = "weights", portable: bool = False) -> Path:
    """GPU-specific engine, or a portable one (hardware-compatible with every Ampere-or-newer NVIDIA GPU,
    so an engine built on one machine runs on another with the same TensorRT version)."""
    if portable:
        return Path(root) / f"repairnet_{h}x{w}_{precision}_ampere_plus.engine"
    import torch
    gpu = torch.cuda.get_device_name(0).replace(" ", "_") if torch.cuda.is_available() else "cpu"
    return Path(root) / f"repairnet_{h}x{w}_{precision}_{gpu}.engine"


def export_onnx(weights: str, h: int, w: int, out: Path | None = None, half: bool = False) -> Path:
    """Static-shape ONNX of RepairNet (inputs RGB in [0,1], NCHW; mask 1 = lost). half: FP16 graph and I/O."""
    import torch
    from .models.repairnet import RepairNet

    ck = torch.load(weights, map_location="cpu")
    net = RepairNet(base=ck.get("base", 32))
    net.load_state_dict(ck["model"])
    net.eval()
    dev, dt = ("cuda", torch.float16) if half else ("cpu", torch.float32)
    net = net.to(dev, dt)
    out = out or onnx_path(h, w, variant="fp16" if half else "fp32")
    ins = tuple(torch.rand(1, 1 if n == "mask" else 3, h, w, device=dev, dtype=dt) for n in INPUTS)
    torch.onnx.export(net, ins, str(out), input_names=list(INPUTS), output_names=[OUTPUT], opset_version=17,
                      dynamo=False)
    return out


def quantize_int8(fp32_onnx: Path, out: Path, calib_batches) -> Path:
    """Explicit INT8 quantization (what TensorRT 10+/11 expects): calibrate activation ranges on real frames
    and insert symmetric quantize/dequantize nodes around every convolution; weights per output channel."""
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)

    batches = list(calib_batches)

    class Reader(CalibrationDataReader):
        """Chunked calibration (CalibStridedMinMax) asks for the length and reads one range at a time."""

        def __init__(self):
            self.it = iter(batches)

        def __len__(self):
            return len(batches)

        def set_range(self, start_index: int, end_index: int):
            self.it = iter(batches[start_index:end_index])

        def get_next(self):
            b = next(self.it, None)
            return None if b is None else {k: b[k] for k in INPUTS}

    # Min-max ranges, computed in chunks of 4 samples: entropy calibration keeps every activation of every
    # sample in memory at once (tens of GB at 360p) and was killed for running out of RAM.
    quantize_static(str(fp32_onnx), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=["Conv", "ConvTranspose"], calibrate_method=CalibrationMethod.MinMax,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "CalibStridedMinMax": 4,
                                   "QuantizeBias": False})   # TensorRT rejects INT32 bias Q/DQ; it fuses FP biases
    return out


def int8_to_fp16(qdq_onnx: Path, out: Path) -> Path:
    """INT8 convolutions with FP16 everywhere else. Quantized from the FP32 model, every operation between the
    INT8 convolutions would run in FP32 under TensorRT's strong typing and cost back what INT8 saves; in FP16
    the INT8 engine is ~25 % faster than the FP16 one (RTX 5060 Ti) for ~0.2 dB."""
    import onnx
    from onnxconverter_common import float16

    m = float16.convert_float_to_float16(onnx.load(str(qdq_onnx)), keep_io_types=False, op_block_list=[])
    onnx.save(m, str(out))
    return out


_TORCH_DT = {"FLOAT": "float32", "HALF": "float16"}


class TRTRunner:
    """Runs a serialized TensorRT engine on torch CUDA tensors (no extra CUDA Python packages needed)."""

    def __init__(self, path: str | Path):
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.ERROR)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"{Path(path).name} cannot run here (different TensorRT version, GPU or platform)")
        self.ctx = self.engine.create_execution_context()
        dt = lambda n: getattr(torch, _TORCH_DT[self.engine.get_tensor_dtype(n).name])
        self.in_dt = {n: dt(n) for n in INPUTS}
        shape = self.engine.get_tensor_shape(OUTPUT)
        self.h, self.w = int(shape[2]), int(shape[3])
        self.out = torch.empty(tuple(shape), dtype=dt(OUTPUT), device="cuda")
        self.ctx.set_tensor_address(OUTPUT, self.out.data_ptr())

    def __call__(self, p2, p1, cur, m):
        """Inputs: (1,C,H,W) CUDA tensors; returns the (1,3,H,W) output tensor (a reused buffer)."""
        torch = self.torch
        keep = []
        for name, t in zip(INPUTS, (p2, p1, cur, m)):
            t = t.to(self.in_dt[name]).contiguous()
            keep.append(t)
            self.ctx.set_tensor_address(name, t.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out


def build_engine(onnx_file: Path, out: Path, opt_level: int = 2, portable: bool = False) -> Path:
    """TensorRT 11 networks are strongly typed: precision comes from the ONNX (FP16 graph, or Q/DQ for INT8).
    opt_level 0-5: how many kernel variants the builder times; lower is gentler on a laptop GPU (the default 3
    can trigger Windows GPU-timeout resets on some laptops)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_file.read_bytes()):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
    config.builder_optimization_level = opt_level
    if portable:
        config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    out.write_bytes(bytes(blob))
    return out


class DMLRunner:
    """ONNX Runtime with the DirectML provider: any DirectX 12 GPU / NPU on Windows."""

    def __init__(self, onnx_file: str | Path, device_id: int = 0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = False                  # required by the DirectML provider
        self.sess = ort.InferenceSession(str(onnx_file), so, providers=[("DmlExecutionProvider",
                                                                         {"device_id": device_id})])
        if self.sess.get_providers()[0] != "DmlExecutionProvider":   # ORT silently falls back to the CPU
            raise RuntimeError(f"DirectML device {device_id} is not available")
        i0 = self.sess.get_inputs()[0]
        self.h, self.w = int(i0.shape[2]), int(i0.shape[3])
        self.dt = np.float16 if i0.type == "tensor(float16)" else np.float32

    def __call__(self, p2, p1, cur, m) -> np.ndarray:
        """Inputs: (1,C,H,W) numpy arrays; returns (1,3,H,W) float32."""
        feed = {k: v.astype(self.dt, copy=False) for k, v in zip(INPUTS, (p2, p1, cur, m))}
        return self.sess.run([OUTPUT], feed)[0].astype(np.float32, copy=False)
