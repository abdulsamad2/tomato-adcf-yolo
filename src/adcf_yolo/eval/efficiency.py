"""Params, GFLOPs and latency/FPS for trained checkpoints.

LEARN (fair speed measurement, see docs/notes/05-metrics.md):
  * Measure every model in ONE session on the SAME idle GPU; never mix numbers
    measured on different days/machines.
  * Warm up first (cuDNN autotuning, memory allocation), then time many iterations.
  * GPU calls are asynchronous: synchronize before reading the clock.
  * Batch 1 is the deployment-relevant setting; report precision (FP16/FP32).
  * This times the network only. NMS cost is reported separately by Ultralytics
    val ("postprocess" ms); YOLO26 is NMS-free, so compare end-to-end too.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def model_complexity(weights: str | Path, imgsz: int = 640) -> dict[str, float]:
    model = copy.deepcopy(YOLO(str(weights)).model).float().cpu().eval().fuse(verbose=False)
    return {"params_M": get_num_params(model) / 1e6, "GFLOPs": float(get_flops(model, imgsz))}


@torch.inference_mode()
def measure_latency(
    weights: str | Path,
    imgsz: int = 640,
    batch: int = 1,
    half: bool = True,
    warmup: int = 50,
    iters: int = 500,
    device: str | None = None,
) -> dict[str, float | str | bool]:
    dev = pick_device(device)
    half = half and dev.type == "cuda"  # FP16 timing is only meaningful on CUDA
    model = copy.deepcopy(YOLO(str(weights)).model).float().eval().fuse(verbose=False).to(dev)
    if half:
        model.half()
    x = torch.rand(batch, 3, imgsz, imgsz, device=dev, dtype=torch.float16 if half else torch.float32)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(iters):
        _sync(dev)
        t0 = time.perf_counter()
        model(x)
        _sync(dev)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type,
        "half": half,
        "batch": batch,
        "imgsz": imgsz,
        "latency_ms_mean": float(t.mean()),
        "latency_ms_std": float(t.std()),
        "latency_ms_p50": float(np.percentile(t, 50)),
        "latency_ms_p95": float(np.percentile(t, 95)),
        "fps": float(1000 * batch / t.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("runs"), help="experiment root written by adcf-run")
    p.add_argument("--imgsz", type=int, help="default: each run's training imgsz (from its args.yaml)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--fp32", action="store_true", help="time in FP32 instead of FP16")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--device")
    p.add_argument("--seed-dir", default="seed0", help="which seed's checkpoint to time (speed is seed-independent)")
    args = p.parse_args()

    checkpoints = sorted(args.runs.glob(f"*/{args.seed_dir}/weights/best.pt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints under {args.runs}/*/{args.seed_dir}/weights/best.pt")
    for ckpt in checkpoints:
        run_dir = ckpt.parents[2]
        train_args = ckpt.parents[1] / "args.yaml"
        imgsz = args.imgsz or (yaml.safe_load(train_args.read_text())["imgsz"] if train_args.exists() else 640)
        result = {
            **model_complexity(ckpt, imgsz),
            **measure_latency(ckpt, imgsz, args.batch, not args.fp32, args.warmup, args.iters, args.device),
        }
        (run_dir / "speed.json").write_text(json.dumps(result, indent=2))
        print(f"{run_dir.name:32s} {result['params_M']:.2f}M  {result['GFLOPs']:.1f} GFLOPs  "
              f"{result['latency_ms_mean']:.2f}±{result['latency_ms_std']:.2f} ms  {result['fps']:.0f} FPS")


if __name__ == "__main__":
    main()
