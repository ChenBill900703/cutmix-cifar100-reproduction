"""Controlled CUDA memory preflight for training and validation batch sizes."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import PyramidNet, count_parameters
from .runtime import configure_cuda_memory_limit, environment_info, write_json


def _cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)


def training_trial(batch_size: int, steps: int, limit_bytes: int) -> dict[str, Any]:
    _cleanup()
    started = time.perf_counter()
    try:
        model = PyramidNet(200, 240, 100, True).to("cuda:0")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25, momentum=0.9, weight_decay=1e-4, nesterov=True)
        criterion = nn.CrossEntropyLoss().to("cuda:0")
        scaler = torch.amp.GradScaler("cuda")
        for _ in range(steps):
            inputs = torch.randn(batch_size, 3, 32, 32, device="cuda:0")
            targets = torch.randint(0, 100, (batch_size,), device="cuda:0")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize(0)
        peak_allocated = torch.cuda.max_memory_allocated(0)
        peak_reserved = torch.cuda.max_memory_reserved(0)
        return {
            "batch_size": batch_size,
            "status": "safe" if peak_reserved < limit_bytes else "over_limit",
            "steps": steps,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "seconds": time.perf_counter() - started,
        }
    except torch.OutOfMemoryError as exc:
        return {"batch_size": batch_size, "status": "oom", "steps": steps, "error": str(exc), "seconds": time.perf_counter() - started}
    finally:
        for name in ("model", "optimizer", "criterion", "scaler", "inputs", "targets", "loss"):
            if name in locals():
                del locals()[name]
        _cleanup()


def validation_trial(batch_size: int, steps: int, limit_bytes: int) -> dict[str, Any]:
    _cleanup()
    started = time.perf_counter()
    try:
        model = PyramidNet(200, 240, 100, True).to("cuda:0").eval()
        with torch.inference_mode():
            for _ in range(steps):
                inputs = torch.randn(batch_size, 3, 32, 32, device="cuda:0")
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    model(inputs)
        torch.cuda.synchronize(0)
        peak_allocated = torch.cuda.max_memory_allocated(0)
        peak_reserved = torch.cuda.max_memory_reserved(0)
        return {
            "batch_size": batch_size,
            "status": "safe" if peak_reserved < limit_bytes else "over_limit",
            "steps": steps,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "seconds": time.perf_counter() - started,
        }
    except torch.OutOfMemoryError as exc:
        return {"batch_size": batch_size, "status": "oom", "steps": steps, "error": str(exc), "seconds": time.perf_counter() - started}
    finally:
        if "model" in locals():
            del model
        _cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/preflight/report.json")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 5:
        raise ValueError("preflight requires at least five complete steps")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible CUDA GPU, found {torch.cuda.device_count()}")
    limit = configure_cuda_memory_limit(0)
    model_probe = PyramidNet(200, 240, 100, True)
    parameters = count_parameters(model_probe)
    del model_probe
    training = [training_trial(batch, args.steps, int(limit["limit_bytes"])) for batch in (64, 32, 16, 8)]
    validation = [validation_trial(batch, args.steps, int(limit["limit_bytes"])) for batch in (64, 32, 16, 8)]
    safe_training = [trial["batch_size"] for trial in training if trial["status"] == "safe"]
    safe_validation = [trial["batch_size"] for trial in validation if trial["status"] == "safe"]
    report = {
        "environment": environment_info(),
        "memory_limit": limit,
        "model": {"name": "PyramidNet-200", "alpha": 240, "bottleneck": True, "parameter_count": parameters},
        "training_trials": training,
        "validation_trials": validation,
        "recommended_micro_batch": max(safe_training) if safe_training else None,
        "recommended_validation_batch": max(safe_validation) if safe_validation else None,
        "fallback_required": not safe_training,
        "note": "If fallback_required is true, inspect other GPU processes before using PyramidNet-110 alpha=64.",
    }
    write_json(Path(args.output).resolve(), report)
    print(Path(args.output).resolve())
    print(report)
    if not safe_training:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

