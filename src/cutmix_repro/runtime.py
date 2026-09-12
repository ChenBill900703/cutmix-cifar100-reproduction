"""Runtime, reproducibility, checkpoint, and metrics helpers."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

GIB = 1024**3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_numpy_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def memory_limit_bytes(total_bytes: int) -> int:
    return min(int(7.2 * GIB), int(total_bytes * 0.90))


def configure_cuda_memory_limit(device_index: int = 0) -> dict[str, float | int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal reproduction")
    total = torch.cuda.get_device_properties(device_index).total_memory
    limit = memory_limit_bytes(total)
    torch.cuda.set_per_process_memory_fraction(limit / total, device=device_index)
    return {"total_bytes": total, "limit_bytes": limit, "fraction": limit / total}


def environment_info() -> dict[str, Any]:
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        driver = "unavailable"
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except ImportError:
        torchvision_version = "unavailable"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": driver,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
    }


def capture_rng_state(cutmix_rng: np.random.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "numpy_cutmix": cutmix_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any], cutmix_rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_legacy"])
    cutmix_rng.bit_generator.state = state["numpy_cutmix"]
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

