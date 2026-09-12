"""TOML configuration loading and validation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    config["_config_path"] = str(path)
    required = ["experiment", "data", "model", "training", "cutmix", "memory"]
    missing = [section for section in required if section not in config]
    if missing:
        raise ValueError(f"missing config sections: {missing}")
    training = config["training"]
    if int(training["effective_batch_size"]) != 64:
        raise ValueError("paper reproduction requires effective_batch_size=64")
    if config["data"].get("dataset") != "cifar100":
        raise ValueError("this scoped reproduction supports CIFAR-100 only")
    if bool(config["model"].get("bottleneck")) is not True:
        raise ValueError("paper reproduction requires bottleneck=true")
    return config


def resolve_from_project(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()

