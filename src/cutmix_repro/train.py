"""Single-GPU CIFAR-100 trainer with AMP, accumulation, and exact resume."""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import load_config, resolve_from_project
from .cutmix import apply_cutmix
from .data import make_loaders
from .model import PyramidNet, count_parameters
from .runtime import (
    append_csv,
    capture_rng_state,
    configure_cuda_memory_limit,
    environment_info,
    make_numpy_rng,
    restore_rng_state,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-report", default="outputs/preflight/report.json")
    parser.add_argument("--resume", default=None, help="checkpoint path or 'auto'")
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="1-based epoch used for resume testing")
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def _batch_sizes(config: dict[str, Any], preflight_path: Path) -> tuple[int, int]:
    micro = config["training"]["micro_batch_size"]
    val = config["training"]["validation_batch_size"]
    if micro == "auto" or val == "auto":
        if not preflight_path.exists():
            raise FileNotFoundError(f"auto batch size requires {preflight_path}; run preflight first")
        import json

        report = json.loads(preflight_path.read_text(encoding="utf-8"))
        micro = report["recommended_micro_batch"] if micro == "auto" else micro
        val = report["recommended_validation_batch"] if val == "auto" else val
    return int(micro), int(val)


def _lr_for_epoch(initial_lr: float, epoch_index: int, milestones: list[int]) -> float:
    return initial_lr * (0.1 ** sum(epoch_index >= milestone for milestone in milestones))


def _error_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    top5 = logits.topk(5, dim=1).indices
    correct = top5.eq(targets[:, None])
    return int((~correct[:, 0]).sum().item()), int((~correct.any(dim=1)).sum().item())


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    effective_batch_size: int,
    use_cutmix: bool,
    beta: float,
    probability: float,
    cutmix_rng: np.random.Generator,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    dataset_samples = len(loader.dataset)
    group_samples = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        at_group_start = group_samples == 0
        if at_group_start:
            group_target = min(effective_batch_size, dataset_samples - total_samples)
        batch_samples = targets.shape[0]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            if use_cutmix and cutmix_rng.random() < probability:
                mixed = apply_cutmix(inputs, targets, beta, rng=cutmix_rng)
                logits = model(mixed.inputs)
                raw_loss = mixed.lam * criterion(logits, mixed.target_a) + (1.0 - mixed.lam) * criterion(logits, mixed.target_b)
            else:
                logits = model(inputs)
                raw_loss = criterion(logits, targets)
            # Sample-weighted scaling also handles a final incomplete accumulation group.
            backward_loss = raw_loss * (batch_samples / group_target)
        scaler.scale(backward_loss).backward()
        group_samples += batch_samples
        total_samples += batch_samples
        total_loss += float(raw_loss.detach()) * batch_samples
        last_batch = batch_index + 1 == len(loader)
        if group_samples >= group_target or last_batch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
    return total_loss / total_samples


@torch.inference_mode()
def validate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    top1_errors = 0
    top5_errors = 0
    samples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(inputs)
            loss = criterion(logits, targets)
        err1, err5 = _error_counts(logits, targets)
        batch = targets.shape[0]
        samples += batch
        total_loss += float(loss) * batch
        top1_errors += err1
        top5_errors += err5
    return total_loss / samples, 100.0 * top1_errors / samples, 100.0 * top5_errors / samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best: dict[str, Any],
    cutmix_rng: np.random.Generator,
    config: dict[str, Any],
) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best": best,
        "rng": capture_rng_state(cutmix_rng),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    temp.replace(path)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(args.config)
    output = resolve_from_project(args.output, project_root)
    output.mkdir(parents=True, exist_ok=True)
    preflight_path = resolve_from_project(args.preflight_report, project_root)
    micro_batch, val_batch = _batch_sizes(config, preflight_path)
    effective_batch = int(config["training"]["effective_batch_size"])
    if effective_batch % micro_batch != 0:
        raise ValueError("effective batch size must be divisible by micro batch size")
    accumulation_steps = effective_batch // micro_batch
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device cuda:0 is required")
    device = torch.device("cuda:0")
    limit = configure_cuda_memory_limit(0)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    cutmix_rng = make_numpy_rng(seed + 10_000)
    torch.backends.cudnn.benchmark = bool(config["training"].get("cudnn_benchmark", True))
    model = PyramidNet(
        depth=int(config["model"]["depth"]),
        alpha=float(config["model"]["alpha"]),
        num_classes=100,
        bottleneck=bool(config["model"]["bottleneck"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        momentum=float(config["training"]["momentum"]),
        weight_decay=float(config["training"]["weight_decay"]),
        nesterov=bool(config["training"]["nesterov"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    data_root = resolve_from_project(config["data"]["root"], project_root)
    train_loader, val_loader = make_loaders(
        str(data_root), micro_batch, val_batch, int(config["data"]["workers"]),
        download=args.download, train_samples=args.train_samples, val_samples=args.val_samples,
    )
    start_epoch = 0
    best: dict[str, Any] = {"top1_error": 100.0, "top5_error": 100.0, "epoch": 0}
    resume_path: Path | None = None
    if args.resume:
        resume_path = output / "last.pt" if args.resume == "auto" else resolve_from_project(args.resume, project_root)
        # Keep CPU/Python RNG tensors on CPU. load_state_dict moves model and
        # optimizer tensors to their parameter devices as needed.
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        best = checkpoint["best"]
        start_epoch = int(checkpoint["epoch"])
        restore_rng_state(checkpoint["rng"], cutmix_rng)
    env = environment_info()
    metadata = {
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "environment": env,
        "parameter_count": count_parameters(model),
        "device": "cuda:0",
        "memory_limit": limit,
        "micro_batch_size": micro_batch,
        "effective_batch_size": effective_batch,
        "accumulation_steps": accumulation_steps,
        "validation_batch_size": val_batch,
        "resumed_from": str(resume_path) if resume_path else None,
    }
    write_json(output / "metadata.json", metadata)
    epochs = int(config["training"]["epochs"])
    milestones = [int(value) for value in config["training"]["lr_milestones"]]
    last_row: dict[str, Any] | None = None
    max_epoch = min(epochs, args.stop_after_epoch) if args.stop_after_epoch else epochs
    for epoch_index in range(start_epoch, max_epoch):
        lr = _lr_for_epoch(float(config["training"]["learning_rate"]), epoch_index, milestones)
        for group in optimizer.param_groups:
            group["lr"] = lr
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, effective_batch,
            bool(config["cutmix"]["enabled"]), float(config["cutmix"]["beta"]),
            float(config["cutmix"]["probability"]), cutmix_rng,
        )
        val_loss, top1_error, top5_error = validate(model, val_loader, criterion, device)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        if reserved >= int(limit["limit_bytes"]):
            raise RuntimeError(f"peak reserved VRAM {reserved} reached safety limit {limit['limit_bytes']}")
        epoch_number = epoch_index + 1
        row = {
            "epoch": epoch_number,
            "lr": lr,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "top1_error": top1_error,
            "top5_error": top5_error,
            "epoch_seconds": elapsed,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
        }
        append_csv(output / "metrics.csv", row)
        is_best = top1_error < float(best["top1_error"])
        if is_best:
            best = {"top1_error": top1_error, "top5_error": top5_error, "epoch": epoch_number}
        save_checkpoint(output / "last.pt", model, optimizer, scaler, epoch_number, best, cutmix_rng, config)
        if is_best:
            shutil.copy2(output / "last.pt", output / "best.pt")
        last_row = row
        summary = {"status": "running" if epoch_number < epochs else "complete", "best": best, "last": row, **metadata}
        write_json(output / "summary.json", summary)
        print(
            f"epoch={epoch_number}/{epochs} lr={lr:.5g} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} top1_err={top1_error:.3f} top5_err={top5_error:.3f} "
            f"seconds={elapsed:.1f} peak_reserved_gib={reserved / 1024**3:.3f}",
            flush=True,
        )
    if last_row is None and (output / "summary.json").exists():
        return


if __name__ == "__main__":
    main()
