from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch


def _extract_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("state_dict", "model_state_dict", "model", "net"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    if all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload  # raw state_dict
    raise ValueError("Checkpoint does not contain a recognizable model state dict")


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path | None,
    strict: bool = True,
    allow_partial: bool = True,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"loaded": False, "mode": "random_init", "missing_keys": [], "unexpected_keys": []}

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(payload if isinstance(payload, dict) else {"state_dict": payload})

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        mode = "strict" if strict else "partial"
    except RuntimeError:
        if not allow_partial:
            raise
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        mode = "partial"

    return {
        "loaded": True,
        "path": str(path),
        "mode": mode,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "config": config,
    }
    torch.save(payload, path)


def load_training_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    strict: bool = False,
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(payload)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return {
        "path": str(path),
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "metrics": payload.get("metrics", {}),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def export_state_dict(src_checkpoint: str | Path, dst_path: str | Path, key_name: str = "model_state_dict") -> None:
    src_checkpoint = Path(src_checkpoint)
    dst_path = Path(dst_path)
    payload = torch.load(src_checkpoint, map_location="cpu")
    state_dict = payload.get(key_name)
    if state_dict is None:
        state_dict = _extract_state_dict(payload)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, dst_path)


def copy_best_alias(src_checkpoint: str | Path, run_dir: str | Path, alias_name: str = "seg_sd_ft.pth") -> Path:
    run_dir = Path(run_dir)
    dst_path = run_dir / alias_name
    export_state_dict(src_checkpoint, dst_path)
    return dst_path


def mirror_file(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
