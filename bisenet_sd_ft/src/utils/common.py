from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_link_or_copy(src: str | Path, dst: str | Path, mode: str = "hardlink") -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    ensure_dir(dst_path.parent)
    if dst_path.exists():
        return
    if mode == "none":
        return
    if mode == "copy":
        shutil.copy2(src_path, dst_path)
        return
    if mode == "symlink":
        try:
            dst_path.symlink_to(src_path)
            return
        except OSError:
            shutil.copy2(src_path, dst_path)
            return
    if mode == "hardlink":
        try:
            os.link(src_path, dst_path)
            return
        except OSError:
            shutil.copy2(src_path, dst_path)
            return
    raise ValueError(f"Unsupported copy mode: {mode}")


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)

