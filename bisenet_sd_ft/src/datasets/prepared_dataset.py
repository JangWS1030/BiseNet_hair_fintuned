from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.common import IMAGENET_MEAN, IMAGENET_STD, worker_init_fn


def _to_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return "true"
    if text in {"0", "false", "no", "n"}:
        return "false"
    return text


def load_manifest(manifest_path: str | Path, split_path: str | Path | None = None) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    if split_path is not None:
        split_path = Path(split_path)
        sample_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = df[df["sample_id"].astype(str).isin(sample_ids)].copy()
    return df.reset_index(drop=True)


def build_joint_transform(output_size: int, augment_cfg: dict[str, Any], train: bool = True) -> A.Compose:
    transforms: list[Any] = [A.Resize(output_size, output_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)]
    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=float(augment_cfg.get("horizontal_flip", 0.5))),
                A.ShiftScaleRotate(
                    shift_limit=float(augment_cfg.get("shift_limit", 0.04)),
                    scale_limit=float(augment_cfg.get("scale_limit", 0.08)),
                    rotate_limit=float(augment_cfg.get("rotate_limit", 8)),
                    border_mode=cv2.BORDER_REFLECT_101,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=0.7,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=float(augment_cfg.get("brightness_contrast", 0.15)),
                    contrast_limit=float(augment_cfg.get("brightness_contrast", 0.15)),
                    p=0.35,
                ),
                A.RandomGamma(gamma_limit=tuple(augment_cfg.get("gamma_limit", [92, 108])), p=0.2),
                A.ColorJitter(
                    brightness=float(augment_cfg.get("color_jitter", 0.08)),
                    contrast=float(augment_cfg.get("color_jitter", 0.08)),
                    saturation=float(augment_cfg.get("color_jitter", 0.08)),
                    hue=0.02,
                    p=0.25,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    ],
                    p=float(augment_cfg.get("blur_p", 0.08)),
                ),
                A.GaussNoise(var_limit=(5.0, 20.0), p=float(augment_cfg.get("noise_p", 0.08))),
                A.ImageCompression(quality_lower=80, quality_upper=100, p=float(augment_cfg.get("jpeg_p", 0.08))),
            ]
        )
    return A.Compose(transforms)


class PreparedHairSegDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split_path: str | Path | None,
        input_size: int,
        output_size: int,
        augment_cfg: dict[str, Any],
        train: bool = True,
        ignore_index: int = 255,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = load_manifest(self.manifest_path, split_path).to_dict("records")
        self.input_size = input_size
        self.output_size = output_size
        self.ignore_index = ignore_index
        self.transform = build_joint_transform(output_size=output_size, augment_cfg=augment_cfg, train=train)
        self.root_dir = self.manifest_path.parent

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, stored_path: str) -> Path:
        path = Path(stored_path)
        if path.is_absolute():
            return path
        return (self.root_dir / path).resolve()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        image_path = self._resolve_path(row["image_path"])
        mask_path = self._resolve_path(row["mask_path"])

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        transformed = self.transform(image=image, mask=mask)
        image_out = transformed["image"]
        mask_out = transformed["mask"].astype(np.uint8)
        input_image = cv2.resize(image_out, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        input_tensor = torch.from_numpy(input_image.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        input_tensor = (input_tensor - mean) / std
        target_tensor = torch.from_numpy(mask_out.astype(np.int64))

        meta = {
            "sample_id": str(row["sample_id"]),
            "quality": str(row.get("quality", "")),
            "source_split": str(row.get("source_split", "")),
        }
        for column in ("short_hair", "bangs", "sideburn", "dark_hair", "group_id", "subject_id", "sequence_id"):
            if column in row:
                meta[column] = _to_bool(row[column]) if column in {"short_hair", "bangs", "sideburn", "dark_hair"} else str(row[column])

        return {"image": input_tensor, "target": target_tensor, "meta": meta}


def build_dataloader(
    manifest_path: str | Path,
    split_path: str | Path | None,
    input_size: int,
    output_size: int,
    augment_cfg: dict[str, Any],
    batch_size: int,
    num_workers: int,
    train: bool,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
    ignore_index: int = 255,
) -> DataLoader:
    dataset = PreparedHairSegDataset(
        manifest_path=manifest_path,
        split_path=split_path,
        input_size=input_size,
        output_size=output_size,
        augment_cfg=augment_cfg,
        train=train,
        ignore_index=ignore_index,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=drop_last and train,
        worker_init_fn=worker_init_fn,
    )
