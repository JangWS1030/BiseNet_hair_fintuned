from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def binary_boundary(mask: np.ndarray, tolerance: int = 2) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (dilated - eroded) > 0


def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    hair_label: int = 10,
    face_label: int = 1,
    ignore_index: int = 255,
    boundary_tolerance: int = 2,
) -> dict[str, float]:
    valid = gt_mask != ignore_index
    pred = pred_mask[valid]
    gt = gt_mask[valid]

    pred_hair = pred == hair_label
    gt_hair = gt == hair_label
    pred_face = pred == face_label
    gt_face = gt == face_label

    hair_intersection = np.logical_and(pred_hair, gt_hair).sum()
    hair_union = np.logical_or(pred_hair, gt_hair).sum()
    hair_denom = pred_hair.sum() + gt_hair.sum()
    hair_iou = _safe_ratio(hair_intersection, hair_union)
    hair_dice = _safe_ratio(2.0 * hair_intersection, hair_denom)

    face_spill = _safe_ratio(np.logical_and(pred_hair, gt_face).sum(), gt_face.sum())

    pred_boundary = binary_boundary(pred_hair.astype(np.uint8), tolerance=boundary_tolerance)
    gt_boundary = binary_boundary(gt_hair.astype(np.uint8), tolerance=boundary_tolerance)
    boundary_tp = np.logical_and(pred_boundary, gt_boundary).sum()
    boundary_precision = _safe_ratio(boundary_tp, pred_boundary.sum())
    boundary_recall = _safe_ratio(boundary_tp, gt_boundary.sum())
    boundary_f1 = _safe_ratio(2 * boundary_precision * boundary_recall, boundary_precision + boundary_recall)

    face_iou = _safe_ratio(np.logical_and(pred_face, gt_face).sum(), np.logical_or(pred_face, gt_face).sum())

    return {
        "hair_iou": hair_iou,
        "hair_dice": hair_dice,
        "boundary_f1": boundary_f1,
        "face_spill_rate": face_spill,
        "face_iou": face_iou,
    }


def aggregate_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    if not results:
        return {}
    totals = defaultdict(float)
    for row in results:
        for key, value in row.items():
            totals[key] += float(value)
    return {key: value / len(results) for key, value in totals.items()}


def save_prediction_mask(mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8))


def tensor_to_label_map(logits: torch.Tensor, output_size: tuple[int, int] | None = None) -> np.ndarray:
    if output_size is not None:
        logits = torch.nn.functional.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )
    return torch.argmax(logits, dim=1).detach().cpu().numpy()


def subset_metric_rows(rows: list[dict[str, Any]], subset_columns: list[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for column in subset_columns:
        subset_rows = [row["metrics"] for row in rows if str(row.get(column, "")).lower() in {"1", "true", "yes"}]
        if subset_rows:
            grouped[column] = subset_rows
    return {column: aggregate_metrics(values) for column, values in grouped.items()}

