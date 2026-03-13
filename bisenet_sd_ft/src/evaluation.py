from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.metrics import aggregate_metrics, compute_segmentation_metrics, save_prediction_mask, subset_metric_rows


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_size: int,
    coarse_labels: dict[str, int],
    eval_cfg: dict[str, Any],
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    model.eval()
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict[str, Any]] = []
    metrics_buffer: list[dict[str, float]] = []
    saved = 0
    limit = int(eval_cfg.get("prediction_limit", 16))
    tolerance = int(eval_cfg.get("boundary_tolerance", 2))

    for batch in tqdm(dataloader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        logits, _, _ = model(images)
        logits = F.interpolate(logits, size=(output_size, output_size), mode="bilinear", align_corners=True)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        meta_batch = batch["meta"]

        for index in range(preds.shape[0]):
            meta = {key: value[index] if isinstance(value, list) else value[index] for key, value in meta_batch.items()}
            row_metrics = compute_segmentation_metrics(
                pred_mask=preds[index],
                gt_mask=targets_np[index],
                hair_label=int(coarse_labels["hair"]),
                face_label=int(coarse_labels["face"]),
                ignore_index=int(coarse_labels["ignore"]),
                boundary_tolerance=tolerance,
            )
            sample_rows.append({**meta, "metrics": row_metrics})
            metrics_buffer.append(row_metrics)
            if save_dir is not None and bool(eval_cfg.get("save_predictions", False)) and saved < limit:
                save_prediction_mask(preds[index], save_dir / f"{meta['sample_id']}.png")
                saved += 1

    return {
        "overall": aggregate_metrics(metrics_buffer),
        "subsets": subset_metric_rows(sample_rows, list(eval_cfg.get("subset_columns", []))),
        "num_samples": len(sample_rows),
    }

