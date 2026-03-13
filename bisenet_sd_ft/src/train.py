from __future__ import annotations

import argparse
import csv
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.common import ensure_dir, save_json, set_seed
from src.utils.config import load_config, resolve_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BiSeNet for SD hair segmentation.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def create_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int, min_lr: float) -> LambdaLR:
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1.0e-8, float(step + 1) / max(1, warmup_steps))
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / max(base_lrs), cosine)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def set_backbone_trainable(model: BiSeNet, trainable: bool) -> None:
    for parameter in model.cp.resnet.parameters():
        parameter.requires_grad = trainable


def make_run_dir(cfg: dict[str, Any]) -> Path:
    output_root = resolve_path(cfg["experiment"]["output_dir"], cfg)
    assert output_root is not None
    return ensure_dir(output_root / cfg["experiment"]["name"])


def build_optimizer(model: BiSeNet, cfg: dict[str, Any]) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        betas=tuple(cfg["optimizer"].get("betas", [0.9, 0.999])),
    )


def append_history(history_path: Path, row: dict[str, Any]) -> None:
    file_exists = history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    from src.datasets import build_dataloader
    from src.evaluation import run_evaluation
    from src.losses import HairSegLoss
    from src.models import BiSeNet
    from src.utils.checkpoint import (
        copy_best_alias,
        load_model_checkpoint,
        load_training_checkpoint,
        save_training_checkpoint,
    )

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_dir = make_run_dir(cfg)
    save_json(run_dir / "resolved_config.json", cfg)

    manifest_path = resolve_path(cfg["data"]["manifest_path"], cfg)
    splits_dir = resolve_path(cfg["data"]["splits_dir"], cfg)
    if manifest_path is None or splits_dir is None:
        raise ValueError("manifest_path and splits_dir are required")

    train_loader = build_dataloader(
        manifest_path=manifest_path,
        split_path=splits_dir / cfg["data"]["train_split"],
        input_size=int(cfg["model"]["input_size"]),
        output_size=int(cfg["model"]["output_size"]),
        augment_cfg=cfg["data"]["augment"],
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        train=True,
        pin_memory=bool(cfg["data"]["pin_memory"]),
        persistent_workers=bool(cfg["data"]["persistent_workers"]),
        drop_last=bool(cfg["data"]["drop_last"]),
        ignore_index=int(cfg["data"]["coarse_labels"]["ignore"]),
    )
    val_loader = build_dataloader(
        manifest_path=manifest_path,
        split_path=splits_dir / cfg["data"]["val_split"],
        input_size=int(cfg["model"]["input_size"]),
        output_size=int(cfg["model"]["output_size"]),
        augment_cfg=cfg["data"]["augment"],
        batch_size=int(cfg["data"]["val_batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        train=False,
        pin_memory=bool(cfg["data"]["pin_memory"]),
        persistent_workers=bool(cfg["data"]["persistent_workers"]),
        drop_last=False,
        ignore_index=int(cfg["data"]["coarse_labels"]["ignore"]),
    )

    model = BiSeNet(
        n_classes=int(cfg["model"]["num_classes"]),
        pretrained_backbone=bool(cfg["model"]["pretrained_backbone"]),
    )
    model = model.to(device)
    steps_per_epoch = math.ceil(len(train_loader) / int(cfg["train"]["grad_accum_steps"]))
    total_steps = max(1, int(cfg["train"]["epochs"]) * steps_per_epoch)
    warmup_steps = max(1, int(cfg["scheduler"]["warmup_epochs"]) * steps_per_epoch)
    mp_mode = str(cfg["train"].get("mixed_precision", "bf16")).lower()
    use_amp = bool(cfg["train"].get("amp_enabled", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if mp_mode == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    criterion = HairSegLoss(cfg=cfg, num_classes=int(cfg["model"]["num_classes"])).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=float(cfg["scheduler"]["min_lr"]),
    )

    resume_path = resolve_path(cfg["model"].get("resume_checkpoint"), cfg)
    start_epoch = 1
    global_step = 0
    if resume_path is not None:
        resume_info = load_training_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if scaler.is_enabled() else None,
            strict=False,
        )
        load_info = {"loaded": True, "mode": "resume", **resume_info}
        start_epoch = int(resume_info["epoch"]) + 1
        global_step = int(resume_info["global_step"])
    else:
        load_info = load_model_checkpoint(
            model=model,
            checkpoint_path=resolve_path(cfg["model"].get("init_checkpoint"), cfg),
            strict=bool(cfg["model"]["strict_load"]),
            allow_partial=bool(cfg["model"]["allow_partial_load"]),
        )
    if bool(cfg["model"].get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if bool(cfg["model"].get("channels_last", False)) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    save_json(run_dir / "checkpoint_load.json", load_info)

    best_metric = float(load_info.get("metrics", {}).get("overall", {}).get("hair_dice", -1.0))
    history_path = run_dir / "history.csv"
    latest_step_interval = int(cfg["train"].get("save_every_steps", 0))

    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        set_backbone_trainable(model, trainable=epoch > int(cfg["train"]["freeze_backbone_epochs"]))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_total = 0.0
        progress = tqdm(train_loader, desc=f"train {epoch}", leave=False)

        for step, batch in enumerate(progress, start=1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            if bool(cfg["model"].get("channels_last", False)) and device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with autocast_ctx:
                logits, logits16, logits32 = model(images)
                target_size = (int(cfg["model"]["output_size"]), int(cfg["model"]["output_size"]))
                logits = torch.nn.functional.interpolate(logits, size=target_size, mode="bilinear", align_corners=True)
                logits16 = torch.nn.functional.interpolate(logits16, size=target_size, mode="bilinear", align_corners=True)
                logits32 = torch.nn.functional.interpolate(logits32, size=target_size, mode="bilinear", align_corners=True)
                losses = criterion(logits, logits16, logits32, targets)
                loss = losses["total"] / int(cfg["train"]["grad_accum_steps"])

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % int(cfg["train"]["grad_accum_steps"]) == 0 or step == len(train_loader):
                if float(cfg["train"]["gradient_clip_norm"]) > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["gradient_clip_norm"]))
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if latest_step_interval > 0 and global_step % latest_step_interval == 0:
                    save_training_checkpoint(
                        path=run_dir / "latest.pth",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler if scaler.is_enabled() else None,
                        epoch=epoch,
                        global_step=global_step,
                        metrics={"status": "in_progress"},
                        config=cfg,
                    )

            running_total += float(losses["total"].detach().cpu())
            progress.set_postfix(loss=f"{running_total / step:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        eval_result = run_evaluation(
            model=model,
            dataloader=val_loader,
            device=device,
            output_size=int(cfg["model"]["output_size"]),
            coarse_labels=cfg["data"]["coarse_labels"],
            eval_cfg=cfg["eval"],
            save_dir=run_dir / "val_predictions" / f"epoch_{epoch:03d}",
        )
        metric = float(eval_result["overall"].get("hair_dice", 0.0))
        checkpoint_path = run_dir / f"epoch_{epoch:03d}.pth"
        save_training_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if scaler.is_enabled() else None,
            epoch=epoch,
            global_step=global_step,
            metrics=eval_result,
            config=cfg,
        )
        save_training_checkpoint(
            path=run_dir / "latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if scaler.is_enabled() else None,
            epoch=epoch,
            global_step=global_step,
            metrics=eval_result,
            config=cfg,
        )

        if metric >= best_metric:
            best_metric = metric
            save_training_checkpoint(
                path=run_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if scaler.is_enabled() else None,
                epoch=epoch,
                global_step=global_step,
                metrics=eval_result,
                config=cfg,
            )
            copy_best_alias(run_dir / "best.pth", run_dir, alias_name="seg_sd_ft.pth")

        append_history(
            history_path,
            {
                "epoch": epoch,
                "train_loss": running_total / max(1, len(train_loader)),
                "val_hair_iou": eval_result["overall"].get("hair_iou", 0.0),
                "val_hair_dice": eval_result["overall"].get("hair_dice", 0.0),
                "val_boundary_f1": eval_result["overall"].get("boundary_f1", 0.0),
                "val_face_spill_rate": eval_result["overall"].get("face_spill_rate", 0.0),
                "lr": optimizer.param_groups[0]["lr"],
            },
        )
        save_json(run_dir / f"metrics_epoch_{epoch:03d}.json", eval_result)


if __name__ == "__main__":
    main()
