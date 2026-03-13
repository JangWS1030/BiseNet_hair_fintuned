from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BiSeNet checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.datasets import build_dataloader
    from src.evaluation import run_evaluation
    from src.models import BiSeNet
    from src.utils.checkpoint import load_model_checkpoint
    from src.utils.common import save_json
    from src.utils.config import load_config, resolve_path

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = resolve_path(cfg["data"]["manifest_path"], cfg)
    splits_dir = resolve_path(cfg["data"]["splits_dir"], cfg)
    split_name = cfg["data"]["val_split"] if args.split == "val" else cfg["data"]["test_split"]
    dataloader = build_dataloader(
        manifest_path=manifest_path,
        split_path=splits_dir / split_name,
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
    model = BiSeNet(n_classes=int(cfg["model"]["num_classes"]), pretrained_backbone=False).to(device)
    load_model_checkpoint(model, args.checkpoint, strict=False, allow_partial=True)
    result = run_evaluation(
        model=model,
        dataloader=dataloader,
        device=device,
        output_size=int(cfg["model"]["output_size"]),
        coarse_labels=cfg["data"]["coarse_labels"],
        eval_cfg=cfg["eval"],
        save_dir=args.save_dir,
    )
    print(result)
    if args.save_dir is not None:
        save_json(Path(args.save_dir) / "metrics.json", result)


if __name__ == "__main__":
    main()
