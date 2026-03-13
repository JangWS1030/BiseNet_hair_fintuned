from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on an external prepared dataset such as AIHub 83.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--save-dir", required=True)
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

    manifest_path = Path(args.manifest).resolve() if args.manifest else resolve_path(cfg["data"].get("external_manifest_path"), cfg)
    if args.split_file:
        split_path = Path(args.split_file).resolve()
    else:
        default_splits_dir = manifest_path.parent / "splits"
        split_path = default_splits_dir / cfg["data"].get("external_split", "external.txt")
    dataloader = build_dataloader(
        manifest_path=manifest_path,
        split_path=split_path if split_path.exists() else None,
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
    save_json(Path(args.save_dir) / "external_metrics.json", result)
    print(result)


if __name__ == "__main__":
    main()
