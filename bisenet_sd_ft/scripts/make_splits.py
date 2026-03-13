from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.common import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test splits for prepared AIHub 85 data.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--group-col", default=None)
    return parser.parse_args()


def ratio_penalty(current: int, total: int, target: float) -> float:
    return abs((current / max(1, total)) - target)


def choose_group_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for column in ["group_id", "subject_id", "sequence_id"]:
        if column in df.columns and df[column].astype(str).str.len().gt(0).any():
            return column
    return "sample_id"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    df = pd.read_csv(manifest_path)
    group_col = choose_group_column(df, args.group_col)
    attr_cols = [column for column in ["short_hair", "bangs", "sideburn", "dark_hair"] if column in df.columns]

    grouped_rows = list(df.groupby(group_col))
    random.Random(args.seed).shuffle(grouped_rows)
    grouped_rows.sort(key=lambda item: len(item[1]), reverse=True)
    total_samples = len(df)
    target_ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    split_members: dict[str, list[str]] = defaultdict(list)
    split_counts = defaultdict(int)
    split_attr_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    global_attr_ratios = {
        attr: df[attr].astype(str).str.lower().isin({"1", "true", "yes"}).mean() for attr in attr_cols
    }

    for group_value, group_df in grouped_rows:
        group_size = len(group_df)
        group_attrs = {
            attr: int(group_df[attr].astype(str).str.lower().isin({"1", "true", "yes"}).sum()) for attr in attr_cols
        }
        best_split = None
        best_score = None
        for split_name in ["train", "val", "test"]:
            projected_count = split_counts[split_name] + group_size
            score = ratio_penalty(projected_count, total_samples, target_ratios[split_name]) * 3.0
            for attr in attr_cols:
                projected_ratio = (split_attr_counts[split_name][attr] + group_attrs[attr]) / max(1, projected_count)
                score += abs(projected_ratio - global_attr_ratios[attr])
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        assert best_split is not None
        split_counts[best_split] += group_size
        split_members[best_split].extend(group_df["sample_id"].astype(str).tolist())
        for attr in attr_cols:
            split_attr_counts[best_split][attr] += group_attrs[attr]

    splits_dir = ensure_dir(data_dir / "splits")
    summary: dict[str, Any] = {"group_col": group_col, "seed": args.seed, "counts": {}, "ratios": {}}
    for split_name, members in split_members.items():
        (splits_dir / f"{split_name}.txt").write_text("\n".join(members) + "\n", encoding="utf-8")
        summary["counts"][split_name] = len(members)
        summary["ratios"][split_name] = len(members) / max(1, total_samples)

    save_json(splits_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
