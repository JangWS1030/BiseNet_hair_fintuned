from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.common import maybe_link_or_copy, save_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ID_CANDIDATES = ["image_id", "imageid", "img_id", "id", "image_name", "file_name", "filename", "path"]
FILE_CANDIDATES = ["image_path", "file_path", "path", "file_name", "filename", "image_name", "img_name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AIHub 85 for coarse BiSeNet fine-tuning.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qualities", nargs="+", default=["hq"], choices=["hq", "mq", "raw"])
    parser.add_argument("--copy-mode", default="hardlink", choices=["hardlink", "copy", "symlink", "none"])
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise RuntimeError(f"Failed to read csv: {path}") from last_error


def find_column(columns: list[str], keywords: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for keyword in keywords:
        if keyword.lower() in lowered:
            return lowered[keyword.lower()]
    for column in columns:
        normalized = column.lower().replace(" ", "_")
        if any(keyword.lower() in normalized for keyword in keywords):
            return column
    return None


def load_package_tables(annotation_csv: Path) -> tuple[pd.DataFrame, str, str]:
    package_dir = annotation_csv.parent
    annotation_df = read_csv_auto(annotation_csv)
    for sibling_name in ["image.csv", "attribute.csv", "meta-annotation.csv"]:
        sibling = package_dir / sibling_name
        if sibling.exists():
            right = read_csv_auto(sibling)
            shared = [column for column in annotation_df.columns if column in right.columns]
            join_col = find_column(shared, ID_CANDIDATES)
            if join_col is None:
                continue
            right = right.drop_duplicates(subset=[join_col])
            overlap = [column for column in right.columns if column in annotation_df.columns and column != join_col]
            if overlap:
                right = right.rename(columns={column: f"{column}_{sibling.stem}" for column in overlap})
            annotation_df = annotation_df.merge(right, on=join_col, how="left")

    source_split = "train" if "training" in str(annotation_csv).lower() else "val"
    quality = "hq" if "hqset" in str(annotation_csv).lower() else "mq" if "mqset" in str(annotation_csv).lower() else "raw"
    return annotation_df, source_split, quality


def parse_polygon_value(value: Any) -> list[np.ndarray]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        parsed = value
    else:
        text = str(value).strip()
        if not text or text in {"[]", "nan", "None"}:
            return []
        try:
            parsed = json.loads(text.replace("'", '"'))
        except Exception:  # noqa: BLE001
            parsed = ast.literal_eval(text)

    if isinstance(parsed, dict):
        parsed = parsed.get("points", [])
    if not isinstance(parsed, list):
        return []

    if parsed and isinstance(parsed[0], dict):
        parsed = [parsed]

    polygons: list[np.ndarray] = []
    for polygon in parsed:
        if not polygon:
            continue
        if isinstance(polygon, dict):
            polygon = polygon.get("points", [])
        if isinstance(polygon, list) and polygon and isinstance(polygon[0], (int, float)):
            polygon = [polygon[i : i + 2] for i in range(0, len(polygon), 2)]
        points = []
        for point in polygon:
            if isinstance(point, dict):
                x = point.get("x")
                y = point.get("y")
            else:
                x, y = point[0], point[1]
            points.append([int(round(float(x))), int(round(float(y)))])
        if len(points) >= 3:
            polygons.append(np.array(points, dtype=np.int32))
    return polygons


def build_image_index(raw_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.name.lower()].append(path)
    return index


def resolve_image_path(row: pd.Series, raw_dir: Path, image_index: dict[str, list[Path]]) -> Path | None:
    for candidate in FILE_CANDIDATES:
        if candidate in row and pd.notna(row[candidate]):
            value = str(row[candidate]).strip()
            if not value:
                continue
            direct = Path(value)
            if direct.is_absolute() and direct.exists():
                return direct
            rel = (raw_dir / value).resolve()
            if rel.exists():
                return rel
            basename = Path(value).name.lower()
            if basename in image_index:
                return image_index[basename][0]
    key_column = find_column(list(row.index), ID_CANDIDATES)
    if key_column is not None and pd.notna(row[key_column]):
        key = str(row[key_column]).strip()
        for extension in IMAGE_EXTENSIONS:
            basename = f"{key}{extension}".lower()
            if basename in image_index:
                return image_index[basename][0]
    return None


def draw_polygons(mask: np.ndarray, polygons: list[np.ndarray], value: int) -> None:
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], color=value)


def sanitize_sample_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("_") or "sample"


def fuzzy_pick(row: pd.Series, keywords: list[str]) -> str:
    for column in row.index:
        normalized = column.lower().replace(" ", "_")
        if any(keyword in normalized for keyword in keywords) and pd.notna(row[column]):
            return str(row[column]).strip()
    return ""


def fuzzy_bool(row: pd.Series, keywords: list[str]) -> str:
    value = fuzzy_pick(row, keywords)
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "있음", "유"}:
        return "true"
    if lowered in {"0", "false", "no", "n", "없음", "무"}:
        return "false"
    if any(keyword in lowered for keyword in ["short", "bang", "sideburn", "dark", "black"]):
        return "true"
    return lowered


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    annotation_csvs = [path for path in raw_dir.rglob("annotation.csv") if any(q in str(path).lower() for q in args.qualities)]
    if not annotation_csvs:
        raise FileNotFoundError("annotation.csv not found under raw_dir")

    image_index = build_image_index(raw_dir)
    manifest_rows: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = defaultdict(int)
    processed = 0

    for annotation_csv in annotation_csvs:
        merged_df, source_split, quality = load_package_tables(annotation_csv)
        key_column = find_column(list(merged_df.columns), ID_CANDIDATES)
        if key_column is None:
            key_column = merged_df.columns[0]

        for _, group in tqdm(merged_df.groupby(key_column), desc=f"prepare-{quality}", leave=False):
            row = group.iloc[0]
            image_path = resolve_image_path(row, raw_dir=raw_dir, image_index=image_index)
            if image_path is None:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            height, width = image.shape[:2]
            face_mask = np.zeros((height, width), dtype=np.uint8)
            hair_mask = np.zeros((height, width), dtype=np.uint8)
            for _, polygon_row in group.iterrows():
                draw_polygons(face_mask, parse_polygon_value(polygon_row.get("polygon2")), value=1)
                draw_polygons(hair_mask, parse_polygon_value(polygon_row.get("polygon1")), value=10)

            coarse_mask = np.zeros((height, width), dtype=np.uint8)
            coarse_mask[face_mask > 0] = 1
            coarse_mask[hair_mask > 0] = 10

            base_id = sanitize_sample_id(str(Path(image_path).stem))
            seen_ids[base_id] += 1
            sample_id = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]:03d}"
            image_out = images_dir / f"{sample_id}{image_path.suffix.lower()}"
            mask_out = masks_dir / f"{sample_id}.png"
            maybe_link_or_copy(image_path, image_out, mode=args.copy_mode)
            cv2.imwrite(str(mask_out), coarse_mask)

            subject_id = fuzzy_pick(row, ["subject", "person", "model", "identity"])
            sequence_id = fuzzy_pick(row, ["sequence", "shot", "scene", "session", "take", "video"])
            group_id = subject_id or sequence_id or base_id
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(image_out.relative_to(out_dir)),
                    "mask_path": str(mask_out.relative_to(out_dir)),
                    "source_split": source_split,
                    "quality": quality,
                    "subject_id": subject_id,
                    "sequence_id": sequence_id,
                    "group_id": group_id,
                    "short_hair": fuzzy_bool(row, ["short", "단발", "쇼트"]),
                    "bangs": fuzzy_bool(row, ["bang", "앞머리"]),
                    "sideburn": fuzzy_bool(row, ["sideburn", "구레나룻"]),
                    "dark_hair": fuzzy_bool(row, ["dark", "black", "hair_color", "색상", "검"]),
                    "orig_width": width,
                    "orig_height": height,
                }
            )
            processed += 1
            if args.max_samples is not None and processed >= args.max_samples:
                break
        if args.max_samples is not None and processed >= args.max_samples:
            break

    manifest_path = out_dir / "manifest.csv"
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    save_json(
        out_dir / "prepare_summary.json",
        {
            "raw_dir": str(raw_dir),
            "out_dir": str(out_dir),
            "qualities": args.qualities,
            "num_samples": len(manifest_rows),
            "copy_mode": args.copy_mode,
            "assumptions": [
                "polygon1 is hair and mapped to label 10",
                "polygon2 is face and mapped to label 1",
                "unlabeled pixels default to background 0",
                "ignore label 255 is reserved for runtime padding/invalid regions",
            ],
        },
    )
    print(f"Prepared {len(manifest_rows)} samples -> {manifest_path}")


if __name__ == "__main__":
    main()

