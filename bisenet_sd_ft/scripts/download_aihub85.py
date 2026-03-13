from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.common import ensure_dir


DATASET_KEY = 85
API_VERSION = "0.6"
BASE_DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{API_VERSION}/{DATASET_KEY}.do"

TRAIN_RAW_KEYS = list(range(44608, 44620)) + list(range(44525, 44537))
VAL_RAW_KEYS = [44542, 44543, 44544]
PROFILE_KEYS = {
    "hq": {"train": [44601, 44604, 44605, 44606], "val": [44537, 44540]},
    "hq_mq": {"train": [44601, 44602, 44604, 44605, 44606, 44607], "val": [44537, 44538, 44540, 44541]},
    "full": {"train": [44601, 44602, 44603, 44604, 44605, 44606, 44607, *TRAIN_RAW_KEYS], "val": [44537, 44538, 44539, 44540, 44541, *VAL_RAW_KEYS]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download AIHub 85 directly inside the workspace or Runpod.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile", default="hq", choices=["hq", "hq_mq", "full"])
    parser.add_argument("--stage", default="all", choices=["train", "val", "all"])
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_api_key(explicit_env_file: str | None) -> str:
    if explicit_env_file:
        load_dotenv(explicit_env_file)
    else:
        discovered = find_dotenv(usecwd=True)
        if discovered:
            load_dotenv(discovered)
    api_key = os.getenv("AIHUB_APIKEY", "").strip()
    if not api_key:
        raise RuntimeError("AIHUB_APIKEY not found. Put it in .env or export it in the shell.")
    return api_key


def merge_part_files(target_dir: Path) -> None:
    grouped: dict[str, list[Path]] = {}
    for part in target_dir.rglob("*.part*"):
        prefix = str(part).rsplit(".part", 1)[0]
        grouped.setdefault(prefix, []).append(part)
    for prefix, parts in grouped.items():
        parts.sort(key=lambda item: item.suffix)
        out_path = Path(prefix)
        with out_path.open("wb") as handle:
            for part in parts:
                handle.write(part.read_bytes())
        for part in parts:
            part.unlink()


def stream_download(url: str, api_key: str, tar_path: Path) -> None:
    with requests.get(url, headers={"apikey": api_key}, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tar_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_tar(tar_path: Path, out_dir: Path) -> None:
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=out_dir)
    tar_path.unlink(missing_ok=True)
    merge_part_files(out_dir)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    keys = []
    if args.stage in {"train", "all"}:
        keys.extend(PROFILE_KEYS[args.profile]["train"])
    if args.stage in {"val", "all"}:
        keys.extend(PROFILE_KEYS[args.profile]["val"])
    url = f"{BASE_DOWNLOAD_URL}?fileSn={{{','.join(str(key) for key in keys)}}}"

    if args.dry_run:
        print(url)
        print(keys)
        return

    api_key = load_api_key(args.env_file)
    tar_path = out_dir / f"aihub85_{args.profile}_{args.stage}.tar"
    stream_download(url=url, api_key=api_key, tar_path=tar_path)
    extract_tar(tar_path=tar_path, out_dir=out_dir)
    print(f"Downloaded and extracted to {out_dir}")


if __name__ == "__main__":
    main()

