#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data outputs

echo "Bootstrap complete."
echo "Next:"
echo "1. cp .env.example .env"
echo "2. edit .env and set AIHUB_APIKEY"
echo "3. python scripts/download_aihub85.py --out-dir data/aihub85_raw --profile hq --stage all"
echo "4. python scripts/prepare_aihub85.py --raw-dir data/aihub85_raw --out-dir data/aihub85_prepared"
echo "5. python scripts/make_splits.py --data-dir data/aihub85_prepared --seed 42"
echo "6. edit configs/bisenet_ft.yaml and set model.init_checkpoint"
echo "7. bash scripts/runpod_train.sh configs/bisenet_ft.yaml 6 on_success"
