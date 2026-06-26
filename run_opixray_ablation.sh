#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_opixray_ablation.sh /path/to/OPIXray_raw 0
# Runs YOLO11n baseline and the three manuscript-aligned ablation variants:
#   YOLO11n, YOLO11n+CGOA, YOLO11n+LSAFF, XraySafe-YOLO full.

RAW_ROOT=${1:-datasets_raw/OPIXray}
DEVICE=${2:-0}

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/test_custom_modules.py
python scripts/patch_ultralytics.py

python scripts/run_all.py \
  --dataset opixray \
  --raw-root "$RAW_ROOT" \
  --out-root experiments/opixray_ablation \
  --models \
    yolo11n.pt \
    configs/models/xraysafe_yolo11n_cgoa.yaml \
    configs/models/xraysafe_yolo11n_lsaff.yaml \
    configs/models/xraysafe_yolo11n_full.yaml \
  --pretrained yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device "$DEVICE" \
  --workers 8 \
  --seed 42 \
  --cos-lr \
  --save-json

python scripts/make_paper_tables.py \
  --metrics-dir experiments/opixray_ablation/metrics \
  --out-dir paper_tables/opixray_ablation
