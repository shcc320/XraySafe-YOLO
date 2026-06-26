#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/test_custom_modules.py
python scripts/patch_ultralytics.py
python scripts/smoke_custom_yolo.py
