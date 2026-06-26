@echo off
setlocal enabledelayedexpansion
REM Usage: run_opixray_ablation.bat C:\path\to\OPIXray_raw 0
set RAW_ROOT=%1
if "%RAW_ROOT%"=="" set RAW_ROOT=datasets_raw\OPIXray
set DEVICE=%2
if "%DEVICE%"=="" set DEVICE=0

python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts\test_custom_modules.py
python scripts\patch_ultralytics.py

python scripts\run_all.py ^
  --dataset opixray ^
  --raw-root "%RAW_ROOT%" ^
  --out-root experiments\opixray_ablation ^
  --models yolo11n.pt configs/models/xraysafe_yolo11n_cgoa.yaml configs/models/xraysafe_yolo11n_lsaff.yaml configs/models/xraysafe_yolo11n_full.yaml ^
  --pretrained yolo11n.pt ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch 16 ^
  --device "%DEVICE%" ^
  --workers 8 ^
  --seed 42 ^
  --cos-lr ^
  --save-json

python scripts\make_paper_tables.py ^
  --metrics-dir experiments\opixray_ablation\metrics ^
  --out-dir paper_tables\opixray_ablation
