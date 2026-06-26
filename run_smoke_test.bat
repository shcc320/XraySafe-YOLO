@echo off
setlocal enabledelayedexpansion
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts\test_custom_modules.py
python scripts\patch_ultralytics.py
python scripts\smoke_custom_yolo.py
