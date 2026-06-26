@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Python executable not found: "%PYTHON%"
  exit /b 1
)

if "%XRAYSAFE_EPOCHS%"=="" set "XRAYSAFE_EPOCHS=100"
if "%XRAYSAFE_OUT_ROOT%"=="" set "XRAYSAFE_OUT_ROOT=experiments\opixray_opt_v3_formal"
if "%XRAYSAFE_TABLES_OUT%"=="" set "XRAYSAFE_TABLES_OUT=paper_tables\opixray_opt_v3_formal"
if "%XRAYSAFE_PRETRAINED%"=="" set "XRAYSAFE_PRETRAINED=experiments\opixray_ablation\runs_train\xraysafe_yolo11n_cgoa\weights\best.pt"

if not exist "%XRAYSAFE_PRETRAINED%" (
  echo Preferred CGOA pretrained checkpoint not found: "%XRAYSAFE_PRETRAINED%"
  echo Falling back to yolo11n.pt
  set "XRAYSAFE_PRETRAINED=yolo11n.pt"
)

echo Starting OPIXray v3 formal candidate run at %DATE% %TIME%
echo Root: %ROOT%
echo Python: %PYTHON%
echo OutRoot: %XRAYSAFE_OUT_ROOT%
echo TablesOut: %XRAYSAFE_TABLES_OUT%
echo Epochs: %XRAYSAFE_EPOCHS%
echo Pretrained: %XRAYSAFE_PRETRAINED%
echo Models: opt_p34_resgated, opt_spatial_resgated

if not "%XRAYSAFE_DRY_RUN%"=="" (
  echo Dry run requested; exiting before training.
  exit /b 0
)

"%PYTHON%" scripts\test_custom_modules.py
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" scripts\patch_ultralytics.py
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" scripts\run_all.py ^
  --dataset opixray ^
  --raw-root .\OPIXray ^
  --out-root "%XRAYSAFE_OUT_ROOT%" ^
  --models ^
    configs/models/xraysafe_yolo11n_opt_p34_resgated.yaml ^
    configs/models/xraysafe_yolo11n_opt_spatial_resgated.yaml ^
  --pretrained "%XRAYSAFE_PRETRAINED%" ^
  --epochs %XRAYSAFE_EPOCHS% ^
  --imgsz 640 ^
  --batch 16 ^
  --device 0 ^
  --workers 8 ^
  --seed 42 ^
  --patience 30 ^
  --cos-lr ^
  --save-json
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" scripts\make_paper_tables.py ^
  --metrics-dir "%XRAYSAFE_OUT_ROOT%\metrics" ^
  --out-dir "%XRAYSAFE_TABLES_OUT%"
if errorlevel 1 exit /b %errorlevel%

echo Finished OPIXray v3 formal candidate run at %DATE% %TIME%
exit /b 0
