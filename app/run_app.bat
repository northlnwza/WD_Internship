@echo off
REM Launch the YOLO Model Tester desktop app.
REM Reads models + images from the sibling ML_pipeline\ folder.
cd /d "%~dp0"

REM Prefer the project virtualenv (has ultralytics/torch/Pillow); fall back to system python.
set "VENV_PY=%~dp0..\.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    "%VENV_PY%" model_tester.py
) else (
    python model_tester.py
)
if errorlevel 1 pause
