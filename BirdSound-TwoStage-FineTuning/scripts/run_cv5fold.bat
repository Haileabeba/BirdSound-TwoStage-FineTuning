@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    echo Run: python -m venv .venv ^& pip install -r requirements.txt ^& pip install -e .
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
set CONFIG=%~1
if "%CONFIG%"=="" set CONFIG=configs\cv5fold.yaml
python -m progressive_hubert.cli --experiment cv5fold --config %CONFIG%
pause
