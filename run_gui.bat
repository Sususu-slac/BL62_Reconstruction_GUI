@echo off
setlocal
cd /d "%~dp0"
set PYTHONNOUSERSITE=1
set PYTHONUTF8=1
where conda >nul 2>nul
if errorlevel 1 (
  echo Conda was not found in PATH. Open Miniforge Prompt and run this file again.
  pause
  exit /b 1
)
call conda run --no-capture-output -n tomocupy_astra pythonw app.py
if errorlevel 1 pause
