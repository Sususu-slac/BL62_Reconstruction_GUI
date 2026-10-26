@echo off
setlocal
cd /d "%~dp0"
set PYTHONNOUSERSITE=1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0prepare_environments.ps1"
if errorlevel 1 (
 echo Environment setup failed. Read the output above.
 pause
 exit /b 1
)
echo Environment setup completed.
pause
