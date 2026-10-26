$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path 'runtime_sam3\sam3\model_builder.py')) {
 python prepare_sam3_source.py
 if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the pinned SAM3 source.' }
}
python prepare_worker_code.py
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare isolated worker scripts.' }
python -m PyInstaller --noconfirm --clean --windowed --onedir --name BL62_Reconstruction_GUI --icon BL62_Recon.ico `
 --add-data 'BL62_Recon.ico;.' --add-data 'worker_code;worker_code' --add-data 'runtime_sam3;sam3_repo' `
 --add-data 'environment.yml;.' --add-data 'requirements-sam3-windows.txt;.' --add-data 'prepare_environments.ps1;.' `
 --add-data 'verify_installation.py;.' --add-data 'verify_sam3.py;.' --add-data 'install_windows.bat;.' --add-data 'THIRD_PARTY_NOTICES.md;.' --add-data 'licenses;licenses' `
 --exclude-module torch --exclude-module torchvision --exclude-module cupy --exclude-module astra --exclude-module scipy --exclude-module matplotlib `
 --exclude-module pandas --exclude-module IPython --exclude-module pytest --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 app.py
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }
$iscc=Join-Path $env:LOCALAPPDATA 'BL62-BuildTools\InnoSetup\ISCC.exe'
if (-not (Test-Path $iscc)) { $iscc='C:\Program Files (x86)\Inno Setup 6\ISCC.exe' }
if (-not (Test-Path $iscc)) { throw 'Install Inno Setup 6 to build the Windows installer.' }
& $iscc installer.iss
if ($LASTEXITCODE -ne 0) { throw 'Installer build failed' }
