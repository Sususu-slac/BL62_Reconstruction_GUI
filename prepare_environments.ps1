$ErrorActionPreference = 'Stop'
# Keep Conda interpreters away from the frozen GUI's extension-module directory.
Set-Location ([System.IO.Path]::GetTempPath())
$candidates = @(
 (Join-Path $env:LOCALAPPDATA 'miniforge3\condabin\conda.bat'),
 (Join-Path $env:USERPROFILE 'miniforge3\condabin\conda.bat'),
 (Join-Path $env:USERPROFILE 'miniconda3\condabin\conda.bat'),
 (Join-Path $env:USERPROFILE 'anaconda3\condabin\conda.bat')
)
$conda = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $conda) { $cmd = Get-Command conda -ErrorAction SilentlyContinue; if ($cmd) { $conda=$cmd.Source } }
if (-not $conda) { throw 'Install Miniforge first, then run Install Environments again.' }
function Invoke-Conda {
 & $conda @args
 if ($LASTEXITCODE -ne 0) { throw "Conda failed: $args" }
}
$envs = (& $conda env list --json | ConvertFrom-Json).envs
$recon = $envs | Where-Object { (Split-Path $_ -Leaf) -eq 'tomocupy_astra' } | Select-Object -First 1
if (-not $recon) { Invoke-Conda env create -f (Join-Path $PSScriptRoot 'environment.yml'); $envs=(& $conda env list --json | ConvertFrom-Json).envs; $recon=$envs | Where-Object { (Split-Path $_ -Leaf) -eq 'tomocupy_astra' } | Select-Object -First 1 }
Invoke-Conda run --no-capture-output -n tomocupy_astra python -I (Join-Path $PSScriptRoot 'verify_installation.py')
$sam3 = $envs | Where-Object { (Split-Path $_ -Leaf) -eq 'SAM3p1' } | Select-Object -First 1
if (-not $sam3) {
 Invoke-Conda create -n SAM3p1 python=3.12 pip -y
 Invoke-Conda run --no-capture-output -n SAM3p1 python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
}
Invoke-Conda run --no-capture-output -n SAM3p1 python -m pip install -r (Join-Path $PSScriptRoot 'requirements-sam3-windows.txt')
$repo = 'D:\SAM3p1\sam3'
if (-not (Test-Path (Join-Path $repo 'sam3\__init__.py'))) {
 $repo = Join-Path $PSScriptRoot 'sam3_repo'
 if (-not (Test-Path (Join-Path $repo 'sam3\__init__.py'))) { $repo = Join-Path $PSScriptRoot 'runtime_sam3' }
}
if (-not (Test-Path (Join-Path $repo 'pyproject.toml'))) { throw 'SAM3 source not found. Set the SAM3 repository in Settings.' }
Invoke-Conda run --no-capture-output -n SAM3p1 python -m pip install --no-deps -e $repo
Invoke-Conda run --no-capture-output -n SAM3p1 python -I (Join-Path $PSScriptRoot 'verify_sam3.py') --repo $repo
Write-Host 'Both isolated Windows environments are ready.' -ForegroundColor Green
Write-Host 'Select your checkpoint in Settings if D:\Chcekpoints\checkpoint_demo_05142026.pt is unavailable.'
