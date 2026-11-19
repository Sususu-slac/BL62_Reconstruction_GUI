param([string]$RuntimeArchive='', [string]$InstallRoot='', [string]$DistributionName='BL62-Tofu', [string]$ConfigurationPath='')
$ErrorActionPreference='Stop'
$distro=$DistributionName
if ($distro -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$') { throw 'Invalid WSL distribution name.' }
$names=((& wsl.exe --list --quiet) -join "`n") -replace "`0",''
if ($LASTEXITCODE -ne 0) { throw 'Tofu GPU requires WSL2. Enable WSL2 before running this setup.' }
if (($names -split "`n" | ForEach-Object {$_.Trim()}) -notcontains $distro) {
    if (-not $RuntimeArchive) {
        $candidates=@((Join-Path $PSScriptRoot 'runtime\BL62-Tofu-GPU-1.0.0.tar.gz'),
                       (Join-Path $PSScriptRoot 'BL62-Tofu-GPU-1.0.0.tar.gz'),
                       'D:\BL62_Runtime\Tofu\BL62-Tofu-GPU-1.0.0.tar.gz')
        $RuntimeArchive=$candidates | Where-Object {Test-Path -LiteralPath $_} | Select-Object -First 1
    }
    if (-not $RuntimeArchive -or -not (Test-Path -LiteralPath $RuntimeArchive)) {
        throw 'Place the separate BL62-Tofu-GPU-1.0.0.tar.gz runtime and its .sha256 file in D:\BL62_Runtime\Tofu, then retry.'
    }
    $checksum="$RuntimeArchive.sha256"
    if (-not (Test-Path -LiteralPath $checksum)) { throw "Runtime checksum is missing: $checksum" }
    $expected=((Get-Content -LiteralPath $checksum -Raw).Trim() -split '\s+')[0]
    $actual=(Get-FileHash -LiteralPath $RuntimeArchive -Algorithm SHA256).Hash
    if ($actual -ne $expected) { throw 'Tofu runtime checksum does not match. Do not use this archive.' }
    if (-not $InstallRoot) {
        $sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
        if (Test-Path 'D:\') { $InstallRoot="D:\BL62_Runtime\Tofu\WSL\$sid\$distro" }
        else { $InstallRoot=Join-Path $env:LOCALAPPDATA "BL62\Tofu\WSL\$distro" }
    }
    New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
    $ownerSid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    & icacls.exe $InstallRoot /grant "*$($ownerSid):(OI)(CI)F" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not grant the current account access to its private runtime directory.' }
    Write-Output 'Installing the private Tofu GPU runtime for this Windows account...'
    & wsl.exe --import $distro $InstallRoot $RuntimeArchive --version 2
    if ($LASTEXITCODE -ne 0) { throw 'WSL could not import the Tofu runtime.' }
}
# Verify the isolated runtime before telling the GUI that setup succeeded.
& wsl.exe --distribution $distro --exec /opt/bl62-tofu/python/bin/python /opt/bl62-tofu/check_runtime.py
if ($LASTEXITCODE -ne 0) { throw 'The Tofu GPU runtime self-check failed. See the runtime log before retrying.' }
# No personal source files or another user's Linux home are required.
$config=@{distribution=$distro;python='/opt/bl62-tofu/python/bin/python';ufo_python='/opt/bl62-tofu/python/bin/python';
          cuda_prefix='/opt/bl62-tofu/python';ufo_prefix='/opt/bl62-tofu/ufo';plugin_root='/opt/bl62-tofu/plugin'}
if (-not $ConfigurationPath) { $ConfigurationPath=Join-Path $env:APPDATA 'BL62_Reconstruction\tofu_runtime.json' }
$directory=Split-Path $ConfigurationPath -Parent
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$utf8=New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($ConfigurationPath,($config | ConvertTo-Json),$utf8)
Write-Output 'Tofu GPU runtime configured for this Windows account.'
