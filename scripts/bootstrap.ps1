[CmdletBinding()]
param(
    [switch]$SkipEnvironment,
    [switch]$SkipGpuPackages
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$environmentName = "timbre-share"
$conda = (Get-Command conda.exe -ErrorAction SilentlyContinue).Source
if ([string]::IsNullOrWhiteSpace($conda)) {
    $conda = "D:\miniconda3\Scripts\conda.exe"
}
if (-not (Test-Path -LiteralPath $conda -PathType Leaf)) {
    throw "conda.exe was not found; install Miniconda under the documented path"
}
$python = "D:\miniconda3\envs\timbre-share\python.exe"

$runtimeDirectories = @(
    "data/uploads",
    "data/datasets",
    "data/features",
    "data/profiles",
    "data/outputs",
    "data/temp",
    "data/logs",
    "models"
)

foreach ($directory in $runtimeDirectories) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root $directory) | Out-Null
}

if (-not $SkipEnvironment) {
    $environmentExists = & $conda env list | Select-String -Pattern "^\s*${environmentName}\s+"
    if ($null -eq $environmentExists) {
        & $conda env create --file (Join-Path $root "environment.yml")
    }
    else {
        & $conda env update --name $environmentName --file (Join-Path $root "environment.yml")
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Conda environment setup failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipGpuPackages) {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "固定 timbre-share Python 解释器不存在: $python"
    }
    & $python -m pip install `
        --timeout 30 `
        --retries 1 `
        --index-url https://download.pytorch.org/whl/cu124 `
        torch==2.5.1 torchaudio==2.5.1
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch installation failed with exit code $LASTEXITCODE"
    }

    & $python -c "import torch; assert torch.__version__.startswith('2.5.1'), torch.__version__; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 1"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA runtime probe failed with exit code $LASTEXITCODE"
    }
    Write-Output "GPU_PACKAGES=PASS"
}
else {
    Write-Output "GPU_PACKAGES=SKIPPED"
}

Write-Output "BOOTSTRAP=PASS"
Write-Output "ROOT=$root"
Write-Output "ENVIRONMENT=$environmentName"
