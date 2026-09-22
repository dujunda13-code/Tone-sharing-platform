[CmdletBinding()]
param(
    [switch]$SkipGpu,
    [string]$AuthorizedReference = $env:TIMBRE_AUTHORIZED_REFERENCE
)

# Final acceptance is the local zero-shot client contract. `-SkipGpu` is only
# a CPU_ONLY diagnostic mode and can never produce a final PASS.

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$python = "D:\miniconda3\envs\timbre-share\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "固定 timbre-share Python 解释器不存在: $python"
}
$node = (Get-Command node.exe -ErrorAction SilentlyContinue).Source
if ([string]::IsNullOrWhiteSpace($node)) { throw "node.exe was not found" }
$viteEntry = Join-Path $root "frontend/node_modules/vite/bin/vite.js"
$vitestEntry = Join-Path $root "frontend/node_modules/vitest/vitest.mjs"
$playwrightEntry = Join-Path $root "frontend/node_modules/@playwright/test/cli.js"
foreach ($entry in $viteEntry, $vitestEntry, $playwrightEntry) {
    if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
        throw "前端依赖入口缺失: $entry"
    }
}
$env:TIMBRE_AUTHORIZED_REFERENCE = $AuthorizedReference
$reportPath = Join-Path $root "reports/final-evaluation.json"
New-Item -ItemType Directory -Force -Path (Split-Path $reportPath) | Out-Null
$pytestBase = Join-Path $root "tmp/verify-pytest"
New-Item -ItemType Directory -Force -Path $pytestBase | Out-Null
$report = [ordered]@{
    generated_at = [DateTime]::UtcNow.ToString("o")
    authorized_reference_supplied = [bool](-not [string]::IsNullOrWhiteSpace($AuthorizedReference))
    skip_gpu = [bool]$SkipGpu
    checks = [ordered]@{}
    final_result = "INCOMPLETE"
    first_failure = $null
}

function Invoke-Checked {
    param([string]$Name, [scriptblock]$Action)
    if ($null -ne $report.first_failure) { return }
    try {
        & $Action
        if ($LASTEXITCODE -ne 0) { throw "$Name exited with code $LASTEXITCODE" }
        $report.checks[$Name] = "PASS"
        Write-Output "$Name=PASS"
    }
    catch {
        $report.checks[$Name] = "FAIL"
        $report.first_failure = $_.Exception.Message
        Write-Output "$Name=FAIL: $($report.first_failure)"
    }
}

Invoke-Checked "MODEL_CHECKSUMS" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts/download_models.ps1")
}
Invoke-Checked "BACKEND_LINT" {
    & $python -m ruff check backend/app backend/tests
}
Invoke-Checked "BACKEND_CPU_TESTS" {
    & $python -m pytest backend/tests/unit backend/tests/integration -q --basetemp $pytestBase
}
Invoke-Checked "AUTH_GUARD" {
    $authBase = Join-Path $pytestBase "auth-guard"
    & $python -m pytest backend/tests/integration/test_auth_api.py::test_protected_training_route_requires_login -q --basetemp $authBase
}
Invoke-Checked "FRONTEND_TESTS" {
    Push-Location (Join-Path $root "frontend")
    try { & $node $vitestEntry run } finally { Pop-Location }
}
Invoke-Checked "FRONTEND_E2E_MOCK" {
    Push-Location (Join-Path $root "frontend")
    try { & $node $playwrightEntry test --project=chromium } finally { Pop-Location }
}

if ($SkipGpu) {
    $report.checks["GPU_RUNTIME"] = "SKIPPED"
    $report.checks["AUTHORIZED_REFERENCE_CONTRACT"] = "SKIPPED"
    $report.checks["GPT_SOVITS_ZERO_SHOT_SMOKE"] = "SKIPPED"
    $report.checks["REAL_QUALITY_EVALUATION"] = "SKIPPED"
    $report.checks["SECURITY_ATTACK_MATRIX"] = "SKIPPED"
    Write-Output "GPU_RUNTIME=SKIPPED"
    # The console result distinguishes the diagnostic mode, while the JSON
    # report remains INCOMPLETE until every real GPU gate has passed.
    $report.final_result = "INCOMPLETE"
}
else {
    Invoke-Checked "AUTHORIZED_REFERENCE_CONTRACT" {
        if ([string]::IsNullOrWhiteSpace($AuthorizedReference)) { throw "TIMBRE_AUTHORIZED_REFERENCE is required for final real-data evaluation" }
        if (-not (Test-Path -LiteralPath $AuthorizedReference -PathType Leaf)) { throw "authorized reference file does not exist" }
        & $python -c "from pathlib import Path; import sys; import soundfile as sf; from backend.app.services.audio_validation import prepare_reference_dataset, validate_reference_duration; source=Path(sys.argv[1]).resolve(); manifest=prepare_reference_dataset('verify-reference', [source], output_dir=Path('tmp/verify-reference')); validate_reference_duration(float(manifest.effective_seconds)); assert len(manifest.rows) == 1, 'reference contains no valid speech'; print(f'reference_effective_seconds={manifest.effective_seconds:.3f}')" $AuthorizedReference
    }
    Invoke-Checked "GPU_RUNTIME" {
        & $python -c "import torch; assert torch.__version__.split('+',1)[0] == '2.5.1'; assert torch.version.cuda == '12.4'; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert torch.cuda.get_device_properties(0).total_memory >= 8000 * 1024**2"
    }
    Invoke-Checked "GPT_SOVITS_ZERO_SHOT_SMOKE" {
        & $python -m pytest backend/tests/gpu/test_gpt_sovits_smoke.py backend/tests/gpu/test_watermark_robustness.py -m gpu -q -rs --basetemp $pytestBase --junitxml tmp/verify-gpu.xml
        if ($LASTEXITCODE -eq 0) {
            $xml = [xml](Get-Content -LiteralPath "tmp/verify-gpu.xml" -Raw)
            $skipped = @($xml.testsuites.testsuite | ForEach-Object { [int]$_.skipped } | Measure-Object -Sum).Sum
            if ([int]$skipped -gt 0) { throw "GPU smoke contains skipped tests" }
        }
    }
    Invoke-Checked "REAL_QUALITY_EVALUATION" {
        & $python -m pytest backend/tests/gpu/test_end_to_end_gpu.py -m gpu -q -rs --basetemp $pytestBase --junitxml tmp/verify-real-e2e.xml
        if ($LASTEXITCODE -ne 0) { throw "real authorized GPU E2E failed" }
    }
    Invoke-Checked "SECURITY_ATTACK_MATRIX" {
        & $python -m pytest backend/tests/gpu/test_watermark_robustness.py -m gpu -q -rs --basetemp $pytestBase --junitxml tmp/verify-watermark.xml
        if ($LASTEXITCODE -ne 0) { throw "watermark attack matrix failed" }
    }
}

if ($null -eq $report.first_failure -and -not $SkipGpu) {
    $report.final_result = "PASS"
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
if ($report.final_result -eq "PASS") {
    Write-Output "FINAL_RESULT=PASS"
    exit 0
}
if ($SkipGpu -and $null -eq $report.first_failure) {
    Write-Output "FINAL_RESULT=CPU_ONLY"
    exit 0
}
exit 1
