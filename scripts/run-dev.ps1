[CmdletBinding()]
param(
    [switch]$SkipWorker
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

# This script must keep running under the documented Windows PowerShell 5.1 entry
# (`powershell -ExecutionPolicy Bypass -File scripts/run-dev.ps1`) as well as pwsh:
# no ProcessStartInfo.ArgumentList (missing on .NET Framework), no scriptblock
# process event handlers (no runspace on their threadpool threads), and no
# Process.Kill($true) tree kill (also PowerShell 7 only).
$python = "D:\miniconda3\envs\timbre-share\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "固定 timbre-share Python 解释器不存在: $python"
}
$node = (Get-Command node.exe -ErrorAction SilentlyContinue).Source
if ([string]::IsNullOrWhiteSpace($node)) {
    throw "node.exe was not found; install Node.js before starting the local UI"
}
$viteEntry = Join-Path $root "frontend/node_modules/vite/bin/vite.js"
if (-not (Test-Path -LiteralPath $viteEntry -PathType Leaf)) {
    throw "Vite 依赖未安装，请先在 frontend 安装依赖: $viteEntry"
}

$logRoot = Join-Path $root "data/logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$processes = [System.Collections.Generic.List[object]]::new()
$pidFile = Join-Path $logRoot "dev-pids.json"

function Normalize-CommandLine {
    param(
        [string]$Value
    )
    if ($null -eq $Value) { return "" }
    return (($Value -replace '"', '') -replace '/', '\')
}

function ConvertTo-ProcessArgument {
    param(
        [string]$Value
    )
    if ($null -eq $Value) { return '""' }
    if ($Value -notmatch '[\s"]') { return $Value }
    # Start-Process on Windows PowerShell 5.1 joins ArgumentList values into a
    # command line. Quote values with spaces before handing that line over so a
    # repository path such as `D:\Voice Projects\jianxi` remains one argument.
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Test-ExpectedProcess {
    param(
        [object]$ProcessInfo,
        [string]$ExecutablePath,
        [string]$CommandLineMarker
    )
    if ($null -eq $ProcessInfo -or [string]::IsNullOrWhiteSpace($ExecutablePath)) {
        return $false
    }
    $actualExecutable = [string]$ProcessInfo.ExecutablePath
    if (-not [StringComparer]::OrdinalIgnoreCase.Equals($actualExecutable, $ExecutablePath)) {
        return $false
    }
    $actualCommandLine = Normalize-CommandLine -Value ([string]$ProcessInfo.CommandLine)
    $marker = Normalize-CommandLine -Value $CommandLineMarker
    return $actualCommandLine.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0
}

function Start-LocalProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory
    )
    # Start-Process writes child stdout/stderr straight to files, so no event
    # handlers are needed. Windows PowerShell 5.1 joins ArgumentList values
    # without quoting; pass one explicitly quoted command line instead.
    $commandLine = ($ArgumentList | ForEach-Object {
        ConvertTo-ProcessArgument -Value ([string]$_)
    }) -join " "
    $process = Start-Process -FilePath $FilePath `
        -ArgumentList $commandLine `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot "$Name.out.log") `
        -RedirectStandardError (Join-Path $logRoot "$Name.err.log") `
        -PassThru
    $processes.Add([pscustomobject]@{
        Name = $Name
        Process = $process
        ExecutablePath = $FilePath
        CommandLineMarker = ($ArgumentList -join " ")
    })
    return $process
}

function Stop-LocalProcess {
    param(
        [object]$Item
    )
    if ($null -ne $Item.Process -and -not $Item.Process.HasExited) {
        # taskkill /T kills the whole child tree; the
        # Process.Kill($true) tree overload is not available on PowerShell 5.1.
        & taskkill.exe /PID $Item.Process.Id /T /F | Out-Null
        [void]$Item.Process.WaitForExit(5000)
    }
    if ($null -ne $Item.Process) {
        $Item.Process.Dispose()
    }
}

function Stop-ProcessTree {
    param(
        [int]$Id
    )
    & taskkill.exe /PID $Id /T /F | Out-Null
}

function Get-PortListenerPid {
    param(
        [int]$Port
    )
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($listener) { return [int]$listener.OwningProcess }
    return $null
}

function Get-ProcessInfo {
    param(
        [int]$Id
    )
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$Id" -ErrorAction SilentlyContinue
    return $proc
}

# Stale-instance recovery: when a previous run of this script is killed
# abruptly (e.g. its terminal window is closed), the finally block below never
# runs, so its backend/GPU worker/vite children stay alive holding ports
# 8000/5173 and the SQLite database. A fresh uvicorn would then die with
# WSAEADDRINUSE, surfacing only as "backend exited with code 1". Stop the
# leftovers before binding.

# 1) Processes recorded by the previous unclean run; dev-pids.json only
#    survives when finally did not run. Require the exact executable path and
#    command marker recorded by this script so a recycled PID is never killed.
if (Test-Path -LiteralPath $pidFile) {
    foreach ($entry in (Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json)) {
        $stale = Get-ProcessInfo -Id ([int]$entry.pid)
        if (Test-ExpectedProcess -ProcessInfo $stale `
            -ExecutablePath ([string]$entry.executable_path) `
            -CommandLineMarker ([string]$entry.command_line_marker)) {
            Write-Output "stale $($entry.name) pid $($entry.pid) left from a previous run; stopping it"
            Stop-ProcessTree -Id $entry.pid
        }
    }
    Remove-Item -LiteralPath $pidFile -Force
}

# 2) A GPU worker of this repo left from an earlier session would fight a new
#    worker over cuda:0 and the SQLite database (single worker by design), so
#    stop it as well. Match python processes whose command line runs this
#    repo's worker script; anything else is left alone.
 $workerScript = (Resolve-Path -LiteralPath (Join-Path $root "scripts/worker_process.py")).Path
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
        Test-ExpectedProcess -ProcessInfo $_ -ExecutablePath $python -CommandLineMarker $workerScript
    } |
    ForEach-Object {
        Write-Output "stale gpu-worker pid $($_.ProcessId) left from a previous run; stopping it"
        Stop-ProcessTree -Id $_.ProcessId
    }

# 3) Ports 8000/5173 must be free before uvicorn/vite start. A leftover
#    listener of this dev stack that escaped steps 1-2 (started outside this
#    script) is stopped too; any other listener is a foreign process we must
#    not kill, so fail fast with its PID instead of a cryptic bind error.
foreach ($port in 8000, 5173) {
    $listenerPid = Get-PortListenerPid -Port $port
    if ($null -eq $listenerPid) { continue }
    $listenerInfo = Get-ProcessInfo -Id $listenerPid
    $isOwnedBackend = Test-ExpectedProcess `
        -ProcessInfo $listenerInfo -ExecutablePath $python -CommandLineMarker "backend.app.main:app"
    $isOwnedFrontend = Test-ExpectedProcess `
        -ProcessInfo $listenerInfo -ExecutablePath $node -CommandLineMarker $viteEntry
    if ($isOwnedBackend -or $isOwnedFrontend) {
        Write-Output "stale listener on port $port pid $listenerPid left from a previous run; stopping it"
        Stop-ProcessTree -Id $listenerPid
    } else {
        $owner = Get-Process -Id $listenerPid -ErrorAction SilentlyContinue
        $ownerName = if ($owner) { $owner.ProcessName } else { "unknown" }
        throw "port $port is already used by PID $listenerPid ($ownerName), which is not a leftover of this dev stack; stop it and rerun"
    }
}

try {
    [void](Start-LocalProcess -Name "backend" -FilePath $python -ArgumentList @(
        "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"
    ) -WorkingDirectory $root)
    if (-not $SkipWorker) {
        [void](Start-LocalProcess -Name "gpu-worker" -FilePath $python -ArgumentList @(
            $workerScript
        ) -WorkingDirectory $root)
    }
    [void](Start-LocalProcess -Name "frontend" -FilePath $node -ArgumentList @(
        $viteEntry, "--host", "127.0.0.1"
    ) -WorkingDirectory (Join-Path $root "frontend"))
    $state = $processes | ForEach-Object {
        [pscustomobject]@{
            name = $_.Name
            pid = $_.Process.Id
            executable_path = $_.ExecutablePath
            command_line_marker = $_.CommandLineMarker
        }
    }
    # WriteAllText always writes UTF-8 without BOM; Set-Content -Encoding UTF8
    # adds a BOM on Windows PowerShell 5.1 and breaks strict JSON readers.
    [System.IO.File]::WriteAllText($pidFile, ($state | ConvertTo-Json))
    Write-Output "BACKEND=http://127.0.0.1:8000"
    Write-Output "FRONTEND=http://127.0.0.1:5173"
    Write-Output "PIDS=$pidFile"
    while ($true) {
        foreach ($item in $processes) {
            if ($item.Process.HasExited) {
                throw "$($item.Name) exited with code $($item.Process.ExitCode)"
            }
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    foreach ($item in $processes) {
        Stop-LocalProcess -Item $item
    }
    if (Test-Path -LiteralPath $pidFile) {
        Remove-Item -LiteralPath $pidFile -Force
    }
}
