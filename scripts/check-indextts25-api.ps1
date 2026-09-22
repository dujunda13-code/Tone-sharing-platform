#requires -Version 7.0
param()

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$baseUrl = 'https://cp.compshare.cn'
$secret = $null
$credential = $null
$apiKey = $null
$headers = $null

Write-Host 'IndexTTS 2.5 API check (read-only, no credits consumed)'
Write-Host 'Paste the audio/video studio sk-ml- key at the hidden prompt.'
Write-Host 'Press Enter without typing to use COMPSHARE_API_KEY.'

try {
    $secret = Read-Host 'API Key' -AsSecureString
    $credential = [System.Net.NetworkCredential]::new('', $secret)
    $apiKey = $credential.Password.Trim()
    if (-not $apiKey) { $apiKey = $env:COMPSHARE_API_KEY }
    if (-not $apiKey -or -not $apiKey.StartsWith('sk-ml-')) {
        throw 'Enter the original sk-ml- key from the audio/video studio. Do not add a prefix to a different key.'
    }
    $headers = @{ Authorization = "Bearer $apiKey"; Accept = 'application/json' }
    $checks = @(
        @{ Name = 'Shared authentication'; Path = '/minimax/v2/query/point_packages' },
        @{ Name = 'Audio pricing'; Path = '/audio/v1/pricing' },
        @{ Name = 'Preset voices'; Path = '/audio/v1/voices?kind=preset' }
    )
    $results = foreach ($check in $checks) {
        Write-Host "`n$($check.Name): GET $($check.Path)"
        $watch = [Diagnostics.Stopwatch]::StartNew()
        try {
            $response = Invoke-WebRequest -Uri ($baseUrl + $check.Path) -Method Get `
                -Headers $headers -TimeoutSec 30 -MaximumRedirection 0 -SkipHttpErrorCheck
            $status = [int]$response.StatusCode
            Write-Host "HTTP $status ($([math]::Round($watch.Elapsed.TotalSeconds, 2)) seconds)"
            $body = $response.Content.Replace($apiKey, '[REDACTED]')
            $json = $null
            try { $json = $body | ConvertFrom-Json } catch {}
            $ok = $status -ge 200 -and $status -lt 300 -and $null -ne $json
            if ($json.error -or ($null -ne $json.RetCode -and $json.RetCode -ne 0)) { $ok = $false }
            if (-not $ok) {
                Write-Host ($body.Substring(0, [Math]::Min(1200, $body.Length)))
            } elseif ($check.Name -eq 'Audio pricing') {
                Write-Host "enabled=$($json.enabled); available_points=$($json.balance.available_points)"
                Write-Host "unit_characters=$($json.unit_characters); cost=$($json.costs.text_to_speech)"
            } elseif ($check.Name -eq 'Preset voices') {
                Write-Host "total=$($json.total)"
                $json.items | Select-Object voice_id, name, language | Format-Table | Out-Host
            } else {
                Write-Host 'JSON response received. Check audio endpoints below.'
            }
            [pscustomobject]@{ Name = $check.Name; Status = $status; OK = $ok; Data = $json }
        } catch {
            Write-Host ('Request failed: ' + $_.Exception.Message.Replace($apiKey, '[REDACTED]'))
            [pscustomobject]@{ Name = $check.Name; Status = 0; OK = $false; Data = $null }
        }
    }
    Write-Host "`n--- Result ---"
    $results | Select-Object Name, Status, OK | Format-Table | Out-Host
    $pricing = $results[1]
    if ($pricing.OK -and $pricing.Data.enabled -eq $true -and $results[2].OK) {
        Write-Host 'Audio API is reachable and enabled. Actual synthesis has NOT been tested.' -ForegroundColor Green
    } elseif ($pricing.Status -eq 404) {
        Write-Host 'Documented audio endpoint returned 404. This alone does not prove the reason or account permissions.' -ForegroundColor Yellow
    } else {
        Write-Host 'Audio API availability is not confirmed. See the HTTP statuses and response above.' -ForegroundColor Yellow
    }
} finally {
    if ($headers) { $headers.Clear() }
    $apiKey = $null
    $credential = $null
    if ($secret) { $secret.Dispose() }
}
