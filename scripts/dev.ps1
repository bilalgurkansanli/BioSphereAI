<#
.SYNOPSIS
    Starts the BioSphere engine and the web app together for local development.

.EXAMPLE
    ./scripts/dev.ps1
    ./scripts/dev.ps1 -EngineOnly
    ./scripts/dev.ps1 -WebOnly
#>
param(
    [switch]$EngineOnly,
    [switch]$WebOnly,
    [int]$EnginePort = 8008,
    [int]$WebPort = 3000
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root 'engine\.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host "Python venv missing. Run ./scripts/setup.ps1 first." -ForegroundColor Red
    exit 1
}

$jobs = @()

if (-not $WebOnly) {
    Write-Host "engine  -> http://127.0.0.1:$EnginePort" -ForegroundColor Cyan
    $jobs += Start-Process -PassThru -NoNewWindow -FilePath $python `
        -ArgumentList @('-m', 'uvicorn', 'biosphere.api.main:app', '--reload',
                        '--host', '127.0.0.1', '--port', "$EnginePort") `
        -WorkingDirectory (Join-Path $root 'engine')
}

if (-not $EngineOnly) {
    Write-Host "web     -> http://localhost:$WebPort" -ForegroundColor Cyan
    # npm.cmd, NOT whatever `Get-Command npm` resolves to. On a Node install that ships
    # npm.ps1 -- which the current LTS does -- PowerShell prefers the .ps1, and Start-Process
    # cannot launch a script: it fails with "not a valid Win32 application" and takes the web
    # half of dev down while the engine keeps running, so the panel is simply absent with no
    # obvious reason. The .cmd shim sits beside the .ps1 in every Windows Node install.
    $npmCommand = Get-Command 'npm.cmd' -ErrorAction SilentlyContinue
    if (-not $npmCommand) { $npmCommand = Get-Command 'npm' -ErrorAction SilentlyContinue }
    if (-not $npmCommand) {
        Write-Host "Missing 'npm'. Install Node 20 or newer from nodejs.org." -ForegroundColor Red
        exit 1
    }
    $npm = $npmCommand.Source
    $jobs += Start-Process -PassThru -NoNewWindow -FilePath $npm `
        -ArgumentList @('run', 'dev', '--', '--port', "$WebPort") `
        -WorkingDirectory (Join-Path $root 'web')
}

Write-Host "`nCtrl+C to stop both.`n" -ForegroundColor DarkGray

try {
    while ($true) {
        foreach ($j in $jobs) {
            if ($j.HasExited) {
                Write-Host "Process $($j.Id) exited with $($j.ExitCode)." -ForegroundColor Yellow
                throw 'child exited'
            }
        }
        Start-Sleep -Milliseconds 700
    }
}
finally {
    foreach ($j in $jobs) {
        if (-not $j.HasExited) {
            try { Stop-Process -Id $j.Id -Force -Confirm:$false } catch {}
        }
    }
    Write-Host 'stopped.' -ForegroundColor DarkGray
}
