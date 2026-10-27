<#
.SYNOPSIS
    Bring the whole stack up behind HTTPS tunnels, for testing the guided capture on a phone.

.DESCRIPTION
    The guided capture cannot be tested on a LAN address: getUserMedia, DeviceOrientation,
    DeviceMotion and MediaRecorder all require a secure context, and http://192.168.x.x is
    not one. So both the engine and the web app go behind Cloudflare quick tunnels.

    Two tunnels, not one, because an HTTPS page calling an HTTP API is mixed content and the
    browser blocks it silently.

    Quick tunnels get a fresh hostname every run and NEXT_PUBLIC_* is inlined at compile
    time, so the order matters: tunnels first, then write web/.env.local, then start Next.
    Doing it by hand and forgetting the restart is how you end up debugging a bundle that
    still points at the previous tunnel.

    DO NOT JUDGE THE GUIDED CAPTURE'S PERFORMANCE AGAINST THIS SCRIPT. It serves
    `next dev`, and a development build is not a slower production build, it is a different
    program: React runs its development build, `reactStrictMode: true` makes it DOUBLE-INVOKE
    every render function and every state initialiser, and Turbopack ships each module as its
    own un-minified chunk. Measured on the /cek route: 5902 KB of JavaScript to parse in dev
    against 885 KB in production, 6.7x. On a desktop that is invisible. On a mid-range Android
    already decoding a 1080x1920 camera stream and running a hardware H.264 encode, it is the
    difference between a usable capture and an unusable one — and it cost two rounds of
    misdirected optimisation to notice.

    Use ./scripts/tunnel-prod.ps1 for anything you intend to measure or to show anyone.

.EXAMPLE
    ./scripts/tunnel-dev.ps1
#>
param(
    [int]$EnginePort = 8008,
    [int]$WebPort = 3000
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$engine = Join-Path $root 'engine'
$web = Join-Path $root 'web'
$python = Join-Path $engine '.venv\Scripts\python.exe'

foreach ($tool in @('cloudflared', 'npx')) {
    try { Get-Command $tool -ErrorAction Stop | Out-Null }
    catch { Write-Host "Missing '$tool'." -ForegroundColor Red; exit 1 }
}
if (-not (Test-Path $python)) { Write-Host 'Run ./scripts/setup.ps1 first.' -ForegroundColor Red; exit 1 }

$procs = @()
$logs = Join-Path $env:TEMP 'biosphere-tunnels'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Start-Logged($file, $exe, $argList, $workDir) {
    $out = Join-Path $logs $file
    # A leftover child from a crashed run still holds its log open; a timestamped name
    # sidesteps that instead of failing the whole bring-up on a file lock.
    if (Test-Path $out) {
        try { Remove-Item $out -Force -ErrorAction Stop }
        catch { $out = Join-Path $logs ((Get-Date -Format 'HHmmss') + '-' + $file) }
    }

    # `npx` and friends are .cmd shims, not Win32 images, and Start-Process refuses them
    # outright ("%1 is not a valid Win32 application"). Route anything that is not a real
    # .exe through cmd.exe, which is what a shell would have done anyway.
    $isExe = $exe -match '\.exe$' -or (Test-Path $exe)
    if ($isExe) {
        Start-Process -PassThru -NoNewWindow -FilePath $exe -ArgumentList $argList `
            -WorkingDirectory $workDir -RedirectStandardOutput $out -RedirectStandardError "$out.err"
    }
    else {
        $joined = ($argList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
        Start-Process -PassThru -NoNewWindow -FilePath 'cmd.exe' `
            -ArgumentList @('/c', "$exe $joined") `
            -WorkingDirectory $workDir -RedirectStandardOutput $out -RedirectStandardError "$out.err"
    }
}

function Wait-TunnelUrl($file, $label) {
    $out = Join-Path $logs $file
    foreach ($i in 1..60) {
        foreach ($f in @($out, "$out.err")) {
            if (Test-Path $f) {
                $m = Select-String -Path $f -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue |
                     Select-Object -First 1
                if ($m) { return $m.Matches[0].Value }
            }
        }
        Start-Sleep -Milliseconds 900
    }
    Write-Host "Timed out waiting for the $label tunnel." -ForegroundColor Red
    exit 1
}

Write-Host '==> engine' -ForegroundColor Cyan
$env:BIOSPHERE_CORS_ALLOW_ORIGINS = '["*"]'
$procs += Start-Logged 'engine.log' $python @('-m', 'uvicorn', 'biosphere.api.main:app',
    '--host', '0.0.0.0', '--port', "$EnginePort") $engine

Write-Host '==> tunnels' -ForegroundColor Cyan
$procs += Start-Logged 'engine-tunnel.log' 'cloudflared' @('tunnel', '--url', "http://127.0.0.1:$EnginePort", '--no-autoupdate') $root
$procs += Start-Logged 'web-tunnel.log' 'cloudflared' @('tunnel', '--url', "http://127.0.0.1:$WebPort", '--no-autoupdate') $root

$engineUrl = Wait-TunnelUrl 'engine-tunnel.log' 'engine'
$webUrl = Wait-TunnelUrl 'web-tunnel.log' 'web'

@"
# Written by scripts/tunnel-dev.ps1 — quick tunnels change hostname on every run.
NEXT_PUBLIC_API_BASE=$engineUrl
NEXT_PUBLIC_SITE_URL=$webUrl
"@ | Set-Content -Path (Join-Path $web '.env.local') -Encoding utf8

Write-Host '==> web (after the env, so the bundle inlines the right host)' -ForegroundColor Cyan
$procs += Start-Logged 'web.log' 'npx' @('next', 'dev', '-H', '0.0.0.0', '-p', "$WebPort") $web

Write-Host ''
Write-Host "  Guided capture : $webUrl/cek" -ForegroundColor Green
Write-Host "  Device probe   : $webUrl/tani" -ForegroundColor Green
Write-Host "  Dashboard      : $webUrl/properties" -ForegroundColor Green
Write-Host "  Engine         : $engineUrl/api/v1/health" -ForegroundColor DarkGray
Write-Host ''
Write-Host '  These URLs are public while this runs. Ctrl+C stops everything.' -ForegroundColor Yellow

try {
    while ($true) {
        foreach ($p in $procs) { if ($p.HasExited) { throw "a child exited ($($p.Id))" } }
        Start-Sleep -Milliseconds 800
    }
}
finally {
    foreach ($p in $procs) {
        if (-not $p.HasExited) { try { Stop-Process -Id $p.Id -Force -Confirm:$false } catch {} }
    }
    Write-Host 'stopped.' -ForegroundColor DarkGray
}
