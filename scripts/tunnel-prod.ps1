<#
.SYNOPSIS
    Bring the stack up behind HTTPS tunnels as a PRODUCTION build, for testing on a phone.

.DESCRIPTION
    This exists because `tunnel-dev.ps1` was serving the guided capture from `next dev`, and
    a development build is not a slower version of the app — it is a different one:

      * React runs its development build, which carries the dev warnings, the component
        stacks and the `_owner` bookkeeping on every single render.
      * `reactStrictMode: true` makes React DOUBLE-INVOKE every render function and every
        `useState`/`useMemo` initialiser in development. `CaptureStage` is 2800 lines; it was
        running its whole body twice per render, and mounting every effect twice.
      * Turbopack serves each module as its own un-minified chunk — hundreds of requests and
        hundreds of separate scripts to parse, with none of the code locality a bundle has.
      * The HMR client holds a websocket open and reacts to it, through the tunnel.

    On a desktop none of that is noticeable. On a mid-range Android that is simultaneously
    decoding a 1080x1920 camera stream and running a hardware H.264 encode at 12 Mbps, it is
    the difference between a usable capture and an unusable one. THE CAPTURE MUST BE TESTED
    AGAINST THIS SCRIPT, not against the dev one.

    Order matters and is different from the dev script. `NEXT_PUBLIC_*` is inlined at BUILD
    time, and quick tunnels get a fresh hostname on every run, so the tunnels have to exist
    before the build starts: engine -> tunnels -> .env.local -> build -> start. Building
    first and hoping the hostname matches is how you end up serving a bundle that points at
    the previous run's tunnel.

.EXAMPLE
    ./scripts/tunnel-prod.ps1

.EXAMPLE
    ./scripts/tunnel-prod.ps1 -SkipBuild   # reuse .next from the last run, same tunnels only
#>
param(
    [int]$EnginePort = 8008,
    [int]$WebPort = 3000,
    [switch]$SkipBuild
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
# Written by scripts/tunnel-prod.ps1 — quick tunnels change hostname on every run, and
# NEXT_PUBLIC_* is inlined at BUILD time, so the build below has to come after this file.
NEXT_PUBLIC_API_BASE=$engineUrl
NEXT_PUBLIC_SITE_URL=$webUrl
"@ | Set-Content -Path (Join-Path $web '.env.local') -Encoding utf8

if (-not $SkipBuild) {
    Write-Host '==> building (this is the point of this script; it takes a minute)' -ForegroundColor Cyan
    $build = Start-Process -PassThru -NoNewWindow -Wait -FilePath 'cmd.exe' `
        -ArgumentList @('/c', 'npx next build') -WorkingDirectory $web
    if ($build.ExitCode -ne 0) {
        Write-Host 'next build failed.' -ForegroundColor Red
        foreach ($p in $procs) { if (-not $p.HasExited) { try { Stop-Process -Id $p.Id -Force -Confirm:$false } catch {} } }
        exit 1
    }
}

Write-Host '==> web (production server)' -ForegroundColor Cyan
$procs += Start-Logged 'web.log' 'npx' @('next', 'start', '-H', '0.0.0.0', '-p', "$WebPort") $web

Write-Host ''
Write-Host "  Guided capture : $webUrl/cek" -ForegroundColor Green
Write-Host "  Device probe   : $webUrl/tani" -ForegroundColor Green
Write-Host "  Dashboard      : $webUrl/properties" -ForegroundColor Green
Write-Host "  Engine         : $engineUrl/api/v1/health" -ForegroundColor DarkGray
Write-Host ''
Write-Host '  PRODUCTION build. This is the one to judge the capture on.' -ForegroundColor Yellow
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
