<#
.SYNOPSIS
    One-time setup: Python virtual environment, engine dependencies, web dependencies.

.DESCRIPTION
    Requires ffmpeg, Node 20+, and uv on PATH. Pass -Gpu to also install the CUDA build
    of PyTorch, which studio mode uses for RAFT optical-flow parallax repair. Without it
    the engine falls back to OpenCV's DIS optical flow on the CPU — slower, still works.
#>
param(
    [switch]$Gpu,
    [string]$MediaRoot
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$engine = Join-Path $root 'engine'
$web = Join-Path $root 'web'

function Require($name, $hint) {
    try { Get-Command $name -ErrorAction Stop | Out-Null }
    catch { Write-Host "Missing '$name'. $hint" -ForegroundColor Red; exit 1 }
}

Require 'uv'     'Install with: winget install astral-sh.uv'
Require 'node'   'Install Node 20 or newer from nodejs.org'
Require 'ffmpeg' 'Install with: winget install Gyan.FFmpeg'

Write-Host '==> engine virtualenv' -ForegroundColor Cyan
Set-Location $engine
if (-not (Test-Path '.venv')) { uv venv --python 3.12 .venv }
$env:VIRTUAL_ENV = Join-Path $engine '.venv'
uv pip install -e '.[dev]'

if ($Gpu) {
    # CUDA 12.8, NOT 12.6. A cu126 wheel carries no kernels for sm_120, so on a Blackwell card
    # (RTX 50-series) every CUDA call fails at runtime with "no kernel image is available" —
    # torch still imports and `cuda.is_available()` still says True, which is what makes it
    # worth a comment rather than a version bump. cu128 covers Blackwell and everything older
    # this project runs on.
    Write-Host '==> PyTorch (CUDA 12.8)' -ForegroundColor Cyan
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

    # The room-emptying models. Optional even with a GPU: without them /ai/empty-room falls
    # back to the classical filler and reports `model_backed: false`. See the licence note in
    # pyproject.toml -- these weights are not all licensed the same way.
    Write-Host '==> room-emptying models (segmentation + diffusion inpainting)' -ForegroundColor Cyan
    uv pip install -e '.[ai]'
}

Write-Host '==> web dependencies' -ForegroundColor Cyan
Set-Location $web
npm install

if ($MediaRoot) {
    Write-Host "==> media root: $MediaRoot" -ForegroundColor Cyan
    foreach ($d in @('uploads', 'properties', 'work', 'cache', 'db')) {
        New-Item -ItemType Directory -Force -Path (Join-Path $MediaRoot $d) | Out-Null
    }
    Set-Content -Path (Join-Path $engine '.env') -Encoding utf8 `
        -Value "BIOSPHERE_MEDIA_ROOT=$MediaRoot"
}

Write-Host '==> checking the install' -ForegroundColor Cyan
Set-Location $engine
& '.\.venv\Scripts\python.exe' -m biosphere.cli doctor

Write-Host "`nDone. Start everything with ./scripts/dev.ps1" -ForegroundColor Green
