<#
.SYNOPSIS
    Runs the BioSphere AI worker — the segmentation and inpainting service — on this machine.

.DESCRIPTION
    Start this on the computer with the good GPU, then point the engine at it:

        BIOSPHERE_AI_WORKER_URL=http://<this machine's LAN address>:8010

    Nothing else moves. The media library, the database, the job queue and the web app stay
    on the engine's machine; only two model calls travel, one image at a time.

    The worker needs the model weights locally and does not download them at runtime unless
    BIOSPHERE_AI_ALLOW_DOWNLOADS says otherwise:
      facebook/mask2former-swin-large-ade-semantic   in the Hugging Face cache
      stabilityai/stable-diffusion-2-inpainting      in the Hugging Face cache, or as a
                                                     folder under -ModelsDir
      big-lama.pt                                    in the torch hub cache, or -ModelsDir

    Both inpainters have a job and neither replaces the other: the diffusion model fills the
    perspective windows and big-lama fills the equirect canvas afterwards. A machine with only
    one of them still works; see biosphere/ai/backend.py.

.EXAMPLE
    ./scripts/ai-worker.ps1
    ./scripts/ai-worker.ps1 -Port 8010 -ModelsDir 'D:\biosphere-models'
#>
param(
    [int]$Port = 8010,
    [string]$ModelsDir = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root 'engine\.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host "Python venv missing. Run ./scripts/setup.ps1 first." -ForegroundColor Red
    exit 1
}

if ($ModelsDir) { $env:BIOSPHERE_AI_MODELS_DIR = $ModelsDir }
# This process IS the GPU, so it always uses its own.
$env:BIOSPHERE_AI_LOCAL_MODELS = 'true'
$env:BIOSPHERE_AI_WORKER_URL = ''

# 0.0.0.0 so the engine on the other machine can reach it. LAN only — the worker has no
# authentication, by design; see the module docstring.
$addresses = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
    Select-Object -ExpandProperty IPAddress) -join ', '

Write-Host ''
Write-Host "  BioSphere AI worker on port $Port" -ForegroundColor Cyan
Write-Host "  Reachable at: $addresses" -ForegroundColor DarkGray
Write-Host "  Set on the engine machine:" -ForegroundColor DarkGray
Write-Host "    BIOSPHERE_AI_WORKER_URL=http://<one of those>:$Port" -ForegroundColor Green
Write-Host ''

& $python -m uvicorn biosphere.ai.worker:app --host 0.0.0.0 --port $Port --app-dir (Join-Path $root 'engine')
