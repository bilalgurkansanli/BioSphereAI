@echo off
rem Starts BioSphere for development on Windows, each server in its own window.
rem
rem Double-click it, or run it from any shell. Unlike scripts/dev.ps1 it needs no PowerShell
rem execution policy, and each window stays open on its own: close a window (or press Ctrl+C
rem in it) to stop that server. Run scripts/setup.ps1 once first.
rem
rem   engine  http://127.0.0.1:8008   (API docs at /docs)
rem   web     http://localhost:3000
rem   phone   https://<this machine's LAN IP>:3443   (only if scripts\certs\cert.pem exists)

setlocal
set "ROOT=%~dp0.."

if not exist "%ROOT%\engine\.venv\Scripts\python.exe" (
  echo The engine environment is missing. Run scripts\setup.ps1 first.
  pause
  exit /b 1
)

rem Each command runs from its own folder (/D) with a RELATIVE path, so nothing needs a quote
rem inside a quote: cmd /k mangles nested quotes, and an install path with a space in it
rem would otherwise break the engine window without a word.
start "BioSphere engine :8008" /D "%ROOT%\engine" cmd /k .venv\Scripts\python.exe -m uvicorn biosphere.api.main:app --host 127.0.0.1 --port 8008
start "BioSphere web :3000" /D "%ROOT%\web" cmd /k npm.cmd run dev

if exist "%ROOT%\scripts\certs\cert.pem" (
  start "BioSphere phone HTTPS :3443" /D "%ROOT%" cmd /k node scripts\lan-proxy.mjs
) else (
  echo No LAN certificate in scripts\certs - skipping the phone proxy. See scripts\certs\san.cnf.
)

endlocal
