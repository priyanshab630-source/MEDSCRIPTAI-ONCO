# Start the MEDSCRIPTAI-ONCO dashboard on Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 8080
param([int]$Port = 8000)

Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) {
    Write-Host "==> creating .venv"
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
Write-Host "==> installing dependencies (first run takes a couple of minutes)"
& ".\.venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt

Write-Host "==> starting the dashboard on http://127.0.0.1:$Port"
& ".\.venv\Scripts\python.exe" -m uvicorn api.app:app --reload --port $Port
