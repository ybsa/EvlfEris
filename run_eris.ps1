# Quick launcher for Evlf Eris (PowerShell)

Write-Host ""
Write-Host "===================================" -ForegroundColor Cyan
Write-Host "   Evlf Eris - Model Surgery" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Install dependencies if needed
if (-not (Test-Path ".venv\Lib\site-packages\torch")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Run chat
python inference\chat.py $args

deactivate
