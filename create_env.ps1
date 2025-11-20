<#
PowerShell helper to create a virtual environment and install dependencies from requirements.txt.
Usage (PowerShell):
  .\create_env.ps1 -EnvName .venv
#>
param(
    [string]$EnvName = ".venv"
)

Write-Host "Creating virtual environment: $EnvName"
python -m venv $EnvName

if (-Not (Test-Path "$EnvName/Scripts/Activate.ps1")) {
    Write-Error "Virtual environment creation failed. Ensure python is on PATH and try again."
    exit 1
}

Write-Host "Activating virtual environment and installing requirements..."
& "$EnvName/Scripts/Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Environment created and dependencies installed. Activate with: .\$EnvName\Scripts\Activate.ps1"