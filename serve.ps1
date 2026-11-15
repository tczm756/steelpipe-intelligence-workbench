$ErrorActionPreference = "Stop"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python 3 is required to serve the workbench locally."
}

Set-Location $PSScriptRoot
python -m http.server 4173 --bind 127.0.0.1
