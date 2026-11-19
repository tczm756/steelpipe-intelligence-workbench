$ErrorActionPreference = "Stop"

$venvPython = Join-Path (Split-Path $PSScriptRoot -Parent) ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
} else {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Python 3 is required to serve the workbench locally."
    }
    $python = $python.Source
}

Set-Location $PSScriptRoot
& $python -m http.server 4173 --bind 127.0.0.1
