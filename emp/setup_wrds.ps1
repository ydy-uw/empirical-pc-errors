$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$setupScript = Join-Path $projectRoot "configure_wrds.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found: $python"
}

& $python $setupScript
