$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python environment not found at: $pythonExe"
    exit 1
}

& $pythonExe (Join-Path $repoRoot "src\predict.py") @args
