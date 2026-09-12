$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
$BundledPython = 'C:\Users\ChenBill\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python -m venv "$Project\.venv"
} elseif (Test-Path $BundledPython) {
    & $BundledPython -m venv "$Project\.venv"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 -m venv "$Project\.venv"
} else {
    throw 'Python 3.10 or newer is required. Install Python and rerun this script.'
}
& "$Project\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Project\.venv\Scripts\python.exe" -m pip install -r "$Project\requirements.txt"
