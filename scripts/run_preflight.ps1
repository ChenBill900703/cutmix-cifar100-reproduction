$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
$Python = if (Test-Path "$Project\.venv\Scripts\python.exe") { "$Project\.venv\Scripts\python.exe" } else { "E:\Random Erasing Data Augmentation\.venv\Scripts\python.exe" }
$env:PYTHONPATH = "$Project\src"
& $Python -m cutmix_repro.preflight --output "$Project\outputs\preflight\report.json" --steps 5

