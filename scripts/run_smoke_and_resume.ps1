$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
$Python = if (Test-Path "$Project\.venv\Scripts\python.exe") { "$Project\.venv\Scripts\python.exe" } else { "E:\Random Erasing Data Augmentation\.venv\Scripts\python.exe" }
$env:PYTHONPATH = "$Project\src"
$Out = "$Project\outputs\smoke\cutmix_resume"
& $Python -m cutmix_repro.train --config "$Project\configs\smoke.toml" --output $Out --train-samples 1024 --val-samples 256 --stop-after-epoch 1
& $Python -m cutmix_repro.train --config "$Project\configs\smoke.toml" --output $Out --train-samples 1024 --val-samples 256 --resume auto

