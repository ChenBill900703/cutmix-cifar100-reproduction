$Project = Split-Path -Parent $PSScriptRoot
Get-Content "$Project\outputs\experiments\batch_status.json" -ErrorAction SilentlyContinue
Get-Content "$Project\outputs\experiments\batch_process.json" -ErrorAction SilentlyContinue
Get-Content "$Project\outputs\experiments\batch_stdout.log" -Tail 10 -ErrorAction SilentlyContinue
Get-Content "$Project\outputs\experiments\batch_stderr.log" -Tail 20 -ErrorAction SilentlyContinue

