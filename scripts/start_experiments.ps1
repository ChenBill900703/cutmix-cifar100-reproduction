param([string[]]$Seeds = @('0'))
$ErrorActionPreference = 'Stop'
$Seeds = @($Seeds | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne '' } | ForEach-Object { [int]$_ })
$Project = Split-Path -Parent $PSScriptRoot
$Stdout = "$Project\outputs\experiments\batch_stdout.log"
$Stderr = "$Project\outputs\experiments\batch_stderr.log"
$SeedArguments = $Seeds -join ','
$ArgumentList = "-NoProfile -ExecutionPolicy Bypass -File `"$Project\scripts\run_experiments.ps1`" -Seeds $SeedArguments"
$Process = Start-Process -FilePath 'powershell.exe' -ArgumentList $ArgumentList -WorkingDirectory $Project -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
$ProcessJson = @{ pid = $Process.Id; launched_at = (Get-Date).ToString('o'); seeds = $Seeds; stdout = $Stdout; stderr = $Stderr } | ConvertTo-Json
[System.IO.File]::WriteAllText("$Project\outputs\experiments\batch_process.json", $ProcessJson, [System.Text.UTF8Encoding]::new($false))
Write-Output "Started CutMix experiment batch PID $($Process.Id)"
