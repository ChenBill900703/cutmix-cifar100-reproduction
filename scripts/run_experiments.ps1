param([string[]]$Seeds = @('0'))
$ErrorActionPreference = 'Stop'
$Seeds = @($Seeds | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne '' } | ForEach-Object { [int]$_ })
$Project = Split-Path -Parent $PSScriptRoot
$Python = if (Test-Path "$Project\.venv\Scripts\python.exe") { "$Project\.venv\Scripts\python.exe" } else { "E:\Random Erasing Data Augmentation\.venv\Scripts\python.exe" }
$env:PYTHONPATH = "$Project\src"
$StatusPath = "$Project\outputs\experiments\batch_status.json"
$Started = Get-Date
function Write-Utf8NoBom([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}
Write-Utf8NoBom $StatusPath (@{ status = 'running'; started_at = $Started.ToString('o'); seeds = $Seeds } | ConvertTo-Json)
try {
    $Sequence = @()
    if ($Seeds -contains 0) {
        $Sequence += @(@{ Method = 'baseline'; Seed = 0 }, @{ Method = 'cutmix'; Seed = 0 })
    }
    foreach ($Seed in ($Seeds | Where-Object { $_ -ne 0 })) {
        $Sequence += @{ Method = 'cutmix'; Seed = $Seed }
    }
    foreach ($Seed in ($Seeds | Where-Object { $_ -ne 0 })) {
        $Sequence += @{ Method = 'baseline'; Seed = $Seed }
    }
    foreach ($Run in $Sequence) {
        $Seed = $Run.Seed
        $Method = $Run.Method
        $BaseConfig = Get-Content "$Project\configs\$Method.toml" -Raw
        $TempConfig = "$Project\outputs\experiments\_${Method}_seed${Seed}.toml"
        Write-Utf8NoBom $TempConfig ($BaseConfig -replace 'seed = 0', "seed = $Seed")
        $Out = "$Project\outputs\experiments\pyramidnet200_${Method}_seed${Seed}"
        $ResumeArgs = if (Test-Path "$Out\last.pt") { @('--resume', 'auto') } else { @() }
        Write-Utf8NoBom $StatusPath (@{ status = 'running'; started_at = $Started.ToString('o'); current_method = $Method; current_seed = $Seed; output = $Out } | ConvertTo-Json)
        & $Python -m cutmix_repro.train --config $TempConfig --output $Out @ResumeArgs
        if ($LASTEXITCODE -ne 0) { throw "$Method seed $Seed failed" }
    }
    & $Python -m cutmix_repro.report --experiments-root "$Project\outputs\experiments" --report "$Project\reproduction_report.md"
    Write-Utf8NoBom $StatusPath (@{ status = 'complete'; started_at = $Started.ToString('o'); completed_at = (Get-Date).ToString('o'); seeds = $Seeds } | ConvertTo-Json)
} catch {
    Write-Utf8NoBom $StatusPath (@{ status = 'failed'; started_at = $Started.ToString('o'); failed_at = (Get-Date).ToString('o'); error = $_.Exception.Message } | ConvertTo-Json)
    throw
}
