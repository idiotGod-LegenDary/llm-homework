param(
    [ValidateSet('all','prepare','train','analyze')][string]$Stage='all',
    [ValidateRange(1,10)][int]$Repeats=3,
    [string]$Source='',
    [string]$KenlmBin=''
)
$ErrorActionPreference='Stop'
$venvPython=Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) { $taskPython=$venvPython }
else { $taskPython=(Get-Command python -ErrorAction Stop).Source }
$arguments=@('-X','utf8',(Join-Path $PSScriptRoot 'run_experiment.py'),'--stage',$Stage,'--repeats',$Repeats)
if ($Source) { $arguments+=@('--source',$Source) }
if ($KenlmBin) { $arguments+=@('--kenlm-bin',$KenlmBin) }
& $taskPython @arguments
if ($LASTEXITCODE -ne 0) { throw "Experiment exited with code $LASTEXITCODE" }
