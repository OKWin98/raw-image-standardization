param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [switch]$DryRun,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
$Arguments = @(
    '-m', 'imagecal', 'run-dual',
    '--config', (Join-Path $ProjectRoot 'configs\dual_validation.yaml'),
    '--source-root', $SourceRoot,
    '--output-dir', $OutputDir
)
if ($DryRun) { $Arguments += '--dry-run' }
if ($Resume) { $Arguments += '--resume' }
& .\.venv\Scripts\python.exe @Arguments
exit $LASTEXITCODE
