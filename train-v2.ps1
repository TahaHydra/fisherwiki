<#
.SYNOPSIS
    Train FisherWiki V2 for a bounded number of hours, resuming automatically.

.DESCRIPTION
    Designed for the overnight workflow: start it before bed, give it a budget,
    and it stops cleanly on its own having written a checkpoint it has already
    verified is loadable. Run it again the next night and it continues the same
    epoch from where it stopped - not from the start of the epoch, and not from
    the start of the run.

    Stopping early is normal. Ctrl+C once asks it to finish the current step and
    checkpoint; Ctrl+C twice aborts immediately. A checkpoint write is never
    interrupted by the first signal, because that is how checkpoints get
    corrupted.

.PARAMETER Hours
    Wall-clock budget. The trainer stops when the *next* step would exceed it,
    so it lands under the limit rather than over.

.EXAMPLE
    .\train-v2.ps1 -Hours 11
    .\train-v2.ps1 -Hours 8 -Backbone convnext_tiny
    .\train-v2.ps1 -Fresh          # ignore any existing checkpoint
#>
[CmdletBinding()]
param(
    [double] $Hours = 11,
    [string] $Shards = "E:\FisherWiki\shards",
    [string] $Out = "",
    [string] $Backbone = "efficientnet_v2_s",
    [int]    $BatchSize = 32,
    [int]    $Epochs = 15,
    [int]    $Workers = 8,
    [double] $CheckpointMinutes = 20,
    [switch] $Fresh
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv-train\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "training venv not found at $python - see docs/TRAINING.md"
}
if (-not (Test-Path $Shards)) {
    throw "no shards at $Shards - run tools/prepare_shards_v2.py first"
}

if ([string]::IsNullOrWhiteSpace($Out)) {
    $Out = Join-Path (Split-Path -Parent $Shards) "runs\v2_$Backbone"
}

$argv = @(
    (Join-Path $root "ml\train_v2.py"),
    "--shards", $Shards,
    "--out", $Out,
    "--hours", $Hours,
    "--backbone", $Backbone,
    "--batch-size", $BatchSize,
    "--epochs", $Epochs,
    "--workers", $Workers,
    "--checkpoint-minutes", $CheckpointMinutes
)
if ($Fresh) { $argv += "--no-resume" }

Write-Host ""
Write-Host "FisherWiki V2 training" -ForegroundColor Cyan
Write-Host "  shards   : $Shards"
Write-Host "  run dir  : $Out"
Write-Host "  backbone : $Backbone"
Write-Host "  budget   : $Hours hours (stops cleanly, checkpoint every $CheckpointMinutes min)"
Write-Host "  expected : finishes around $((Get-Date).AddHours($Hours).ToString('HH:mm'))"
Write-Host ""

& $python @argv
exit $LASTEXITCODE
