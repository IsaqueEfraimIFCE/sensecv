$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:SENSECV_DATA_DIR = if ($env:SENSECV_DATA_DIR) { $env:SENSECV_DATA_DIR } else { Join-Path $ProjectRoot "data" }

$port = if ($env:PORT) { [int]$env:PORT } else { 5000 }
Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object {
    Write-Host "Stopping existing process on port ${port}: $_"
    Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
  }

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  $python = "python"
}
& $python -m sensecv.app