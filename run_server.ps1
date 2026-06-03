$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = (Get-Location).Path

$port = 5000
Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object {
    Write-Host "Stopping existing process on port ${port}: $_"
    Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
  }

& ".\.venv\Scripts\python.exe" (Join-Path (Get-Location).Path "app.py")
