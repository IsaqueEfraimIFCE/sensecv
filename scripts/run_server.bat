@echo off
setlocal
cd /d "%~dp0\.."
set "PROJECT_ROOT=%CD%"
set "PYTHONPATH=%PROJECT_ROOT%\src"
if not defined SENSECV_DATA_DIR set "SENSECV_DATA_DIR=%PROJECT_ROOT%\data"
if not defined PORT set "PORT=5000"

if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
  set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

echo Restarting senseCV server from "%PROJECT_ROOT%"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  echo Stopping existing process on port %PORT%: %%P
  taskkill /F /PID %%P >nul 2>nul
)

"%PYTHON%" -m sensecv.app
if errorlevel 1 pause