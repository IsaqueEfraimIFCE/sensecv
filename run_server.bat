@echo off
setlocal
cd /d "%~dp0"
set "APP_DIR=%CD%"
set "PYTHONPATH=%APP_DIR%"
set "PORT=5000"

if not exist "%APP_DIR%\.venv\Scripts\python.exe" (
  echo Python virtual environment not found: "%APP_DIR%\.venv"
  pause
  exit /b 1
)

echo Restarting PilotGuru server from "%APP_DIR%"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  echo Stopping existing process on port %PORT%: %%P
  taskkill /F /PID %%P >nul 2>nul
)

"%APP_DIR%\.venv\Scripts\python.exe" "%APP_DIR%\app.py"
if errorlevel 1 pause
