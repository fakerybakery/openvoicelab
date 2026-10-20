@echo off
REM Creates .\venv with the OpenVoiceLab app. Model environments are installed
REM later, on demand, from the Models tab (or `openvoicelab install <model>`).
setlocal
cd /d "%~dp0\.."

echo OpenVoiceLab Setup
echo ==================

if not exist "venv\" (
    where uv >nul 2>&1
    if not errorlevel 1 (
        echo Creating virtual environment with uv...
        uv venv --seed --python 3.12 venv
    ) else (
        python --version >nul 2>&1
        if errorlevel 1 (
            echo Python 3.10+ is required ^(or install uv: https://docs.astral.sh/uv/^).
            pause
            exit /b 1
        )
        echo Creating virtual environment...
        python -m venv venv
    )
)
if errorlevel 1 goto :fail

echo Installing uv...
venv\Scripts\python -m pip install --quiet --upgrade pip uv
if errorlevel 1 goto :fail

echo Installing OpenVoiceLab...
REM --torch-backend=auto picks the right PyTorch build for your GPU (CUDA or CPU).
venv\Scripts\uv pip install --python venv\Scripts\python.exe --torch-backend=auto -e .
if errorlevel 1 goto :fail

echo.
echo Setup complete! Start the app with: scripts\run.bat
pause
exit /b 0

:fail
echo Setup failed.
pause
exit /b 1
