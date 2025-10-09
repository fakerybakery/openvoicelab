@echo off
setlocal enabledelayedexpansion

echo OpenVoiceLab Setup
echo ====================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.9 or later.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Create virtual environment
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install light-the-torch
echo Installing light-the-torch...
pip install light-the-torch

REM Install PyTorch with optimal configuration
echo Installing PyTorch with optimal configuration...
ltt install torch torchvision torchaudio

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
if not exist "data\" mkdir data
if not exist "logs\" mkdir logs
if not exist "training_runs\" mkdir training_runs

echo.
echo Setup complete!
echo.
echo To get started:
echo   1. Activate the environment: venv\Scripts\activate
echo   2. Run the app: python -m ovl.cli
echo.
pause
