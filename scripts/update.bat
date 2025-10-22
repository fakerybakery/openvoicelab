@echo off

echo Updating OpenVoiceLab...
echo.

REM Pull latest changes from repository
echo Pulling latest changes from repository...
git pull
if errorlevel 1 (
    echo Failed to pull updates. Please check your git configuration.
    pause
    exit /b 1
)

REM Check if venv exists
if not exist "venv\" (
    echo Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Update light-the-torch
echo Updating light-the-torch...
pip install --upgrade light-the-torch

REM Update PyTorch with optimal configuration
echo Updating PyTorch...
ltt install --upgrade torch torchvision torchaudio

REM Update requirements
echo Updating dependencies...
pip install --upgrade -r requirements.txt

echo.
echo Update complete!
echo.
pause

