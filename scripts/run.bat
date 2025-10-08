@echo off

echo Starting OpenVoiceLab...
echo.

REM Check if venv exists
if not exist "venv\" (
    echo Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the app
python -m ovl.cli %*
