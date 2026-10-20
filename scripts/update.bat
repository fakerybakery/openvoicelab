@echo off
cd /d "%~dp0\.."

echo Updating OpenVoiceLab...
git pull
if errorlevel 1 (
    echo Failed to pull updates. Please check your git configuration.
    pause
    exit /b 1
)

if not exist "venv\" (
    echo Virtual environment not found. Run scripts\setup.bat first.
    pause
    exit /b 1
)

venv\Scripts\python -m pip install --quiet --upgrade pip uv
venv\Scripts\uv pip install --python venv\Scripts\python.exe --torch-backend=auto --upgrade -e .
if errorlevel 1 (
    echo Update failed.
    pause
    exit /b 1
)

echo.
echo Update complete! Installed models are refreshed automatically when their requirements change.
pause
