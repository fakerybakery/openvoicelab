@echo off
cd /d "%~dp0\.."

if not exist "venv\" (
    echo Virtual environment not found. Run scripts\setup.bat first.
    pause
    exit /b 1
)

venv\Scripts\python -m ovl.cli %*
