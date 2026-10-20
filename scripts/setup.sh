#!/bin/bash
# Creates ./venv with the OpenVoiceLab app. Model environments are installed
# later, on demand, from the Models tab (or `openvoicelab install <model>`).
set -e
cd "$(dirname "$0")/.."

echo "OpenVoiceLab Setup"
echo "=================="

if [ ! -d "venv" ]; then
    if command -v uv &> /dev/null; then
        echo "Creating virtual environment with uv..."
        uv venv --seed --python 3.12 venv
    elif command -v python3 &> /dev/null; then
        echo "Creating virtual environment with $(python3 --version)..."
        python3 -m venv venv
    else
        echo "Python 3.10+ is required (or install uv: https://docs.astral.sh/uv/)."
        exit 1
    fi
fi

echo "Installing uv..."
venv/bin/python -m pip install --quiet --upgrade pip uv

echo "Installing OpenVoiceLab..."
# --torch-backend=auto picks the right PyTorch build for your GPU (CUDA, ROCm or CPU).
venv/bin/uv pip install --python venv/bin/python --torch-backend=auto -e .

echo ""
echo "Setup complete! Start the app with: scripts/run.sh"
