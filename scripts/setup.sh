#!/bin/bash
set -e

echo "OpenVoiceLab Setup"
echo "===================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.9 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install light-the-torch
echo "Installing light-the-torch..."
pip install light-the-torch

# Install PyTorch with optimal configuration
echo "Installing PyTorch with optimal configuration..."
ltt install torch torchvision torchaudio

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs training_runs

echo ""
echo "Setup complete!"
echo ""
echo "To get started:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Run the app: python -m ovl.cli"
echo ""
