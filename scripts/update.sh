#!/bin/bash
set -e

echo "Updating OpenVoiceLab..."
echo ""

# Pull latest changes from repository
echo "Pulling latest changes from repository..."
git pull
if [ $? -ne 0 ]; then
    echo "Failed to pull updates. Please check your git configuration."
    exit 1
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Update light-the-torch
echo "Updating light-the-torch..."
pip install --upgrade light-the-torch

# Update PyTorch with optimal configuration
echo "Updating PyTorch..."
ltt install --upgrade torch torchvision torchaudio

# Update requirements
echo "Updating dependencies..."
pip install --upgrade -r requirements.txt

echo ""
echo "Update complete!"
echo ""
