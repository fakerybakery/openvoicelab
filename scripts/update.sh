#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "Updating OpenVoiceLab..."
git pull

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Run scripts/setup.sh first."
    exit 1
fi

venv/bin/python -m pip install --quiet --upgrade pip uv
venv/bin/uv pip install --python venv/bin/python --torch-backend=auto --upgrade -e .

echo ""
echo "Update complete! Installed models are refreshed automatically when their requirements change."
