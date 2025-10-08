#!/bin/bash
set -e

echo "Starting OpenVoiceLab..."
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run the app
python -m ovl.cli "$@"
