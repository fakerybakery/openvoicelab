#!/bin/bash
set -e
cd "$(dirname "$0")/.."

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Run scripts/setup.sh first."
    exit 1
fi

venv/bin/python -m ovl.cli "$@"
