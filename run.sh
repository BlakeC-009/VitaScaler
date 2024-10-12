#!/bin/bash
set -e
# Activate the virtual environment
echo "Activating virtual environment..."
source vita-env/bin/activate
echo "Running vita_upscaler.py..."
python vita_upscaler.py
echo "Deactivating virtual environment..."
deactivate
sleep 1
