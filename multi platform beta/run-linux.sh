#!/bin/bash

# Define the virtual environment directory name
VENV_DIR="venv"

# Check if the virtual environment already exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip to the latest version
pip install --upgrade pip

# Install the dependencies from requirements.txt
pip install -r requirements.txt

# Run the Python script
python main.py

# Deactivate the virtual environment after script execution
deactivate
