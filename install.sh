#!/bin/bash
set -e  # Exit on error
PYTHON_VERSION="python3"

echo "Updating package lists..."
sudo apt update
echo "Installing Python and pip..."
sudo apt install -y $PYTHON_VERSION $PYTHON_VERSION-pip
echo "Upgrading pip..."
sudo $PYTHON_VERSION -m pip install --upgrade pip
# Create a virtual environment
echo "Creating a virtual environment..."
$PYTHON_VERSION -m venv vita-env
echo "Activating the virtual environment..."
source vita-env/bin/activate

# Install required libraries
echo "Installing required libraries..."
pip install PyOpenGL PyOpenGL_accelerate glfw pillow pywin32 opencv-python keyboard pyautogui
pip install numpy==1.23.5

echo "VitaScaler installed successfully"
echo "You can run VitaScaler by executing run.sh"
