#!/bin/bash

# Define the virtual environment directory name
VENV_DIR="venv"

# List of packages to try installing with apt
apt_packages=(
    "python3-numpy"
    "python3-pygame"
    "python3-opengl"
    "python3-opencv"
    "python3-pil"
    "python3-pyautogui"
)

# Function to check and install each apt package
install_with_apt() {
    echo "Checking and installing apt packages..."
    for pkg in "${apt_packages[@]}"; do
        dpkg -s "$pkg" &> /dev/null
        if [ $? -ne 0 ]; then
            echo "Installing $pkg with sudo apt install..."
            sudo apt install -y "$pkg"
            if [ $? -ne 0 ]; then
                echo "Failed to install $pkg with apt. Please check the package or your internet connection."
                exit 1
            fi
        else
            echo "$pkg is already installed."
        fi
    done
}

# Install apt dependencies
install_with_apt

# Check if the virtual environment already exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to create a virtual environment. Ensure Python 3 is installed."
        exit 1
    fi
else
    echo "Virtual environment found."
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate the virtual environment."
    exit 1
fi

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Failed to upgrade pip."
    exit 1
fi

# Install Python packages not handled by apt using pip
echo "Installing remaining Python packages from requirements.txt using pip..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install some dependencies with pip."
    exit 1
fi

# Run the Python script and log output
echo "Running the Python script..."
python main.py

if [ $? -ne 0 ]; then
    echo "Python script encountered an error."
    exit 1
fi

# Deactivate the virtual environment after script execution
echo "Deactivating the virtual environment..."
deactivate

echo "Script completed successfully."

# Pause the terminal to keep it open
echo "Press any key to close..."
read -n 1 -s
