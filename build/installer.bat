@echo off
SETLOCAL

REM Define the version of Python to install
SET PYTHON_VERSION=3.8.10

REM Download Python installer
echo Downloading Python...
curl -O https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe

REM Install Python silently
echo Installing Python...
start /wait python-%PYTHON_VERSION%-amd64.exe /quiet InstallAllUsers=1 PrependPath=1

REM Clean up the installer
del python-%PYTHON_VERSION%-amd64.exe

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Create a virtual environment
echo Creating a virtual environment...
python -m venv vita-env

REM Activate the virtual environment
call vita-env\Scripts\activate

REM Install required libraries in the virtual environment
echo Installing required libraries...
pip install PyOpenGL PyOpenGL_accelerate glfw pillow pywin32 opencv-python keyboard pyautogui
pip install numpy==1.23.5

echo VitaScaler installed sucsessfully
echo You can run VitaScaler by running run.bat
ENDLOCAL
pause
