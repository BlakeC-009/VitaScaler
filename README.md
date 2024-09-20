# VitaScaler

VitaScaler is a small, lightweight program written in python and glsl to capture, upscale and display the video feed from a ps vita outputting video over usb via plugins such as https://github.com/xerpi/vita-udcd-uvc.

## Features
- Real-time PS Vita display on your PC at 60fps
- Upscaling support for a variety of screen resolutions
- Self-contained through a python virtual environment
- Custom upscaling method using edge detection
- Easy to use with a custom install and run script for fast setup
- ~16ms of latency

## Requirements
- A gpu capable of running the latest opengl
- Windows 10+ (with Visual C++ 2022 redistributable installed)
- A ps vita running vita-udcd-uvc or equivalent plugin

## Installation

1. Clone the repository or download the latest release and run the install.bat file.
2. Wait for install and then run the run.bat file. Do not click the python file on its own.
3. The program will ask you to select your device EG: "1" and once entered, it will open a stream and you can play from your pc screen.
4. Press "ESC" to exit the stream