@echo off
SETLOCAL
call vita-env\Scripts\activate
python vita_upscaler.py
deactivate
ENDLOCAL
pause
SLEEP 1
