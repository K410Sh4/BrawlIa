@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 echo Execute iniciar_windows.bat antes.
 pause
 exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install pyinstaller
if errorlevel 1 (pause & exit /b 1)
python build_exe.py
if errorlevel 1 (pause & exit /b 1)
pause
