@echo off
setlocal
cd /d "%~dp0"
set "VENV=.venv-v1"
if not exist "%VENV%\Scripts\python.exe" (
 echo Execute iniciar_windows.bat antes de compilar.
 pause
 exit /b 1
)
call "%VENV%\Scripts\activate.bat"
python -m pip install pyinstaller
if errorlevel 1 (pause & exit /b 1)
python build_exe.py
if errorlevel 1 (pause & exit /b 1)
pause
