@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 where py >nul 2>&1
 if not errorlevel 1 (py -3 -m venv .venv) else (python -m venv .venv)
 if errorlevel 1 (
  echo Falha ao criar ambiente. Instale Python 3.11 ou superior.
  pause
  exit /b 1
 )
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 (pause & exit /b 1)
python app.py
if errorlevel 1 pause
