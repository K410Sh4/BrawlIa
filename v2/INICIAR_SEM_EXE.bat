@echo off
setlocal EnableExtensions
cd /d "%~dp0" || (echo Nao foi possivel entrar na pasta do projeto.& pause & exit /b 1)
set "PYTHON="
where py >nul 2>&1
if not errorlevel 1 (
  py -3.12 -c "import sys" >nul 2>&1 && set "PYTHON=py -3.12"
  if not defined PYTHON py -3.11 -c "import sys" >nul 2>&1 && set "PYTHON=py -3.11"
)
if not defined PYTHON (
  echo Python 3.12/3.11 nao encontrado. Instale Python ou use o BrawlIA-V2.exe.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  %PYTHON% -m venv ".venv"
  if errorlevel 1 (echo Erro ao preparar ambiente virtual.& pause & exit /b 1)
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (echo Erro nas dependencias.& pause & exit /b 1)
".venv\Scripts\python.exe" run_gui.py
if errorlevel 1 pause
