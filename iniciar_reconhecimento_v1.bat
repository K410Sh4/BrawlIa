@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-v1\Scripts\python.exe" (
    py -3.11 -m venv .venv-v1
    if errorlevel 1 (
        echo Python 3.11 necessario para esta configuracao inicial.
        pause
        exit /b 1
    )
)
".venv-v1\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Falha ao instalar dependencias.
    pause
    exit /b 1
)
echo Abra o BlueStacks visivel e mantenha a janela descoberta.
".venv-v1\Scripts\python.exe" run_recognition_v1.py --preview
pause
