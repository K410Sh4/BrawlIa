@echo off
setlocal
cd /d "%~dp0"
set "VENV=.venv-v1"
if not exist "%VENV%\Scripts\python.exe" (
 echo Preparando o ambiente do BrawlIA V1...
 py -3.12 -m venv "%VENV%" >nul 2>&1
 if errorlevel 1 py -3.11 -m venv "%VENV%" >nul 2>&1
 if errorlevel 1 py -3 -m venv "%VENV%" >nul 2>&1
 if errorlevel 1 python -m venv "%VENV%"
 if errorlevel 1 (
  echo Python nao encontrado. Instale Python 3.11 ou 3.12.
  pause
  exit /b 1
 )
)
call "%VENV%\Scripts\activate.bat"
python -m pip install -r requirements.txt
if errorlevel 1 (pause & exit /b 1)
echo Verificando OCR opcional de participantes...
python -c "import rapidocr, onnxruntime" >nul 2>&1
if errorlevel 1 (
 python -m pip install -r requirements-ocr.txt
 if errorlevel 1 (
  echo ATENCAO: OCR nao foi instalado. Captura e analise continuarao disponiveis.
 )
)
python app.py
if errorlevel 1 pause
