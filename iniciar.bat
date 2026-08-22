@echo off
REM Abre o pipeline de previsao de CBR em uma janela de terminal.
REM Usa a venv do projeto quando existir; senao, o Python do sistema.

setlocal
cd /d "%~dp0"

REM Codepage UTF-8: sem isso o console legado quebra nos acentos e nos simbolos.
chcp 65001 >nul

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" src\main.py
if errorlevel 1 (
    echo.
    echo O pipeline terminou com erro. Leia a mensagem acima.
    pause
)

endlocal
