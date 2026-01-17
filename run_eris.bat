@echo off
REM Quick launcher for Evlf Eris

echo.
echo ===================================
echo    Evlf Eris - Model Surgery
echo ===================================
echo.

REM Check if virtual environment exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install dependencies if needed
if not exist .venv\Lib\site-packages\torch (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Run chat
python inference\chat.py %*

deactivate
