@echo off
title Activation Steering Research Interface
color 0A

echo [Activation Steering] Starting...
echo.

:: Create virtual environment if needed
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate
call venv\Scripts\activate

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet

:: Launch server
echo.
echo [Activation Steering] Launching server...
echo.

python -m uvicorn system.server:app --host 0.0.0.0 --port 8000 --log-level warning

pause
