@echo off
title Compound Synthesis
color 0A

echo ========================================
echo COMPOUND SYNTHESIS
echo ========================================
echo.

:: Check if venv exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

:: Install dependencies
echo Checking dependencies...
pip install -r requirements.txt --quiet

:: Check for substance definitions
if not exist substances (
    echo [ERROR] substances\ directory not found
    pause
    exit /b 1
)

:: Count and synthesize each compound
echo.
echo Synthesizing compounds...
echo.

for %%f in (substances\*.json) do (
    echo Synthesizing: %%~nf
    python tools\synthesize.py --file "%%f" --outdir vectors
    echo.
)

echo ========================================
echo SYNTHESIS COMPLETE
echo ========================================
echo.
echo Vectors saved to: vectors\
dir /b vectors\*.pt 2>nul || echo (no vectors found)
echo.
pause
