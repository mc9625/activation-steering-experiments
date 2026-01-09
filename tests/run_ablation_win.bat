@echo off
echo ============================================
echo STEERING VS PROMPTING ABLATION STUDY
echo ============================================
echo.

cd /d "%~dp0"

REM Check if vectors exist
if not exist "vectors\melatonin.pt" (
    echo ERROR: vectors\melatonin.pt not found
    echo Please ensure steering vectors are in the vectors\ directory
    pause
    exit /b 1
)

echo Running ablation study...
echo Compound: MELATONIN
echo Tests: T5_introspection
echo Intensities: 5.0, 8.0, 12.0
echo Iterations: 20 per condition
echo.

python tests\run_ablation.py -c melatonin -t T5_introspection -n 20 --multi-intensity

echo.
echo ============================================
echo COMPLETE - Results in ablation_results\
echo ============================================
pause
