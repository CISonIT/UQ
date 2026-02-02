@echo off
echo ================================================================
echo  WAREHOUSE ORDER PICKING UQ ANALYSIS - PYTHON SIMULATION
echo ================================================================
echo.
echo Checking Python installation...
echo.

REM Try different Python commands
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Python found!
    echo.
    python --version
    echo.
    echo Installing dependencies...
    python -m pip install -r requirements.txt
    echo.
    echo Running simulation...
    python uq_simulation.py
    goto :end
)

where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Python3 found!
    echo.
    python3 --version
    echo.
    echo Installing dependencies...
    python3 -m pip install -r requirements.txt
    echo.
    echo Running simulation...
    python3 uq_simulation.py
    goto :end
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Python launcher found!
    echo.
    py --version
    echo.
    echo Installing dependencies...
    py -m pip install -r requirements.txt
    echo.
    echo Running simulation...
    py uq_simulation.py
    goto :end
)

echo ERROR: Python not found!
echo.
echo Please install Python from https://www.python.org/downloads/
echo or from the Microsoft Store.
echo.
echo After installation, restart your terminal and run this script again.
echo.
pause
exit /b 1

:end
echo.
echo ================================================================
echo  SIMULATION COMPLETE
echo ================================================================
echo.
echo Generated files:
dir /b *.png *.csv 2>nul
echo.
pause
