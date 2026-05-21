@echo off
title SenseFlow

echo ================================
echo   SenseFlow
echo ================================
echo.

cd /d "%~dp0"

where streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] streamlit not found. Run: pip install streamlit
    pause
    exit /b 1
)

echo Starting SenseFlow...
echo Browser will open http://localhost:8080
echo Press Ctrl+C to stop
echo.

streamlit run app.py --server.port 8080 --server.headless false

pause
