@echo off
title emotion detection setup
color 0B
cls

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo ================================================================================
echo                    emotion detection v3.0 - setup                       
echo ================================================================================
echo.
echo this will install cpu-optimized dependencies for universal compatibility
echo works on all pc configurations without gpu requirements
echo.

REM Check for Python
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo python installation: found
    python --version
) else (
    echo [error] python not found!
    echo please install python from: https://www.python.org/downloads/
    echo make sure to add python to PATH during installation
    pause
    exit /b 1
)

echo.
echo installing required packages...
echo this may take a few minutes...
echo.

REM Install requirements
if exist "requirements.txt" (
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if %errorlevel% == 0 (
        echo.
        echo ================================================================================
        echo                              setup complete!                               
        echo ================================================================================
        echo.
        echo you can now run: run_emotion_detector.bat
        echo.
    ) else (
        echo.
        echo [error] failed to install some packages
        echo try running: pip install torch torchvision opencv-python pillow numpy
        echo note: installing cpu-only versions for universal compatibility
        echo.
    )
) else (
    echo [error] requirements.txt not found!
    echo installing basic cpu-optimized packages...
    python -m pip install torch torchvision opencv-python pillow numpy
)

echo.
pause
