@echo off
title emotion detection v3.0
color 0B
cls

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo ================================================================================
echo                       emotion detection v3.0                              
echo ================================================================================
echo   system specifications                                                   
echo --------------------------------------------------------------------------------
echo   accuracy:     74.43%% model                                
echo   bias control: 60%% sad reduction + 70%% neutral boost                                    
echo   processing:   10 fps optimized                                          
echo   emotions:     7 categories with confidence scoring                      
echo --------------------------------------------------------------------------------
echo.
echo detecting python environment...
echo.

REM Check for Python first
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo python found: 
    python --version
    
    REM Check if required packages are installed
    python -c "import torch, cv2, torchvision, PIL, numpy" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [warning] required packages not found
        echo installing cpu-compatible packages...
        pip install opencv-python torch torchvision pillow numpy --quiet
        if %errorlevel% neq 0 (
            echo [error] failed to install packages
            echo please run: pip install opencv-python torch torchvision pillow numpy
            pause
            exit /b 1
        )
        echo packages installed successfully
    )
    
    echo environment check: passed
    echo loading ai model: emotion_model.pth
    echo bias correction: 60%% sad reduction + 70%% neutral boost
    echo.
    echo --------------------------------------------------------------------------------
    echo   controls guide                                                          
    echo --------------------------------------------------------------------------------
    echo   press 's' to save screenshot                                             
    echo   press 'q' to quit application                                                             
    echo --------------------------------------------------------------------------------
    echo.
    echo starting camera feed... please wait for the interface to load!
    echo ================================================================================
    echo.
    
    REM Run with Python
    python emotion_detector.py
    
) else (
    echo [error] python not found!
    echo.
    echo please install python from: https://www.python.org/downloads/
    echo make sure to add python to PATH during installation
    echo.
    echo alternatively, install anaconda from: https://www.anaconda.com/
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo                           session completed                               
echo ================================================================================
echo.
echo thank you for using emotion detection v3.0!
echo your session has ended successfully.
echo. 
echo github repository: https://github.com/nawjasinghe/facial-emotion-detector
echo please star this project if you found it neat!
echo.
echo press any key to close this window...
echo ========================================================
echo session completed!
echo.
pause
