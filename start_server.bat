@echo off
setlocal EnableDelayedExpansion
REM --------------------------------------------------------------
REM  Start the Re:MEMBER AI Voice Agent demo (Gradio server)
REM  Supports BYO LLM with Groq ($0.07/min) or Deepgram LLM ($0.08/min)
REM  Configure in .env file or config_defaults.py
REM --------------------------------------------------------------

REM Change to the directory where this script resides
pushd %~dp0

REM --- Check if Python 3.11 is installed ---
echo Checking for Python 3.11...
where python >nul 2>nul
if errorlevel 1 (
    echo Python executable not found in PATH.
    goto install_python
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do (
    set "PY_VERSION_OUTPUT=%%i"
)

echo "!PY_VERSION_OUTPUT!" | findstr "3.11" >nul
if !errorlevel! equ 0 (
    echo Python 3.11 found: !PY_VERSION_OUTPUT!
    goto skip_install
) else (
    echo Found different Python version: !PY_VERSION_OUTPUT!
    echo Proceeding with installation to ensure correct version.
    goto install_python
)

:install_python
    echo Python 3.11 not found, attempting to install...
    REM --- Check for Python Installer and run with progress indicator ---
    if exist "python-3.11.0-amd64.exe" (
        echo Found Python installer. Installing Python 3.11...
        echo.
        echo --------------------------------------------------------------------
        echo  ACTION REQUIRED: Please approve the administrator permission prompt
        echo  that has appeared to allow the Python installation to proceed.
        echo  
        echo  (The prompt may be flashing on your taskbar)
        echo --------------------------------------------------------------------
        echo.
        
        REM Start the installer silently in the background
        start "" /B python-3.11.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

    :check_install_loop
        REM Check if the installer process is still running
        tasklist /NH /FI "IMAGENAME eq python-3.11.0-amd64.exe" 2>NUL | find /I "python-3.11.0-amd64.exe" > NUL
        if %errorlevel% neq 0 goto :install_done
        echo. Installation in progress...

        goto :check_install_loop

    :install_done
        echo.
        echo Python installation complete.
    ) else (
        echo Now please close this window and run this script again to connect to the server.
        pause
        popd
        exit /b
    )

:skip_install
REM --- Setup and activate virtual environment ---
if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating one...
    python -m venv .venv || (
        echo Failed to create virtual environment. Ensure Python is installed and on your PATH.
        pause
        popd
        exit /b
    )
    echo Virtual environment created.
)

echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

REM --- Upgrade pip ---
echo Upgrading pip...
python.exe -m pip install --upgrade pip

REM --- Install/update dependencies ---
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

REM --- Launch the Gradio app ---
echo Launching the application...
python run_local.py

REM Keep the window open after the server stops
pause

REM Return to original directory
popd 