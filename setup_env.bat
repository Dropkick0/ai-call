@echo off
REM --------------------------------------------------------------
REM  Setup the Python environment for the AI Cold Calls project
REM --------------------------------------------------------------

pushd %~dp0

REM --- Check for Python Installer ---
if exist "python-3.11.0-amd64.exe" (
    echo Found Python installer.
    echo Installing Python 3.11.0...
    start /wait python-3.11.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    echo Python installation complete.
) else (
    echo Python installer not found. Please download python-3.11.0-amd64.exe and place it in the project root.
    goto :eof
)

REM --- Create Virtual Environment ---
echo Creating virtual environment...
python -m venv .venv
echo Virtual environment created.

REM --- Activate Virtual Environment and Install Dependencies ---
echo Activating virtual environment and installing dependencies...
call .\.venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo Setup complete. The virtual environment is ready to use.

popd
pause 