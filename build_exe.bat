@echo off
setlocal EnableDelayedExpansion
REM --------------------------------------------------------------
REM  Build Re:MEMBER AI Voice Agent into standalone executable
REM --------------------------------------------------------------

echo ============================================================
echo       Building Re:MEMBER Voice Agent Self-Contained Executable
echo       BYO LLM + Word Tracking + Script Selection Edition
echo ============================================================
echo.

REM Change to the directory where this script resides
pushd %~dp0

REM Check if virtual environment exists and activate it
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ".venv\Scripts\activate.bat"
) else (
    echo Warning: Virtual environment not found. Using system Python.
    echo Run start_server.bat first to set up the environment.
    echo.
)

REM Install PyInstaller if not already installed
echo Installing PyInstaller...
pip install pyinstaller

REM Clean previous builds
echo Cleaning previous builds...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

REM Build the executable
echo.
echo Building executable... This may take 5-10 minutes.
echo.
pyinstaller voice_agent.spec --clean --noconfirm

REM Check if build was successful
if exist "dist\ReMemberVoiceAgent.exe" (
    echo.
    echo ============================================================
    echo                    BUILD SUCCESSFUL!
    echo ============================================================
    echo.
    echo Executable created: dist\ReMemberVoiceAgent.exe
    echo File size: 
    dir "dist\ReMemberVoiceAgent.exe" | find ".exe"
    echo.
    echo To distribute to your boss:
    echo 1. Copy ONLY the file: dist\ReMemberVoiceAgent.exe
    echo 2. Send this single file to your boss
    echo.
    echo The executable is completely self-contained with:
    echo - API keys embedded (no .env file needed)
    echo - All prompts and configuration embedded
    echo - No external files required
    echo - No Python installation needed on target machine
    echo.
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo                    BUILD FAILED!
    echo ============================================================
    echo.
    echo Check the error messages above and ensure all dependencies
    echo are properly installed.
    echo.
    echo Try running start_server.bat first to set up the environment.
    echo ============================================================
)

echo.
echo Creating instructions file...

REM Create instructions for distribution
echo Creating instructions for your boss...
(
echo Instructions for Running Re:MEMBER Voice Agent
echo =============================================
echo.
echo COMPLETELY SELF-CONTAINED EXECUTABLE
echo - No installation required
echo - No external files needed  
echo - API keys pre-configured
echo - All settings embedded
echo.
echo FEATURES:
echo - BYO LLM with Groq integration
echo - Advanced word-level tracking for barge-in handling
echo - Natural conversation flow with interruption awareness
echo - Real-time voice processing with <300ms latency
echo.
echo Requirements:
echo - Windows 10/11
echo - Internet connection
echo - Microphone and speakers/headphones
echo.
echo Setup Steps:
echo 1. Double-click ReMemberVoiceAgent.exe to start
echo 2. A browser window will open automatically (wait 30-60 seconds)
echo 3. Follow the on-screen instructions to test your microphone
echo 4. Select voice and script options
echo 5. Click "Start Call" to begin
echo.
echo That's it! No configuration files needed.
echo.
echo Advanced Features:
echo - Barge-in Detection: Agent knows exactly what words were spoken if interrupted
echo - Cost Optimization: Uses Groq LLM for faster inference and lower costs
echo - Natural Conversation: Handles interruptions like a real human
echo - Word Tracking: Perfect conversation context even during interruptions
echo.
echo Troubleshooting:
echo - If the browser doesn't open automatically, go to: http://127.0.0.1:7860
echo - Ensure microphone permissions are granted when prompted
echo - The first startup may take 30-60 seconds to load models
echo - If the call ends immediately, restart the program and try again
echo.
echo To stop: Close the console window or browser tab
echo.
echo Technical Details:
echo - Voice: Cora (aura-2-cora-en)
echo - LLM: Groq meta-llama/llama-4-maverick-17b-128e-instruct
echo - STT: Deepgram Nova-3
echo.
echo Contact: [Your contact information here]
) > "dist\README_FOR_BOSS.txt"

echo Instructions created: dist\README_FOR_BOSS.txt
echo.
echo Build complete! Single self-contained executable ready for distribution.

REM Return to original directory
popd
pause 