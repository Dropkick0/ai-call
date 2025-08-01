@echo off
echo Starting Deepgram Voice Agent + Groq Strategic Planning Demo
echo ============================================================
echo.
echo This demo uses:
echo - Deepgram Voice Agent API for all voice processing (STT, TTS, VAD, Echo cancellation)
echo - Groq API for strategic conversation planning and guidance
echo.
echo Make sure you have set these environment variables:
echo - DEEPGRAM_API_KEY (your Deepgram API key)
echo - GROQ_API_KEY (your Groq API key)
echo.
pause
python demo_mode_deepgram_voice_agent.py
pause 