# Building Re:MEMBER Voice Agent Executable

This guide explains how to create a standalone executable file that your boss can run without any Python setup.

## Quick Start

1. **Run the existing setup first** (to ensure everything works):
   ```
   start_server.bat
   ```

2. **Build the executable**:
   ```
   build_exe.bat
   ```

3. **Distribute to your boss**:
   - Copy `dist/ReMemberVoiceAgent.exe`
   - Copy `dist/.env.template` and rename it to `.env`
   - Add your API keys to the `.env` file
   - Copy `dist/README_FOR_BOSS.txt`

## What Gets Created

The build process creates a `dist/` folder containing:

- **`ReMemberVoiceAgent.exe`** - The main executable (200-400 MB)
- **`README_FOR_BOSS.txt`** - Instructions for your boss
- **`.env.template`** - Template for API keys

## How It Works

The executable:
- ✅ **Self-contained** - No Python installation required
- ✅ **All dependencies included** - Gradio, Deepgram, Groq, etc.
- ✅ **Automatic browser opening** - Opens web interface automatically
- ✅ **Status messages** - Shows progress during startup
- ✅ **Cross-platform** - Works on any Windows 10/11 machine

## Distribution Package

Your boss will need these 2 files:

1. **`ReMemberVoiceAgent.exe`** (the application)
2. **`.env`** (with your actual API keys)

Example `.env` file:
```
DEEPGRAM_API_KEY=your_actual_deepgram_key_here
GROQ_API_KEY=your_actual_groq_key_here
```

## For Your Boss - Usage Instructions

1. **Put both files in the same folder**
2. **Double-click `ReMemberVoiceAgent.exe`**
3. **Wait 30-60 seconds** for the browser to open
4. **Follow the web interface** to test microphone and start calls

## Troubleshooting Build Issues

### If build fails:

1. **Ensure virtual environment is set up**:
   ```
   start_server.bat
   ```

2. **Try manual build**:
   ```
   .venv\Scripts\activate.bat
   pip install pyinstaller
   pyinstaller voice_agent.spec --clean
   ```

3. **Check for missing dependencies**:
   - Look for import errors in the build output
   - Add missing modules to `hidden_imports` in `voice_agent.spec`

### Common issues:

- **"Module not found"** - Add to `hidden_imports` list
- **"File not found"** - Add to `datas` list in spec file
- **Large file size** - Normal (200-400 MB due to ML dependencies)

## Technical Details

### Files Created:
- `app_launcher.py` - Main entry point for executable
- `voice_agent.spec` - PyInstaller configuration
- `build_exe.bat` - Build automation script

### What's Included:
- Complete Python runtime
- All required packages (Gradio, Deepgram, Groq, etc.)
- Prompt files from `prompts/` folder
- Audio processing libraries
- Web server components

### What's Excluded:
- Development tools (pytest, black, etc.)
- Unnecessary packages (tkinter, matplotlib, etc.)
- Source code (compiled to bytecode)

## Security Notes

- The executable contains all your source code (as bytecode)
- API keys are stored in a separate `.env` file
- No sensitive data is embedded in the executable itself
- Your boss can only run the app, not see the source code

## Updates

To update the application:
1. Make changes to your source code
2. Run `build_exe.bat` again
3. Send the new executable to your boss

The executable version number and modification date will help track versions.

---

**Need help?** Check the build output for specific error messages or contact the development team. 