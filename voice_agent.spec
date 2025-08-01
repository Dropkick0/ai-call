# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Re:MEMBER AI Voice Agent
Creates a standalone executable with all dependencies bundled.
"""

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_all

# Get the current directory
current_dir = Path.cwd()

# Automatically collect all data files and imports for packages
gradio_datas, gradio_binaries, gradio_hiddenimports = collect_all('gradio')
gradio_client_datas, gradio_client_binaries, gradio_client_hiddenimports = collect_all('gradio_client')
safehttpx_datas = collect_data_files('safehttpx')
groovy_datas = collect_data_files('groovy')

# Define data files to include
data_files = [
    # No external prompt files needed - everything is embedded
    # ('prompts', 'prompts'),  # Commented out - using embedded prompts
] + gradio_datas + gradio_client_datas + safehttpx_datas + groovy_datas

# Define hidden imports (modules that PyInstaller might miss)
manual_hidden_imports = [
    'app',
    'demo_mode_GROQ_deepgram_voice_agent',
    'conversation_planner',
    'config_defaults',
    'gcal',
    'db',
    'metrics',
    
    # Additional Gradio modules that collect_all might miss
    'gradio.templates',
    'gradio.templating',
    'gradio.flagging',
    'gradio.monitoring_dashboard',
    'gradio.oauth',
    'gradio.queueing',
    'gradio.networking',
    'gradio.analytics',
    'gradio.external',
    'gradio.external_utils',
    'gradio.pipelines',
    'gradio.pipelines_utils',
    'gradio.chat_interface',
    'gradio.mcp',
    'gradio.context',
    'gradio.state_holder',
    'gradio.ranged_response',
    'gradio.renderable',
    'gradio.screen_recording_utils',
    'gradio.server_messages',
    'gradio.wasm_utils',
    'gradio.route_utils',
    'gradio.image_utils',
    'gradio.i18n',
    'gradio.ipython_ext',
    'gradio.node_server',
    'gradio.tunneling',
    'gradio.brotli_middleware',
    'gradio.http_server',
    
    # Web framework
    'fastapi',
    'uvicorn',
    'starlette',
    
    # Audio processing
    'sounddevice',
    'soundfile',
    'numpy',
    'pydub',
    
    # API clients
    'deepgram',
    'groq',
    'requests',
    
    # Environment and config  
    'yaml',
    'pyyaml',
    
    # Async support
    'asyncio',
    'aiohttp',
    'aiosignal',
    'aiohappyeyeballs',
    
    # Other dependencies
    'structlog',
    'websockets',
    'certifi',
    'charset_normalizer',
    'urllib3',
    'packaging',
    'typing_extensions',
    'pydantic',
    'pydantic_core',
    'marshmallow',
    'dataclasses_json',
    
    # Platform specific
    'win32api',  # Windows-specific, will be ignored on other platforms
    'winsound',  # Windows-specific, will be ignored on other platforms
]

# Combine manual imports with automatically collected ones
hidden_imports = manual_hidden_imports + gradio_hiddenimports + gradio_client_hiddenimports

a = Analysis(
    ['app_launcher.py'],
    pathex=[str(current_dir)],
    binaries=gradio_binaries + gradio_client_binaries,
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'tkinter',
        'matplotlib',
        'pytest',
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'black',
        'flake8',
        'mypy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ReMemberVoiceAgent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress the executable
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console window for status messages
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # You can add an .ico file here if you have one
    version=None,
) 