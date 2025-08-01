"""
Deepgram Voice Agent Demo - Using Deepgram Voice Agent API with BYO LLM (Groq)
Leverages Deepgram's native orchestration with Groq as the LLM provider for optimal performance
Includes word-level TTS tracking for accurate barge-in handling
"""
import asyncio
import json
import os
import sys
import time as time_module
import threading
from collections import deque
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import structlog
import sounddevice as sd
import numpy as np
import queue
from dotenv import load_dotenv
from groq import Groq
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    AgentWebSocketEvents,
    AgentKeepAlive,
)
try:
    from deepgram.clients.agent.v1.websocket.options import SettingsOptions
except ImportError:
    # Fallback for different SDK versions
    try:
        from deepgram.clients.agent.v1.options import SettingsOptions
    except ImportError:
        from deepgram.clients.agent.options import SettingsOptions

# Configuration
from config_defaults import (
    AudioConfig, PhoneCallConfig, LLMConfig, PlanningConfig, DebugConfig, TTSConfig,
    get_configuration_summary
)

# BYO LLM Configuration - Can be overridden by environment variables
USE_BYO_LLM = os.getenv("USE_BYO_LLM", str(LLMConfig.USE_BYO_LLM)).lower() == "true"
GROQ_ENDPOINT = LLMConfig.GROQ_ENDPOINT

load_dotenv()

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY environment variable is not set")

# Note: Prompt path is now read at runtime from AGENT_PROMPT_PATH environment variable
# This allows the compiled .exe to properly switch between different prompt files

def load_prompt(path: str) -> str:
    """Load and return the full system prompt.
    
    For compiled executables, uses embedded prompt from environment based on script type.
    For development, loads from file path.
    """
    # Determine script type from file path
    is_gatekeeper_script = "BACKUP" in path
    
    # First try to use embedded prompts (for compiled executable)
    if is_gatekeeper_script:
        embedded_prompt = os.getenv("EMBEDDED_SYSTEM_PROMPT_GATEKEEPER")
        script_name = "Gatekeeper"
    else:
        embedded_prompt = os.getenv("EMBEDDED_SYSTEM_PROMPT_FRIEND")
        script_name = "Friend"
    
    if embedded_prompt:
        print(f"Using embedded {script_name} system prompt (self-contained mode)")
        return embedded_prompt.strip()
    
    # Fallback to generic embedded prompt if specific one not found
    fallback_prompt = os.getenv("EMBEDDED_SYSTEM_PROMPT")
    if fallback_prompt:
        print(f"Using fallback embedded system prompt (self-contained mode)")
        return fallback_prompt.strip()
    
    # Final fallback to file loading (for development)
    try:
        with open(path, "r", encoding="utf-8") as f:
            print(f"Loading {script_name} system prompt from file: {path}")
            return f.read().strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"System prompt file not found at '{path}' and no embedded prompt available. "
            "Please ensure the file is present or run from compiled executable."
        ) from exc

class TTSWordTracker:
    """Advanced word-by-word TTS tracking for precise barge-in handling"""
    
    def __init__(self):
        self.current_text = ""
        self.words = []
        self.word_timings = []  # List of (start_time, end_time) for each word
        self.start_time = None
        self.last_chunk_time = None
        self.chunk_count = 0
        self.speaking_rate_wpm = 150  # Default speaking rate (words per minute)
        self.lock = threading.Lock()
        
        # Real-time tracking
        self.current_word_index = 0
        self.bytes_per_second = 0
        self.total_estimated_duration = 0
        
    def start_tracking(self, text: str):
        """Start tracking a new TTS utterance with optimized timing"""
        with self.lock:
            # Skip if already tracking the same text (prevent duplicate processing)
            if self.current_text == text.strip():
                return
                
            self.current_text = text.strip()
            self.words = self._clean_and_split_words(self.current_text)
            self.start_time = time_module.time()
            self.last_chunk_time = self.start_time
            self.chunk_count = 0
            self.current_word_index = 0
            
            # Fast word timing calculation
            self._calculate_word_timings()
            
            # Minimal logging for performance
            if len(self.words) > 0:
                print(f"🎯 Tracking {len(self.words)} words ({self.total_estimated_duration:.1f}s)")
    
    def _clean_and_split_words(self, text: str) -> list:
        """Clean text and split into trackable words"""
        import re
        # Remove markdown formatting and special characters
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold**
        cleaned = re.sub(r'[—–]', '-', cleaned)  # Normalize dashes
        cleaned = re.sub(r'[^\w\s\-\.,!?]', '', cleaned)  # Remove special chars except basic punctuation
        
        # Split into words, keeping punctuation attached
        words = []
        for word in cleaned.split():
            if word.strip():
                words.append(word.strip())
        
        return words
    
    def _calculate_word_timings(self):
        """Fast word timing calculation with simplified complexity"""
        if not self.words:
            self.word_timings = []
            self.total_estimated_duration = 0
            return
        
        # Fast timing calculation - base rate with simple adjustments
        base_time_per_word = 60.0 / self.speaking_rate_wpm
        
        # Pre-calculate all durations in one pass
        word_durations = []
        for word in self.words:
            # Fast length-based adjustment only
            duration = base_time_per_word
            if len(word) > 6:
                duration *= 1.2
            elif len(word) <= 3:
                duration *= 0.9
            
            # Fast punctuation check
            if word[-1:] in '.!?':
                duration *= 1.3
            elif word[-1:] in ',;:':
                duration *= 1.1
            
            word_durations.append(duration)
        
        # Build timings array efficiently
        self.word_timings = []
        current_time = 0.0
        
        for duration in word_durations:
            start_time = current_time
            end_time = current_time + duration
            self.word_timings.append((start_time, end_time))
            current_time = end_time
        
        self.total_estimated_duration = current_time
    
    def _estimate_syllables(self, word: str) -> int:
        """Estimate syllable count for timing calculations"""
        word = word.lower().strip('.,!?;:')
        if len(word) <= 3:
            return 1
        
        # Simple syllable estimation
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def add_audio_chunk(self, chunk_bytes: int):
        """Fast audio chunk tracking with minimal computation"""
        with self.lock:
            self.chunk_count += 1
            
            # Only update timing occasionally to reduce CPU overhead
            if self.chunk_count % 10 == 0 and self.start_time and self.word_timings:
                current_time = time_module.time()
                elapsed_time = current_time - self.start_time
                self.current_word_index = self._get_current_word_index(elapsed_time)
                self.last_chunk_time = current_time
    
    def _get_current_word_index(self, elapsed_time: float) -> int:
        """Get the index of the word that should be playing at the given time"""
        if not self.word_timings:
            return 0
        
        for i, (start_time, end_time) in enumerate(self.word_timings):
            if start_time <= elapsed_time < end_time:
                return i
            elif elapsed_time < start_time:
                return max(0, i - 1)
        
        # If we're past the end, return the last word index
        return len(self.word_timings) - 1
    
    def get_spoken_words_on_interrupt(self) -> str:
        """Get the exact words spoken when barge-in occurred"""
        with self.lock:
            if not self.words or not self.start_time:
                return ""
            
            # Calculate exact elapsed time
            interrupt_time = time_module.time()
            elapsed_time = interrupt_time - self.start_time
            
            # Find exactly which word was being spoken
            current_word_idx = self._get_current_word_index(elapsed_time)
            
            # Include all words up to and including the current word being spoken
            words_spoken_count = min(current_word_idx + 1, len(self.words))
            words_spoken_count = max(0, words_spoken_count)
            
            spoken_words = self.words[:words_spoken_count]
            spoken_text = " ".join(spoken_words)
            
            # Minimal logging for performance
            completion = (words_spoken_count/len(self.words)*100) if self.words else 0
            print(f"🔴 Barge-in: {words_spoken_count}/{len(self.words)} words ({completion:.0f}%) - '{spoken_text[:30]}{'...' if len(spoken_text) > 30 else ''}'")
            
            return spoken_text
    
    def get_current_word_info(self) -> dict:
        """Get detailed info about current word being spoken"""
        with self.lock:
            if not self.words or not self.start_time:
                return {}
            
            elapsed_time = time_module.time() - self.start_time
            current_idx = self._get_current_word_index(elapsed_time)
            
            return {
                'current_word': self.words[current_idx] if current_idx < len(self.words) else None,
                'word_index': current_idx,
                'total_words': len(self.words),
                'elapsed_time': elapsed_time,
                'completion_percent': (current_idx / len(self.words)) * 100 if self.words else 0,
                'estimated_remaining': self.total_estimated_duration - elapsed_time
            }
    
    def get_full_text(self) -> str:
        """Get the full text that was being spoken"""
        with self.lock:
            return self.current_text
    
    def reset(self):
        """Reset the tracker"""
        with self.lock:
            self.current_text = ""
            self.words = []
            self.word_timings = []
            self.start_time = None
            self.last_chunk_time = None
            self.chunk_count = 0
            self.current_word_index = 0
            self.total_estimated_duration = 0

class DeepgramVoiceAgentPipeline:
    """Main pipeline using Deepgram Voice Agent with BYO LLM (Groq)"""
    
    def __init__(self):
        """Initialize the voice agent pipeline"""
        print("Initializing Deepgram Voice Agent Pipeline with BYO LLM...")
        
        # Initialize Deepgram client
        config = DeepgramClientOptions(
            options={
                "keepalive": "true",
            }
        )
        self.deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
        self.connection = None
        
        # Audio configuration (48 kHz for optimal quality)
        self.sample_rate = 48000
        self.channels = 1
        self.blocksize = 480              # 10 ms frames at 48 kHz
        self.dtype = AudioConfig.DTYPE
        
        # Audio stream and processing
        self.audio_input_queue = queue.Queue()
        self.audio_output_queue = queue.Queue()
        self.audio_stream = None
        self.audio_chunk_counter = 0
        
        # TTS audio buffering for smooth streaming playback
        self.tts_audio_buffer = deque(maxlen=1600)
        self.tts_playback_active = False
        self.audio_lock = threading.Lock()
        self.audio_complete = False
        self.stream_start_threshold = int(os.getenv("DG_START_THRESHOLD", 24))
        self.write_chunks = int(os.getenv("DG_WRITE_CHUNKS", 2))
        self.audio_session_active = False
        self.barge_in_dropped = False
        
        # TTS word tracking for barge-in handling
        self.tts_tracker = TTSWordTracker()
        self.pending_agent_response = ""  # Track what the agent intended to say
        
        # Persistent PortAudio output stream
        try:
            self.output_stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=0,
                latency='low'
            )
            self.output_stream.start()
            print("Output stream started (persistent)")
        except Exception as e:
            self.output_stream = None
            print(f"Could not start persistent output stream: {e}")
        
        # Load system prompt
        prompt_path = os.getenv("AGENT_PROMPT_PATH", "prompts/system_prompt.txt")
        print(f"Loading prompt from: {prompt_path}")
        self.system_prompt = load_prompt(prompt_path)
        
        # Conversation state
        self.conversation_history = []
        self.is_active = False
        
        # Warmup completion flag (set by UI)
        self.warmup_complete = None
        
        # Statistics
        self.stats = {
            'total_exchanges': 0,
            'successful_responses': 0,
            'barge_in_events': 0,
            'partial_responses': 0,
            'start_time': None
        }
        
        print(f"Pipeline initialized successfully (BYO LLM: {USE_BYO_LLM})")
    
    def _display_configuration_summary(self):
        """Display current configuration"""
        print("\n" + "="*80)
        print(" DEEPGRAM VOICE AGENT CONFIGURATION")
        print("="*80)
        print(f" Call Type: Cold Call Sales Demo")
        print(f" Call Objective: Introduce AI solutions and qualify prospects")
        print(f" Sample Rate: {self.sample_rate}Hz")
        if USE_BYO_LLM:
            print(f" LLM Provider: Groq (BYO LLM - ${LLMConfig.BYO_LLM_RATE_PER_MINUTE}/min)")
            print(f" LLM Model: {LLMConfig.MODEL_NAME}")
            print(f" LLM Endpoint: {GROQ_ENDPOINT}")
        else:
            print(f" LLM Provider: Deepgram (Standard - ${LLMConfig.STANDARD_RATE_PER_MINUTE}/min)")
            print(f" LLM Model: {LLMConfig.DEEPGRAM_LLM_MODEL}")
        print(f" STT Provider: Deepgram Nova-3")
        print(f" TTS Provider: Deepgram Aura-2")
        print(f" TTS Voice: {TTSConfig.VOICE_NAME}")
        print(f" Word Tracking: Fast & Accurate (optimized for performance)")
        print(f" First Response: ULTRA-FAST (Pre-warmed LLM+STT+TTS+Pipeline+VAD)")
        print(f" Voice Detection: 500ms endpoint, 70% sensitivity (pre-warmed)")
        print(f" Agent Greeting: None (dial tone signals when FULLY ready)")
        print(f" Call Duration: Unlimited (manual stop only)")
        print(f" Response Optimization: Complete opening responses (reduces delays)")
        print(f" Startup Sequence: Pre-warm → Connect → VAD warmup → Dial tone")
        print(f" Debug Mode: {DebugConfig.DEBUG_LOGGING}")
        print("="*80)
    
    async def _pre_warm_systems(self):
        """Pre-warm LLM, STT, and TTS for ultra-fast first response"""
        try:
            print("🔥 Pre-warming ALL systems for ultra-fast first response...")
            
            # 1. Pre-warm Groq LLM
            if USE_BYO_LLM:
                import httpx
                try:
                    warmup_payload = {
                        "model": LLMConfig.MODEL_NAME,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5,
                        "temperature": 0.1
                    }
                    
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        response = await client.post(
                            GROQ_ENDPOINT,
                            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                            json=warmup_payload
                        )
                        if response.status_code == 200:
                            print("  ✅ Groq LLM pre-warmed")
                        else:
                            print("  ⚠️ Groq pre-warm failed, but continuing...")
                except Exception as e:
                    print(f"  ⚠️ Groq pre-warm error (continuing): {e}")
            
            # 2. Pre-warm Deepgram STT
            try:
                import httpx
                # Generate 100ms of low-volume test audio (1600 bytes at 48kHz, 16-bit)
                import struct
                import math
                
                sample_rate = 48000
                duration_ms = 100
                num_samples = int(sample_rate * duration_ms / 1000)
                
                # Generate very quiet sine wave to avoid noise but activate STT
                test_audio_samples = []
                for i in range(num_samples):
                    # 440Hz sine wave at very low volume
                    sample = int(1000 * math.sin(2 * math.pi * 440 * i / sample_rate))
                    test_audio_samples.append(struct.pack('<h', sample))
                
                test_audio_bytes = b''.join(test_audio_samples)
                
                # Send to Deepgram STT pre-recorded endpoint for warmup
                async with httpx.AsyncClient(timeout=3.0) as client:
                    stt_response = await client.post(
                        "https://api.deepgram.com/v1/listen",
                        headers={
                            "Authorization": f"Token {DEEPGRAM_API_KEY}",
                            "Content-Type": "audio/wav"
                        },
                        params={
                            "model": "nova-2",
                            "smart_format": "true",
                            "punctuate": "false"
                        },
                        content=test_audio_bytes
                    )
                    if stt_response.status_code == 200:
                        print("  ✅ Deepgram STT pre-warmed")
                    else:
                        print("  ⚠️ STT pre-warm failed, but continuing...")
                        
            except Exception as e:
                print(f"  ⚠️ STT pre-warm error (continuing): {e}")
            
            # 3. Pre-warm Deepgram TTS
            try:
                import httpx
                
                async with httpx.AsyncClient(timeout=3.0) as client:
                    tts_response = await client.post(
                        "https://api.deepgram.com/v1/speak",
                        headers={
                            "Authorization": f"Token {DEEPGRAM_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "text": "Hi",
                            "model": "aura-zeus-en",
                            "encoding": "linear16",
                            "sample_rate": 48000
                        }
                    )
                    if tts_response.status_code == 200:
                        print("  ✅ Deepgram TTS pre-warmed")
                    else:
                        print("  ⚠️ TTS pre-warm failed, but continuing...")
                        
            except Exception as e:
                print(f"  ⚠️ TTS pre-warm error (continuing): {e}")
            
            # 4. Pre-warm audio processing pipeline
            test_audio = b'\x00\x00' * 480  # 10ms of silence
            with self.audio_lock:
                self.tts_audio_buffer.append(test_audio)
                self.tts_audio_buffer.clear()
            
            print("  ✅ Audio pipeline pre-warmed")
            

            
            print("🚀🚀 HTTP SYSTEMS PRE-WARMED - VAD will pre-warm after connection! 🚀🚀")
            
        except Exception as e:
            print(f"Pre-warming error (non-critical): {e}")

    async def _pre_warm_vad(self):
        """Pre-warm Voice Activity Detection by sending test audio through live connection"""
        try:
            print("  🎤 Pre-warming VAD (Voice Activity Detection)...")
            
            if not self.connection:
                print("  ⚠️ VAD pre-warm skipped - no connection yet")
                return
                
            # Generate brief test audio to wake up VAD
            import struct
            import math
            
            sample_rate = 48000
            duration_ms = 200  # 200ms of test audio
            num_samples = int(sample_rate * duration_ms / 1000)
            
            # Generate low-volume audio pattern to trigger VAD initialization
            test_audio_samples = []
            for i in range(num_samples):
                # Generate a soft tone that's clearly above silence threshold for VAD
                # but quiet enough not to interfere with user audio
                import random
                frequency = 200  # Low frequency tone
                tone_sample = int(1500 * math.sin(2 * math.pi * frequency * i / sample_rate))
                noise_sample = int(300 * (random.random() - 0.5))  # Add some noise
                combined_sample = tone_sample + noise_sample
                test_audio_samples.append(struct.pack('<h', combined_sample))
            
            test_audio_bytes = b''.join(test_audio_samples)
            
            # Send test audio through live connection to wake up VAD
            if self.connection and hasattr(self.connection, 'send'):
                print(f"    • Sending {len(test_audio_bytes)} bytes of test audio...")
                self.connection.send(test_audio_bytes)
                
                # Give VAD a moment to process and initialize
                await asyncio.sleep(0.4)
                
                # Send a bit of silence to complete the VAD calibration
                silence_samples = []
                for i in range(int(sample_rate * 0.15)):  # 150ms of silence
                    silence_samples.append(struct.pack('<h', 0))
                silence_bytes = b''.join(silence_samples)
                
                print(f"    • Sending {len(silence_bytes)} bytes of silence for calibration...")
                self.connection.send(silence_bytes)
                await asyncio.sleep(0.2)
                
                print("  ✅ VAD pre-warmed with test audio pattern (ready for immediate response)")
            else:
                print("  ⚠️ VAD pre-warm failed - connection not ready")
                
        except Exception as e:
            print(f"  ⚠️ VAD pre-warm error (continuing): {e}")

    async def setup_voice_agent(self) -> bool:
        """Set up the Deepgram Voice Agent connection with BYO LLM configuration"""
        try:
            print("Setting up Deepgram Voice Agent connection...")
            
            # Pre-warm HTTP-based systems for faster first response
            await self._pre_warm_systems()
            
            # Create WebSocket connection
            self.connection = self.deepgram.agent.websocket.v("1")
            
            # Configure agent settings
            options = SettingsOptions()
            
            # Audio configuration
            options.audio.input.encoding = "linear16"
            options.audio.input.sample_rate = 48000
            options.audio.output.encoding = "linear16"
            options.audio.output.sample_rate = 48000
            options.audio.output.container = "none"
            
            # Agent configuration
            options.agent.language = "en"
            
            # STT configuration (Always Deepgram)
            options.agent.listen.provider.type = "deepgram"
            options.agent.listen.provider.model = "nova-3"
            options.agent.listen.provider.keyterms = ["hello", "goodbye", "interested", "not interested", "tell me more"]
            
            # Extend silence timeout
            try:
                options.agent.listen.config.silence_timeout_ms = 15000  # 15 seconds
            except AttributeError:
                try:
                    options.agent.listen.config.silence_timeout = 15000
                except Exception:
                    pass
            
            # LLM configuration - BYO LLM with Groq or standard Deepgram
            if USE_BYO_LLM:
                print("Configuring BYO LLM with Groq...")
                print(f"  Model: {LLMConfig.MODEL_NAME}")
                print(f"  Endpoint: {GROQ_ENDPOINT}")
                print(f"  API Key: {GROQ_API_KEY[:10]}...")
                
                # Debug: Inspect available fields
                try:
                    provider_attrs = [attr for attr in dir(options.agent.think.provider) if not attr.startswith('_')]
                    print(f"  🔍 Available provider fields: {provider_attrs}")
                except Exception as e:
                    print(f"  🔍 Could not inspect provider fields: {e}")
                
                try:
                    # Correct BYO LLM configuration using dictionary syntax
                    # Provider configuration (using dictionary access)
                    options.agent.think.provider["type"] = "open_ai"
                    options.agent.think.provider["model"] = LLMConfig.MODEL_NAME
                    options.agent.think.provider["temperature"] = LLMConfig.TEMPERATURE
                    
                    # Endpoint configuration (required for BYO LLM)
                    options.agent.think.endpoint = {
                        "url": GROQ_ENDPOINT,
                        "headers": {
                            "authorization": f"Bearer {GROQ_API_KEY}"
                        }
                    }
                    print("  ✅ BYO LLM configuration set using dictionary syntax")
                    
                except Exception as e1:
                    print(f"  ❌ BYO LLM configuration failed: {e1}")
                    print("  ⚠️ Falling back to standard Deepgram LLM")
                    # Force fallback to standard LLM configuration using dictionary syntax
                    options.agent.think.provider["type"] = "open_ai"
                    options.agent.think.provider["model"] = LLMConfig.DEEPGRAM_LLM_MODEL
                    options.agent.think.provider["temperature"] = LLMConfig.TEMPERATURE
                    print("  ✅ Standard LLM fallback configured")
                            
            else:
                print("Using standard Deepgram LLM...")
                try:
                    options.agent.think.provider["type"] = "open_ai"  # Deepgram's default
                    options.agent.think.provider["model"] = LLMConfig.DEEPGRAM_LLM_MODEL
                    options.agent.think.provider["temperature"] = LLMConfig.TEMPERATURE
                    print(f"  ✅ Standard LLM configured: {LLMConfig.DEEPGRAM_LLM_MODEL}")
                except Exception as e:
                    print(f"  ❌ Error configuring standard LLM: {e}")
                    # Ultimate fallback - minimal configuration
                    options.agent.think.provider["type"] = "open_ai"
                    options.agent.think.provider["model"] = "gpt-4o-mini"
                    print("  ✅ Using minimal fallback LLM configuration")
            
            # TTS configuration now uses Groq Play-AI
            options.agent.speak.provider.type = "open_ai"
            voice_name = os.getenv("AGENT_VOICE_NAME", "Cheyenne-PlayAI")
            options.agent.speak.provider.model = voice_name
            
            # Set TTS rate
            try:
                options.agent.speak.voice = {"rate": 1.0}
            except AttributeError:
                try:
                    options.agent.speak.voice.rate = 1.0
                except Exception:
                    print("Warning: Could not set TTS voice rate")
            
            # Set system prompt
            try:
                options.agent.think.prompt = self.system_prompt
            except AttributeError:
                options.agent.think.config.prompt = self.system_prompt
            
            # No greeting - user will hear dial tone as ready signal
            
            # Optimize for faster first response (but keep voice detection reasonable)
            try:
                # Balanced settings for good response time without false triggers
                options.agent.listen.config.endpointing_ms = 500  # 500ms (balanced)
                options.agent.listen.config.vad_sensitivity = 0.7  # Moderate sensitivity
                options.agent.listen.config.interim_results = True  # Get partial results faster
            except AttributeError:
                pass
            
            # Register event handlers
            self._register_event_handlers()
            
            # Start the connection
            print("Starting Voice Agent connection...")
            if not self.connection.start(options):
                print("Failed to start Voice Agent connection")
                return False

            print("Voice Agent connection established successfully")
            
            # Pre-warm VAD (Voice Activity Detection) now that connection is live
            await self._pre_warm_vad()
            
            # Brief pause to let VAD warmup settle before starting conversation
            await asyncio.sleep(0.5)
            print("VAD warmup settled - dial tone will signal when ready for input")
            
            # Start keep-alive thread
            self._start_keep_alive()
            
            return True
            
        except Exception as e:
            print(f"Error setting up Voice Agent: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _build_initial_prompt(self) -> str:
        """Return the raw system prompt (kept for backward-compatibility)."""
        return self.system_prompt
    
    def _register_event_handlers(self):
        """Register event handlers for the Voice Agent"""
        print("Registering Voice Agent event handlers...")
        
        # Register all event handlers with the connection
        self.connection.on(AgentWebSocketEvents.Welcome, self._on_welcome)
        self.connection.on(AgentWebSocketEvents.SettingsApplied, self._on_settings_applied)
        self.connection.on(AgentWebSocketEvents.ConversationText, self._on_conversation_text)
        self.connection.on(AgentWebSocketEvents.UserStartedSpeaking, self._on_user_started_speaking)
        self.connection.on(AgentWebSocketEvents.AgentThinking, self._on_agent_thinking)
        self.connection.on(AgentWebSocketEvents.AgentStartedSpeaking, self._on_agent_started_speaking)
        self.connection.on(AgentWebSocketEvents.AudioData, self._on_audio_data)
        self.connection.on(AgentWebSocketEvents.AgentAudioDone, self._on_agent_audio_done)
        self.connection.on(AgentWebSocketEvents.Close, self._on_close)
        self.connection.on(AgentWebSocketEvents.Error, self._on_error)
        self.connection.on(AgentWebSocketEvents.Unhandled, self._on_unhandled)
        
        print("Event handlers registered")
    
    # Event Handler Methods
    def _on_welcome(self, connection, welcome, **kwargs):
        """Handle welcome message"""
        print(f"Voice Agent Welcome: {welcome}")
        print("⚡ Connection ready - optimized for fast first response!")
        self.is_active = True
        self.stats['start_time'] = time_module.time()
        
        # Note: Pipeline is already pre-warmed via HTTP endpoints in setup_voice_agent()
        # No need to send test audio through the live connection as it can interfere
        
        # Reset audio state for new conversation
        with self.audio_lock:
            self.tts_audio_buffer.clear()
            self.tts_playback_active = False
            self.audio_complete = False
            self.audio_session_active = False
            self.tts_tracker.reset() # Reset word tracker
            self.pending_agent_response = ""
    
    def _on_settings_applied(self, connection, settings_applied, **kwargs):
        """Handle settings applied"""
        print(f"Settings Applied: {settings_applied}")
        if USE_BYO_LLM:
            print("BYO LLM (Groq) configuration applied successfully")
    
    def _on_conversation_text(self, connection, conversation_text, **kwargs):
        """Handle conversation text updates with word tracking"""
        try:
            print(f"Conversation Text Event: {conversation_text}")
            
            # Try to extract meaningful conversation data
            if conversation_text:
                # Handle different possible formats
                if hasattr(conversation_text, 'role') and hasattr(conversation_text, 'content'):
                    role = conversation_text.role
                    content = conversation_text.content
                    
                    # For assistant responses, store the intended text for tracking
                    if role == 'assistant':
                        self.pending_agent_response = content
                        # Start word tracking immediately (most reliable trigger point)
                        self.tts_tracker.start_tracking(content)
                        self.stats['successful_responses'] += 1
                        self.stats['total_exchanges'] += 1
                        print(f"ASSISTANT: {content}")
                    else:
                        print(f"USER: {content}")
                    
                    # Add to conversation history
                    self.conversation_history.append({
                        'role': role,
                        'content': content,
                        'timestamp': time_module.time(),
                        'is_partial': False  # Will be updated if barge-in occurs
                    })
                elif hasattr(conversation_text, 'text'):
                    # Handle text-only format
                    text = conversation_text.text
                    print(f"DETECTED SPEECH: {text}")
                else:
                    # Debug: show the actual object structure
                    print(f"Raw conversation object: {type(conversation_text)}")
                    if hasattr(conversation_text, '__dict__'):
                        print(f"Object data: {conversation_text.__dict__}")
                
        except Exception as e:
            print(f"Error processing conversation text: {e}")
    
    def _on_user_started_speaking(self, connection, user_started_speaking, **kwargs):
        """Handle user started speaking with advanced word tracking"""
        print(f"User started speaking! Args: {user_started_speaking}")
        print("Voice detected - audio processing is working!")
        
        # Barge-in: handle partial speech and update conversation history
        with self.audio_lock:
            # Check if we're in the middle of TTS playback
            is_barge_in = (len(self.tts_audio_buffer) > 0 or 
                          self.tts_playback_active or 
                          self.audio_session_active)
            
            if is_barge_in:
                self.barge_in_dropped = True
                self.stats['barge_in_events'] += 1
                
                # Get precise word-level tracking information
                spoken_text = self.tts_tracker.get_spoken_words_on_interrupt()
                full_intended_text = self.tts_tracker.get_full_text()
                word_info = self.tts_tracker.get_current_word_info()
                
                # Fast barge-in processing with minimal logging
                if full_intended_text:  # As long as there was intended text
                    # Ensure we have at least some partial text (even if just first word)
                    if not spoken_text and word_info.get('word_index', 0) >= 0:
                        # If no spoken text calculated, at least include first word
                        words = self.tts_tracker.words
                        if words:
                            spoken_text = words[0]
                    
                    # Update conversation history if we have any partial content
                    if spoken_text and spoken_text.strip():
                        self._update_last_response_with_partial(spoken_text, full_intended_text)
                        self.stats['partial_responses'] += 1
                        print(f"✅ Barge-in #{self.stats['barge_in_events']}: Updated history with partial response")
                
                print("🔄 Barge-in detected - buffer will be flushed gracefully")
    
    def _on_agent_thinking(self, connection, agent_thinking, **kwargs):
        """Handle agent thinking"""
        if USE_BYO_LLM:
            print(f"Groq is processing... Args: {agent_thinking}")
        else:
            print(f"Agent is thinking... Args: {agent_thinking}")
    
    def _on_agent_started_speaking(self, connection, agent_started_speaking, **kwargs):
        """Handle agent started speaking"""
        print(f"Agent started speaking! Args: {agent_started_speaking}")
        
        # Reset audio state for new TTS session
        with self.audio_lock:
            self.tts_audio_buffer.clear()
            self.tts_playback_active = False
            self.audio_complete = False
            self.audio_session_active = True
            
        # Backup trigger (rarely needed due to primary trigger)
        if self.pending_agent_response and not self.tts_tracker.current_text:
            self.tts_tracker.start_tracking(self.pending_agent_response)
            print("🔄 Backup: Word tracking started")
        
        print("New TTS session started - audio state reset for fresh streaming")
    
    def _on_audio_data(self, connection, data, **kwargs):
        """Handle TTS audio data - smart streaming with buffering"""
        try:
            # Check for different possible audio data formats
            audio_data = None
            
            # Format 1: Direct bytes
            if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                audio_data = data
            
            # Format 2: Object with .audio attribute
            elif hasattr(data, 'audio') and isinstance(data.audio, (bytes, bytearray)):
                audio_data = data.audio
            
            # Format 3: Object with .data attribute
            elif hasattr(data, 'data') and isinstance(data.data, (bytes, bytearray)):
                audio_data = data.data
            
            # Format 4: Object with .payload attribute
            elif hasattr(data, 'payload') and isinstance(data.payload, (bytes, bytearray)):
                audio_data = data.payload
            
            # Format 5: Base64 encoded in JSON
            elif hasattr(data, 'audio') and isinstance(data.audio, str):
                try:
                    import base64
                    audio_data = base64.b64decode(data.audio)
                except:
                    pass
            
            if audio_data and len(audio_data) > 0:
                # Track audio chunk for word mapping
                self.tts_tracker.add_audio_chunk(len(audio_data))
                
                # Add to buffer
                with self.audio_lock:
                    # Auto-detect new TTS session: if we receive audio but no session is active,
                    # this must be a new response that didn't trigger AgentStartedSpeaking
                    if not self.audio_session_active and not self.tts_playback_active:
                        print("Auto-detected new TTS session - resetting audio state")
                        self.tts_audio_buffer.clear()  # Clear any stale data
                        self.audio_complete = False
                        self.audio_session_active = True
                        
                        # Emergency backup trigger (very rarely needed)
                        if self.pending_agent_response and not self.tts_tracker.current_text:
                            self.tts_tracker.start_tracking(self.pending_agent_response)
                            print("🔄 Emergency: Word tracking started")
                    
                    self.tts_audio_buffer.append(audio_data)
                    buffer_length = len(self.tts_audio_buffer)
                    
                    # Debug: Show first few chunks only
                    if buffer_length <= 3:
                        print(f"Audio chunk {buffer_length}: session_active={self.audio_session_active}, playback_active={self.tts_playback_active}")
                    
                    # Start streaming after we have enough chunks for smooth playback
                    if buffer_length == self.stream_start_threshold and not self.tts_playback_active:
                        self.tts_playback_active = True
                        self.audio_session_active = True  # Mark session as active when we start streaming
                        print(f"STARTING streaming playback (buffered {buffer_length} chunks)")
                        
                        # Start streaming thread
                        streaming_thread = threading.Thread(target=self._stream_audio_chunks, daemon=True)
                        streaming_thread.start()
                    
                    # Continue adding chunks for the streaming thread to consume
                    elif self.tts_playback_active and buffer_length % 50 == 0:
                        print(f"Streaming... {buffer_length} chunks received")
                    
                # Optional: Save for debugging (first few chunks only)
                if DebugConfig.DEBUG_LOGGING and len(self.tts_audio_buffer) <= 3:
                    self._save_debug_audio(audio_data, f"tts_chunk_{len(self.tts_audio_buffer)}")
                
                return True
            else:
                print(f"No valid audio data found")
                    
        except Exception as e:
            print(f"Error handling audio data: {e}")
            import traceback
            traceback.print_exc()
        
        return False
    
    def _stream_audio_chunks(self):
        """Stream audio chunks using persistent RawOutputStream to minimise latency"""
        try:
            print("Streaming thread active (RawOutputStream mode)...")
            while True:
                with self.audio_lock:
                    # Flush on barge-in
                    if self.barge_in_dropped:
                        self.tts_audio_buffer.clear()
                        self.barge_in_dropped = False

                    # Exit only after we know the agent finished and buffer drained
                    if self.audio_complete and not self.tts_audio_buffer:
                        break

                    # Accumulate up to write_chunks chunks (or whatever is available)
                    buffer = bytearray()
                    while len(buffer) // (self.channels * 2) // 480 < self.write_chunks and self.tts_audio_buffer:
                        buffer.extend(self.tts_audio_buffer.popleft())

                if buffer and self.output_stream:
                    try:
                        self.output_stream.write(bytes(buffer))
                    except sd.PortAudioError as e:
                        print(f"PortAudio write error/underrun: {e}")
                else:
                    # No data ready; yield briefly
                    time_module.sleep(0.002)
        except Exception as e:
            print(f"Error in RawOutputStream streaming: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self.audio_lock:
                self.tts_playback_active = False
                self.audio_session_active = False
            print("Audio streaming session ended (RawOutputStream)")
    
    def _on_agent_audio_done(self, connection, agent_audio_done, **kwargs):
        """Handle agent audio done - signal end of streaming session"""
        print(f"Agent finished speaking - ending streaming session")
        
        # Mark session as complete to let streaming thread finish remaining chunks
        with self.audio_lock:
            self.audio_complete = True
            chunk_count = len(self.tts_audio_buffer)
        
        print(f"Session ended with {chunk_count} total chunks - streaming will finish naturally")
    
    def _on_close(self, connection, close, **kwargs):
        """Handle connection close - log but don't auto-terminate call"""
        print(f"Connection closed: {close}")
        # Don't automatically terminate - only manual stop should end calls
        # self.is_active = False  # Removed to prevent auto-termination
        print("📝 Note: Connection closed but call continues (manual stop required)")
    
    def _on_error(self, connection, error, **kwargs):
        """Handle errors - log but don't auto-terminate call"""
        print(f"Voice Agent Error: {error}")
        print("📝 Note: Error logged but call continues (manual stop required)")
    
    def _on_unhandled(self, connection, unhandled, **kwargs):
        """Handle unhandled events"""
        print(f"Unhandled event: {unhandled}")
    
    def _start_keep_alive(self):
        """Start the keep-alive thread"""
        def keep_alive_loop():
            while self.is_active:
                try:
                    time_module.sleep(5)
                    if self.connection:
                        self.connection.send(str(AgentKeepAlive()))
                        print("Keep-alive sent")
                except Exception as e:
                    print(f"Keep-alive error: {e}")
        
        keep_alive_thread = threading.Thread(target=keep_alive_loop, daemon=True)
        keep_alive_thread.start()
        print("Keep-alive thread started")
    
    def _update_last_response_with_partial(self, spoken_text: str, full_intended_text: str):
        """Update the last assistant response in conversation history with partial text"""
        try:
            # Find the last assistant response quickly (should be the most recent)
            for i in range(len(self.conversation_history) - 1, -1, -1):
                entry = self.conversation_history[i]
                if entry.get('role') == 'assistant' and entry.get('content') == full_intended_text:
                    # Update this entry with the partial response
                    self.conversation_history[i] = {
                        'role': 'assistant',
                        'content': spoken_text,
                        'timestamp': entry['timestamp'],
                        'is_partial': True,
                        'full_intended': full_intended_text
                    }
                    print(f"📝 Updated: '{spoken_text}' (partial)")
                    break
                    
        except Exception as e:
            print(f"Error updating conversation history: {e}")
    
    def _save_debug_audio(self, audio_data: bytes, prefix: str):
        """Save audio data for debugging"""
        try:
            import os
            from datetime import datetime
            
            # Create debug directory if it doesn't exist
            debug_dir = "debug_audio"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # Save raw audio data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{debug_dir}/{prefix}_{timestamp}.raw"
            
            with open(filename, "wb") as f:
                f.write(audio_data)
            
            print(f"Debug audio saved: {filename} ({len(audio_data)} bytes)")
            
        except Exception as e:
            print(f"Could not save debug audio: {e}")
    
    def test_audio_output(self):
        """Test audio output with a dial tone that signals complete readiness"""
        try:
            print("   Playing dial tone - ALL systems including VAD are now ready...")
            
            # Generate a 1-second 440Hz sine wave (dial tone)
            duration = 1.0  # seconds
            sample_rate = 48000
            frequency = 440  # Hz
            
            # Create time array
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Generate sine wave
            tone = 0.3 * np.sin(2 * np.pi * frequency * t)  # 30% volume
            
            # Reshape for mono playback
            tone = tone.reshape(-1, 1)
            
            # Play the tone
            sd.play(tone, samplerate=sample_rate, blocking=True)
            print("   🔔 Dial tone complete - Voice detection will respond IMMEDIATELY!")
            
            # Signal to UI that warmup is complete (dial tone has played)
            if hasattr(self, 'warmup_complete') and self.warmup_complete:
                print("   📡 Sending warmup completion signal to UI...")
                self.warmup_complete.set()
                print("   ✅ Warmup completion signal sent to UI - call timer will now start")
            
        except Exception as e:
            print(f"Audio output test failed: {e}")
            # Still signal completion even if audio failed
            if hasattr(self, 'warmup_complete') and self.warmup_complete:
                print("   📡 Sending warmup completion signal (audio failed, but continuing)...")
                self.warmup_complete.set()
            return False
        
        return True
    
    def audio_callback(self, indata, frames, time, status):
        """Audio callback for sounddevice - captures microphone input"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert numpy array to bytes and send to Deepgram Voice Agent
        try:
            # Convert to int16 PCM format (since our stream dtype is already int16)
            if self.dtype == 'int16':
                # Data is already int16, just convert to bytes
                audio_data = indata.astype(np.int16).tobytes()
            else:
                # Convert float32 to int16 PCM format
                audio_data = (indata * 32767).astype(np.int16).tobytes()
            
            # Send audio data to Deepgram Voice Agent if connected
            if self.connection and self.is_active:
                # Send raw PCM audio to Deepgram Voice Agent
                self.connection.send(audio_data)
                
                # Enhanced logging for initial responsiveness monitoring
                self.audio_chunk_counter += 1
                if self.audio_chunk_counter == 1:
                    print("🎤 First audio chunk sent - microphone active!")
                elif self.audio_chunk_counter % 100 == 0:
                    print(f"Audio flowing: {self.audio_chunk_counter} chunks sent ({len(audio_data)} bytes each)")
                
                # Optional: Add to queue for local processing if needed
                if not self.audio_input_queue.full():
                    self.audio_input_queue.put(audio_data)
                    
        except Exception as e:
            print(f"Audio callback error: {e}")
    
    def log_audio_info(self):
        """Log audio device information"""
        print("\nAUDIO SYSTEM INFORMATION")
        print("=" * 50)
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            
            print(f"Default Input Device: {devices[default_input]['name']}")
            print(f"Default Output Device: {devices[default_output]['name']}")
            print(f" Sample Rate: {self.sample_rate}Hz")
            print(f" Channels: {self.channels}")
            print(f" Block Size: {self.blocksize}")
            print(f" Data Type: {self.dtype}")
            
        except Exception as e:
            print(f"Error getting audio info: {e}")
        print("=" * 50)
    
    def _show_final_statistics(self):
        """Show final conversation statistics"""
        if self.stats['start_time']:
            duration = time_module.time() - self.stats['start_time']
            print("\n" + "="*60)
            print(" FINAL CONVERSATION STATISTICS")
            print("="*60)
            print(f" Total Duration: {duration:.1f} seconds")
            print(f" Total Exchanges: {self.stats['total_exchanges']}")
            print(f" Successful Responses: {self.stats['successful_responses']}")
            print(f" Barge-in Events: {self.stats['barge_in_events']}")
            print(f" Partial Responses: {self.stats['partial_responses']}")
            if self.stats['total_exchanges'] > 0:
                success_rate = (self.stats['successful_responses'] / self.stats['total_exchanges']) * 100
                print(f" Success Rate: {success_rate:.1f}%")
            if self.stats['barge_in_events'] > 0:
                partial_rate = (self.stats['partial_responses'] / self.stats['barge_in_events']) * 100
                print(f" Barge-in Accuracy: {partial_rate:.1f}% (partial responses tracked)")
            print(f" LLM Provider: {'Groq (BYO)' if USE_BYO_LLM else 'Deepgram'}")
            print(f" Word Tracking: Fast & Accurate (optimized word-by-word barge-in detection)")
            print(f" Performance: ULTRA-FAST (pre-warmed LLM+STT+TTS+VAD pipeline)")
            print("="*60)
    
    async def start_voice_conversation(self):
        """Start the voice conversation with real microphone and speaker"""
        try:
            print("\nStarting Deepgram Voice Agent Demo...")
            print("📝 IMPORTANT: Wait for dial tone before speaking - it signals ALL systems are ready!")
            self._display_configuration_summary()
            
            # Log audio information
            self.log_audio_info()
            
            # Set up voice agent FIRST (including VAD pre-warming)
            print("\nInitializing voice systems...")
            if not await self.setup_voice_agent():
                print("Failed to set up voice agent")
                return
            
            print("\nSetting up audio stream...")
            print("Real-time microphone input and speaker output enabled!")
            if USE_BYO_LLM:
                print("Using Groq as LLM provider through Deepgram's BYO LLM feature")
            else:
                print("Using Deepgram's standard LLM")
            
            # Set audio devices from env vars if provided
            input_device_name = os.getenv("SD_INPUT_DEVICE")
            output_device_name = os.getenv("SD_OUTPUT_DEVICE")
            if input_device_name or output_device_name:
                devices = sd.query_devices()
                input_idx = next((i for i, d in enumerate(devices) if d['name'] == input_device_name), None) if input_device_name else sd.default.device[0]
                output_idx = next((i for i, d in enumerate(devices) if d['name'] == output_device_name), None) if output_device_name else sd.default.device[1]
                if input_idx is not None and output_idx is not None:
                    sd.default.device = [input_idx, output_idx]
                    print(f"Set input: {input_device_name}, output: {output_device_name}")
                else:
                    print("Could not find specified devices - using defaults")

            # Create audio stream using sounddevice
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.blocksize,
                dtype=getattr(np, self.dtype)
            )
            
            # Start the conversation with real audio
            with stream:
                print("\nVoice Agent active - microphone listening...")
                print("🔧 Systems pre-warmed - finalizing audio stream setup...")
                print("⏰ UNLIMITED DURATION - Call continues until manually stopped")
                print("Press Ctrl+C to stop the conversation OR use 'End Call' button in web interface")
                
                # Give audio system a moment to initialize
                await asyncio.sleep(0.5)
                
                # NOW truly ready - signal with dial tone
                print("\n🔔 ALL SYSTEMS READY - Playing dial tone to signal readiness...")
                if not self.test_audio_output():
                    print("⚠️ Audio output test failed, but continuing (check speakers)")
                
                print("✅ SYSTEM FULLY READY - Start speaking immediately!")
                print("🎤 Your first 'Hello' will now get an INSTANT response!")
                
                # Main conversation loop - only ends on manual stop or system shutdown
                while self.is_active:
                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                    
                    # No automatic termination - call continues until manually stopped
            
        except KeyboardInterrupt:
            print("\nConversation interrupted by user")
        except Exception as e:
            print(f"\nError during conversation: {str(e)}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources - only called on manual stop"""
        print("\nCleaning up resources (manual stop initiated)...")
        self.is_active = False
        
        # Stop audio playback and clear buffers
        try:
            with self.audio_lock:
                sd.stop()
                # Close persistent output stream
                if self.output_stream:
                    try:
                        self.output_stream.stop()
                        self.output_stream.close()
                        print("Output stream closed")
                    except Exception as e:
                        print(f"Error closing output stream: {e}")
                self.tts_audio_buffer.clear()
                self.tts_playback_active = False
                self.audio_complete = True
                self.audio_session_active = False
                self.tts_tracker.reset() # Reset word tracker on cleanup
            print("Audio streaming stopped and buffers cleared")
        except Exception as e:
            print(f"Error stopping audio: {e}")
        
        if self.connection:
            try:
                self.connection.finish()
                print("Voice Agent connection closed")
            except Exception as e:
                print(f"Error closing connection: {e}")
        
        # Show final statistics only on manual cleanup
        self._show_final_statistics()
        print("Cleanup completed")

###############################################################################
# Gradio one-shot wrapper
###############################################################################

import io as _io
import numpy as _np
import soundfile as _sf
import requests as _requests

async def run_voice_agent(user_pcm: bytes) -> tuple[str, bytes]:
    """Single-utterance helper for the Gradio demo with word tracking.

    Parameters
    ----------
    user_pcm : bytes
        Raw 16-bit little-endian PCM at **48 kHz, mono** coming from the browser.

    Returns
    -------
    tuple[str, bytes]
        Assistant transcript and **raw** 48 kHz PCM reply.
    """
    # 1) Convert the PCM blob into a WAV buffer for Deepgram STT ----------------
    wav_buf = _io.BytesIO()
    _sf.write(wav_buf, _np.frombuffer(user_pcm, dtype='<i2'), 48000,
              format='WAV', subtype='PCM_16')
    wav_bytes = wav_buf.getvalue()

    # 2) Speech-to-Text with Deepgram (nova-3) ----------------------------------
    dg_client = DeepgramClient(DEEPGRAM_API_KEY)
    stt_opts = {
        "model": "nova-3",
        "punctuate": True,
        "language": "en-US"
    }
    source = {"buffer": wav_bytes, "mimetype": "audio/wav"}
    try:
        stt_resp = await dg_client.transcription.pre_recorded(source, stt_opts)
        transcript = (stt_resp["results"]["channels"][0]["alternatives"][0]
                      ["transcript"]).strip()
    except Exception as exc:
        transcript = "[STT error]"  # fail-safe
        print(f"Deepgram STT failed: {exc}")

    # 3) Generate response with Groq LLM ---------------------------------------
    try:
        # Read prompt path from environment at runtime
        prompt_path = os.getenv("AGENT_PROMPT_PATH", "prompts/system_prompt.txt")
        system_prompt = load_prompt(prompt_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript or ""}
        ]
        groq_resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=LLMConfig.MODEL_NAME,
            messages=messages,
            temperature=0.7
        )
        reply_text = groq_resp.choices[0].message.content.strip()
    except Exception as exc:
        reply_text = "Sorry, I had trouble thinking of a reply."
        print(f"Groq completion failed: {exc}")

    # 4) Text-to-Speech with Groq Play-AI --------------------------------------
    try:
        from groq_tts import synthesize_speech
        wav_bytes = synthesize_speech(
            reply_text,
            voice=os.getenv("AGENT_VOICE_NAME", "Cheyenne-PlayAI"),
        )
        import soundfile as sf, io
        pcm_np, _ = sf.read(io.BytesIO(wav_bytes), dtype="int16")
        reply_pcm = pcm_np.tobytes()
    except Exception as exc:
        reply_pcm = b""
        print(f"Groq TTS failed: {exc}")

    return reply_text, reply_pcm

async def main():
    """Main function"""
    print("Deepgram Voice Agent + Groq Strategic Planning Demo")
    print("=" * 60)
    print("⚡ STARTUP: Please wait for dial tone - it signals complete system readiness")
    print("=" * 60)
    
    # Initialize and start the pipeline
    pipeline = DeepgramVoiceAgentPipeline()
    await pipeline.start_voice_conversation()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo terminated by user")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)