"""
Deepgram Voice Agent Demo - Using Deepgram for voice processing and Groq for strategic planning
Combines Deepgram Voice Agent API for all voice handling with Groq for conversation strategy
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

# Import our conversation planner and configuration
from conversation_planner import ConversationPlanner
from config_defaults import (
    AudioConfig, PhoneCallConfig, LLMConfig, PlanningConfig, DebugConfig, TTSConfig,
    get_configuration_summary
)

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

# Path to the authoritative system prompt that drives every demo
PROMPT_PATH = "prompts/system_prompt.txt"  # single source of truth for all demos

def load_prompt(path: str) -> str:
    """Load and return the full system prompt from the given path.

    This helper purposely *does not* provide a fallback string: if the file
    is missing we want to fail fast rather than run the demo with an empty or
    generic prompt which would lead to confusing "robotic" responses.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"System prompt file not found at '{path}'. Please ensure the file "
            "is present before starting the demo."
        ) from exc

class ConversationHistoryProvider:
    """Provides conversation history to the planner"""
    
    def __init__(self):
        self.history = []
    
    async def get_conversation_history(self) -> List[Dict]:
        return self.history.copy()
    
    def update_history(self, history: List[Dict]):
        self.history = history.copy()

class DeepgramVoiceAgentPipeline:
    """Main pipeline combining Deepgram Voice Agent with Groq strategic planning"""
    
    def __init__(self):
        """Initialize the voice agent pipeline"""
        print("🚀 Initializing Deepgram Voice Agent Pipeline...")
        
        # Initialize APIs
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Initialize Deepgram client
        config = DeepgramClientOptions(
            options={
                "keepalive": "true",
            }
        )
        self.deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
        self.connection = None
        
        # Audio configuration (using existing configuration from other demos)
        # Override defaults to match Deepgram Voice-Agent Playground (48 kHz, 10 ms frames)
        self.sample_rate = 48000           # 48 kHz end-to-end for highest fidelity
        self.channels = 1                  # Mono
        self.blocksize = 480              # 10 ms frames at 48 kHz (480 samples)
        self.dtype = AudioConfig.DTYPE
        
        # Audio stream and processing
        self.audio_input_queue = queue.Queue()
        self.audio_output_queue = queue.Queue()
        self.audio_stream = None
        self.audio_chunk_counter = 0
        
        # TTS audio buffering for smooth streaming playback (bounded deque for O(1) pops)
        # Provide ~8 s head-room (800 × 10 ms) to avoid rollover on long responses.
        self.tts_audio_buffer = deque(maxlen=1600)
        self.tts_playback_active = False
        self.audio_lock = threading.Lock()
        self.audio_complete = False
        # Playback buffering parameters (tunable via env-vars for quick A/B tests)
        # Wait until we have ≈250 ms of audio (24 × 10 ms) before starting playback.
        self.stream_start_threshold = int(os.getenv("DG_START_THRESHOLD", 24))
        # Feed PortAudio smaller bursts (20 ms by default) to reduce underruns.
        self.write_chunks = int(os.getenv("DG_WRITE_CHUNKS", 2))
        self.audio_session_active = False
        self.barge_in_dropped = False   # Flag to defer buffer flush until streamer handles it
        
        # Persistent PortAudio output stream to avoid device spin-up latency
        try:
            # Use RawOutputStream for direct bytes write (no numpy conversion)
            self.output_stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=0,      # Let PortAudio choose optimal
                latency='low'
            )
            self.output_stream.start()
            print("🔊 Output stream started (persistent)")
        except Exception as e:
            self.output_stream = None
            print(f"⚠️ Could not start persistent output stream: {e}")
        
        # Initialize conversation components
        self.history_provider = ConversationHistoryProvider()
        self.planner = ConversationPlanner("cold_call")
        
        # Load system prompt (single source-of-truth for both LLM and Agent)
        self.system_prompt = load_prompt(PROMPT_PATH)
        
        # Conversation state
        self.conversation_history = []
        self.is_active = False
        self.planning_context = ""
        self.current_strategy = ""
        
        # Statistics
        self.stats = {
            'total_exchanges': 0,
            'successful_responses': 0,
            'planning_updates': 0,
            'start_time': None
        }
        
        # Strategic planning control
        self.planning_hash = ""        # Debounce identical prompt updates
        self.agent_speaking = False     # Track whether agent is currently speaking
        self.main_loop = None           # Will be set when starting the voice conversation
        
        print("✅ Pipeline initialized successfully")
    
    def _display_configuration_summary(self):
        """Display current configuration"""
        print("\n" + "="*80)
        print("🔧 DEEPGRAM VOICE AGENT CONFIGURATION")
        print("="*80)
        print(f"📞 Call Type: Cold Call Sales Demo")
        print(f"🎯 Call Objective: Introduce AI solutions and qualify prospects")
        print(f"⏱️  Sample Rate: {self.sample_rate}Hz")
        print(f"🧠 LLM Model: {LLMConfig.MODEL_NAME}")
        print(f"🎙️  Voice Agent: Deepgram Nova-3 + Aura-2")
        print(f"📋 Planning Type: {PlanningConfig.PLANNING_TYPE}")
        print(f"🔊 TTS Voice: {TTSConfig.VOICE_NAME}")
        print(f"🐛 Debug Mode: {DebugConfig.DEBUG_LOGGING}")
        print("="*80)
    
    async def setup_voice_agent(self) -> bool:
        """Set up the Deepgram Voice Agent connection and configuration"""
        try:
            print("🔌 Setting up Deepgram Voice Agent connection...")
            
            # Create WebSocket connection
            self.connection = self.deepgram.agent.websocket.v("1")
            
            # Configure agent settings
            options = SettingsOptions()
            
            # Audio configuration for microphone input (16kHz PCM)
            options.audio.input.encoding = "linear16"
            options.audio.input.sample_rate = 48000  # Match device & server
            options.audio.output.encoding = "linear16"
            options.audio.output.sample_rate = 48000
            options.audio.output.container = "none"
            
            # Agent configuration - using our planning for the LLM
            options.agent.language = "en"
            
            # STT configuration (Deepgram)
            options.agent.listen.provider.type = "deepgram"
            options.agent.listen.provider.model = "nova-3"
            options.agent.listen.provider.keyterms = ["hello", "goodbye", "interested", "not interested", "tell me more"]
            
            # Extend silence timeout so Deepgram doesn't hang up during long pauses
            try:
                options.agent.listen.config.silence_timeout_ms = 15000  # 15 seconds
            except AttributeError:
                try:
                    options.agent.listen.config.silence_timeout = 15000
                except Exception:
                    pass
            
            # LLM configuration (we'll inject strategic context)
            options.agent.think.provider.type = "open_ai"  # Deepgram supports OpenAI-compatible APIs
            options.agent.think.provider.model = "gpt-4o-mini"
            options.agent.think.provider.temperature = 0.7
            
            # TTS configuration (Deepgram Aura)
            options.agent.speak.provider.type = "deepgram"
            options.agent.speak.provider.model = "aura-2-cora-en"  # Natural, neutral tone
            
            # Normal playback speed; artefacts are avoided once prompt is correct
            try:
                options.agent.speak.voice = {"rate": 1.0}
            except AttributeError:
                # Fallback for SDK versions that expose rate directly
                try:
                    options.agent.speak.voice.rate = 1.0
                except Exception:
                    pass
            
            # Initial system prompt for the LLM that drives the Agent
            try:
                options.agent.think.prompt = self.system_prompt
            except AttributeError:
                # Legacy SDK path
                options.agent.think.config.prompt = self.system_prompt
            options.agent.greeting = "Hi I'm **Alex calling from Remember Church Directories**, I know this is unexpected—do you have 30 seconds for me to explain why I called?"
            
            # Register event handlers
            self._register_event_handlers()
            
            # Start the connection
            print("🔄 Starting Voice Agent connection...")
            if not self.connection.start(options):
                print("❌ Failed to start Voice Agent connection")
                return False
            
            print("✅ Voice Agent connection established successfully")
            
            # Start keep-alive thread
            self._start_keep_alive()
            
            # Start strategic planning thread
            self._start_strategic_planning()
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up Voice Agent: {str(e)}")
            return False
    
    def _build_initial_prompt(self) -> str:
        """Return the raw system prompt (kept for backward-compatibility)."""
        return self.system_prompt
    
    def _register_event_handlers(self):
        """Register event handlers for the Voice Agent"""
        print("📋 Registering Voice Agent event handlers...")
        
        # Register all event handlers with the connection
        self.connection.on(AgentWebSocketEvents.Welcome, self._on_welcome)
        self.connection.on(AgentWebSocketEvents.SettingsApplied, self._on_settings_applied)
        self.connection.on(AgentWebSocketEvents.ConversationText, self._on_conversation_text)
        self.connection.on(AgentWebSocketEvents.UserStartedSpeaking, self._on_user_started_speaking)
        self.connection.on(AgentWebSocketEvents.AgentThinking, self._on_agent_thinking)
        self.connection.on(AgentWebSocketEvents.AgentStartedSpeaking, self._on_agent_started_speaking)
        self.connection.on(AgentWebSocketEvents.AudioData, self._on_audio_data)  # This should receive TTS audio
        self.connection.on(AgentWebSocketEvents.AgentAudioDone, self._on_agent_audio_done)
        self.connection.on(AgentWebSocketEvents.Close, self._on_close)
        self.connection.on(AgentWebSocketEvents.Error, self._on_error)
        self.connection.on(AgentWebSocketEvents.Unhandled, self._on_unhandled)
        
        # Handle prompt update acknowledgements (Deepgram emits "PromptUpdated")
        try:
            self.connection.on(AgentWebSocketEvents.PromptUpdated, lambda *_: logger.debug("📩 Prompt update acknowledged"))
        except AttributeError:
            # Fallback for SDKs exposing the raw string
            self.connection.on("PromptUpdated", lambda *_: logger.debug("📩 Prompt update acknowledged"))
        
        print("✅ Event handlers registered")
    
    # Event Handler Methods
    def _on_welcome(self, connection, welcome, **kwargs):
        """Handle welcome message"""
        print(f"👋 Voice Agent Welcome: {welcome}")
        self.is_active = True
        self.stats['start_time'] = time_module.time()
        
        # Reset audio state for new conversation
        with self.audio_lock:
            self.tts_audio_buffer.clear()
            self.tts_playback_active = False
            self.audio_complete = False
            self.audio_session_active = False
    
    def _on_settings_applied(self, connection, settings_applied, **kwargs):
        """Handle settings applied"""
        print(f"⚙️ Settings Applied: {settings_applied}")
    
    def _on_conversation_text(self, connection, conversation_text, **kwargs):
        """Handle conversation text updates"""
        try:
            print(f"💬 Conversation Text Event: {conversation_text}")
            
            # Try to extract meaningful conversation data
            if conversation_text:
                # Handle different possible formats
                if hasattr(conversation_text, 'role') and hasattr(conversation_text, 'content'):
                    role = conversation_text.role
                    content = conversation_text.content
                    
                    # Add to conversation history
                    self.conversation_history.append({
                        'role': role,
                        'content': content,
                        'timestamp': time_module.time()
                    })
                    
                    # Update history provider
                    self.history_provider.update_history(self.conversation_history)
                    
                    # Update statistics
                    if role == 'assistant':
                        self.stats['successful_responses'] += 1
                        self.stats['total_exchanges'] += 1
                    
                    # Log conversation progress
                    print(f"📝 {role.upper()}: {content}")
                elif hasattr(conversation_text, 'text'):
                    # Handle text-only format
                    text = conversation_text.text
                    print(f"📝 DETECTED SPEECH: {text}")
                else:
                    # Debug: show the actual object structure
                    print(f"🔍 Raw conversation object: {type(conversation_text)}")
                    if hasattr(conversation_text, '__dict__'):
                        print(f"🔍 Object data: {conversation_text.__dict__}")
                
        except Exception as e:
            print(f"⚠️ Error processing conversation text: {e}")
    
    def _on_user_started_speaking(self, connection, user_started_speaking, **kwargs):
        """Handle user started speaking"""
        print(f"🎤 User started speaking! Args: {user_started_speaking}")
        print("✅ Voice detected - audio processing is working!")
        
        # Barge-in: discard any pending TTS so the assistant stops instantly
        with self.audio_lock:
            if len(self.tts_audio_buffer) > 0:
                # Defer the actual flush to the streaming thread to avoid
                # accidentally dropping packets that belong to the *next* sentence.
                self.barge_in_dropped = True
                print("🚫 Barge-in detected – buffer will be flushed gracefully")
    
    def _on_agent_thinking(self, connection, agent_thinking, **kwargs):
        """Handle agent thinking"""
        print(f"🤔 Agent is thinking... Args: {agent_thinking}")
    
    def _on_agent_started_speaking(self, connection, agent_started_speaking, **kwargs):
        """Handle agent started speaking"""
        print(f"🗣️ Agent started speaking! Args: {agent_started_speaking}")
        
        # Reset audio state for new TTS session
        with self.audio_lock:
            # Always reset for a new TTS session
            self.tts_audio_buffer.clear()
            self.tts_playback_active = False  # Reset playback state
            self.audio_complete = False
            self.audio_session_active = True
            
            # Mark that the agent is currently speaking (used to debounce prompt updates)
            self.agent_speaking = True
        
        print("🎵 New TTS session started - audio state reset for fresh streaming")
    
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
                # Add to buffer
                with self.audio_lock:
                    # Auto-detect new TTS session: if we receive audio but no session is active,
                    # this must be a new response that didn't trigger AgentStartedSpeaking
                    if not self.audio_session_active and not self.tts_playback_active:
                        print("🎵 🔍 Auto-detected new TTS session - resetting audio state")
                        self.tts_audio_buffer.clear()  # Clear any stale data
                        self.audio_complete = False
                        self.audio_session_active = True
                    
                    self.tts_audio_buffer.append(audio_data)
                    buffer_length = len(self.tts_audio_buffer)
                    
                    # Debug: Show first few chunks only
                    if buffer_length <= 3:
                        print(f"🎵 Audio chunk {buffer_length}: session_active={self.audio_session_active}, playback_active={self.tts_playback_active}")
                    
                    # Start streaming after we have enough chunks for smooth playback
                    if buffer_length == self.stream_start_threshold and not self.tts_playback_active:
                        self.tts_playback_active = True
                        self.audio_session_active = True  # Mark session as active when we start streaming
                        print(f"🎵 ✅ STARTING streaming playback (buffered {buffer_length} chunks)")
                        
                        # Start streaming thread
                        streaming_thread = threading.Thread(target=self._stream_audio_chunks, daemon=True)
                        streaming_thread.start()
                    
                    # Continue adding chunks for the streaming thread to consume
                    elif self.tts_playback_active and buffer_length % 50 == 0:
                        print(f"🎵 Streaming... {buffer_length} chunks received")
                    
                # Optional: Save for debugging (first few chunks only)
                if DebugConfig.DEBUG_LOGGING and len(self.tts_audio_buffer) <= 3:
                    self._save_debug_audio(audio_data, f"tts_chunk_{len(self.tts_audio_buffer)}")
                
                return True
            else:
                print(f"❌ No valid audio data found")
                    
        except Exception as e:
            print(f"❌ Error handling audio data: {e}")
            import traceback
            traceback.print_exc()
        
        return False
    
    def _stream_audio_chunks(self):
        """Stream audio chunks using persistent RawOutputStream to minimise latency"""
        try:
            print("🎵 Streaming thread active (RawOutputStream mode)...")
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
                        print(f"⚠️ PortAudio write error/underrun: {e}")
                else:
                    # No data ready; yield briefly
                    time_module.sleep(0.002)
        except Exception as e:
            print(f"❌ Error in RawOutputStream streaming: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self.audio_lock:
                self.tts_playback_active = False
                self.audio_session_active = False
            print("🔇 Audio streaming session ended (RawOutputStream)")
    
    def _on_agent_audio_done(self, connection, agent_audio_done, **kwargs):
        """Handle agent audio done - signal end of streaming session"""
        print(f"🔇 Agent finished speaking - ending streaming session")
        
        # Mark session as complete to let streaming thread finish remaining chunks
        with self.audio_lock:
            self.audio_complete = True  # Let the streaming thread drain remaining audio
            chunk_count = len(self.tts_audio_buffer)
            
        # Agent finished speaking
        self.agent_speaking = False
        
        # Trigger an immediate planning refresh so the next turn reflects latest state
        try:
            if self.main_loop and not self.main_loop.is_closed():
                asyncio.run_coroutine_threadsafe(self._update_strategic_context(), self.main_loop)
        except Exception as e:
            print(f"⚠️ Immediate planning update failed: {e}")
        
        print(f"📝 Session ended with {chunk_count} total chunks - streaming will finish naturally")
    
    def _on_close(self, connection, close, **kwargs):
        """Handle connection close"""
        print(f"🔚 Connection closed: {close}")
        self.is_active = False
        self._show_final_statistics()
    
    def _on_error(self, connection, error, **kwargs):
        """Handle errors"""
        print(f"❌ Voice Agent Error: {error}")
    
    def _on_unhandled(self, connection, unhandled, **kwargs):
        """Handle unhandled events"""
        print(f"❓ Unhandled event: {unhandled}")
    
    def _start_keep_alive(self):
        """Start the keep-alive thread"""
        def keep_alive_loop():
            while self.is_active:
                try:
                    time_module.sleep(5)
                    if self.connection:
                        self.connection.send(str(AgentKeepAlive()))
                        print("💓 Keep-alive sent")
                except Exception as e:
                    print(f"⚠️ Keep-alive error: {e}")
        
        keep_alive_thread = threading.Thread(target=keep_alive_loop, daemon=True)
        keep_alive_thread.start()
        print("💓 Keep-alive thread started")
    
    def _start_strategic_planning(self):
        """Start the strategic planning thread"""
        def planning_loop():
            while self.is_active:
                try:
                    # Use planner's configured interval (default 3s) but never below 2s
                    interval = max(self.planner.planning_interval, 2.0)
                    time_module.sleep(interval)
                    if len(self.conversation_history) > 0:
                        # Schedule coroutine on the main event loop to avoid nested asyncio.run()
                        if self.main_loop and not self.main_loop.is_closed():
                            asyncio.run_coroutine_threadsafe(self._update_strategic_context(), self.main_loop)
                        else:
                            print("⚠️ Main loop not available for strategic planning update")
                except Exception as e:
                    print(f"⚠️ Planning error: {e}")
        
        planning_thread = threading.Thread(target=planning_loop, daemon=True)
        planning_thread.start()
        print("🧠 Strategic planning thread started")
    
    async def _update_strategic_context(self):
        """Update the strategic context based on conversation progress"""
        try:
            # Analyze conversation first
            await self.planner.analyze_conversation(self.conversation_history)
            
            # Get strategic guidance from planner
            guidance = self.planner.get_current_guidance()
            
            if guidance and guidance.get('recommended_action'):
                new_context = self._build_strategic_context(guidance)
                
                # Debounce: only send if context actually changed AND agent isn't speaking
                new_hash = hash(new_context)
                if new_hash != self.planning_hash and not self.agent_speaking:
                    self.planning_context = new_context
                    self.current_strategy = guidance.get('current_objective', '')
                    
                    await self._inject_strategic_update(new_context)
                    self.planning_hash = new_hash
                    self.stats['planning_updates'] += 1
                    print(f"🧠 Strategic context updated: {guidance.get('current_phase', 'Continuing conversation')}")
                    
        except Exception as e:
            print(f"⚠️ Error updating strategic context: {e}")
    
    def _build_strategic_context(self, guidance: Dict[str, Any]) -> str:
        """Build strategic context from guidance"""
        context_parts = []
        
        # Prioritise addressing critical obstacles immediately
        critical_obstacle = None
        for obs in guidance.get('active_obstacles', []):
            if obs.get('severity', 0) >= 7:  # High-severity threshold
                critical_obstacle = obs
                break

        if critical_obstacle:
            context_parts.append(
                f"NEXT ACTION: {critical_obstacle['strategy']} (to address: {critical_obstacle['description']})"
            )
        elif guidance.get('recommended_action'):
            context_parts.append(f"NEXT ACTION: {guidance['recommended_action']}")
        
        if guidance.get('current_objective'):
            context_parts.append(f"CURRENT OBJECTIVE: {guidance['current_objective']}")
        
        if guidance.get('current_phase'):
            context_parts.append(f"CONVERSATION PHASE: {guidance['current_phase']}")
        
        # Add active obstacles if any
        if guidance.get('active_obstacles'):
            for obstacle in guidance['active_obstacles']:
                context_parts.append(
                    f"OBSTACLE: {obstacle['description']} - STRATEGY: {obstacle['strategy']} (severity {obstacle['severity']})"
                )
        
        # Add next tasks
        if guidance.get('next_tasks'):
            next_task = guidance['next_tasks'][0]  # Get the first/highest priority task
            context_parts.append(f"FOCUS ON: {next_task['name']} - {next_task['description']}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    async def _inject_strategic_update(self, strategic_context: str):
        """Inject strategic update into the voice agent"""
        try:
            if not self.connection:
                return
            
            # Keep the base system prompt and append the planner's current guidance
            updated_prompt = f"{self.system_prompt}\n\nCURRENT STRATEGIC CONTEXT:\n{strategic_context}"
            
            # Create an update prompt message
            update_message = {
                "type": "UpdatePrompt",
                "prompt": updated_prompt
            }
            
            # Send the strategic context update
            self.connection.send(json.dumps(update_message))
            print(f"📤 Strategic update injected into voice agent")
            
        except Exception as e:
            print(f"⚠️ Error injecting strategic update: {e}")
    
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
            
            print(f"🐛 Debug audio saved: {filename} ({len(audio_data)} bytes)")
            
        except Exception as e:
            print(f"⚠️ Could not save debug audio: {e}")
    
    def test_audio_output(self):
        """Test audio output with a simple tone"""
        try:
            print("🔊 Testing audio output with a 440Hz tone...")
            
            # Generate a 1-second 440Hz sine wave
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
            print("✅ Audio output test completed - you should have heard a tone")
            
        except Exception as e:
            print(f"❌ Audio output test failed: {e}")
            return False
        
        return True
    
    def audio_callback(self, indata, frames, time, status):
        """Audio callback for sounddevice - captures microphone input"""
        if status:
            print(f"⚠️ Audio callback status: {status}")
        
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
                
                # Debug: Show that we're sending audio (every 100 chunks)
                self.audio_chunk_counter += 1
                if self.audio_chunk_counter % 100 == 0:
                    print(f"🎤 Audio flowing: {self.audio_chunk_counter} chunks sent ({len(audio_data)} bytes each)")
                
                # Optional: Add to queue for local processing if needed
                if not self.audio_input_queue.full():
                    self.audio_input_queue.put(audio_data)
                    
        except Exception as e:
            print(f"❌ Audio callback error: {e}")
    
    def log_audio_info(self):
        """Log audio device information"""
        print("\n🔊 AUDIO SYSTEM INFORMATION")
        print("=" * 50)
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            
            print(f"🎤 Default Input Device: {devices[default_input]['name']}")
            print(f"🔊 Default Output Device: {devices[default_output]['name']}")
            print(f"📊 Sample Rate: {self.sample_rate}Hz")
            print(f"📺 Channels: {self.channels}")
            print(f"📦 Block Size: {self.blocksize}")
            print(f"🔢 Data Type: {self.dtype}")
            
        except Exception as e:
            print(f"❌ Error getting audio info: {e}")
        print("=" * 50)
    
    def _show_final_statistics(self):
        """Show final conversation statistics"""
        if self.stats['start_time']:
            duration = time_module.time() - self.stats['start_time']
            print("\n" + "="*60)
            print("📊 FINAL CONVERSATION STATISTICS")
            print("="*60)
            print(f"⏱️ Total Duration: {duration:.1f} seconds")
            print(f"💬 Total Exchanges: {self.stats['total_exchanges']}")
            print(f"✅ Successful Responses: {self.stats['successful_responses']}")
            print(f"🧠 Strategic Updates: {self.stats['planning_updates']}")
            if self.stats['total_exchanges'] > 0:
                success_rate = (self.stats['successful_responses'] / self.stats['total_exchanges']) * 100
                print(f"📈 Success Rate: {success_rate:.1f}%")
            print(f"🎯 Final Strategy: {self.current_strategy}")
            print("="*60)
    
    async def start_voice_conversation(self):
        """Start the voice conversation with real microphone and speaker"""
        try:
            # Set the main loop for this thread
            self.main_loop = asyncio.get_event_loop()
            
            print("\n🎬 Starting Deepgram Voice Agent Demo...")
            self._display_configuration_summary()
            
            # Log audio information
            self.log_audio_info()
            
            # Test audio output first
            print("\n🔊 Testing audio output before starting...")
            if not self.test_audio_output():
                print("❌ Audio output test failed. Please check your speakers.")
                return
            print("✅ Audio output working correctly!")
            
            # Set up voice agent
            if not await self.setup_voice_agent():
                print("❌ Failed to set up voice agent")
                return
            
            print("\n🎙️ Setting up audio stream...")
            print("📞 Real-time microphone input and speaker output enabled!")
            print("💡 Deepgram handles all voice processing while Groq provides strategic guidance")
            print("🔄 Strategic context will be updated automatically during the conversation")
            
            # Create audio stream using sounddevice (same as existing demos)
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.blocksize,
                dtype=getattr(np, self.dtype)
            )
            
            # Start the conversation with real audio
            with stream:
                print("\n🎯 Voice Agent active - microphone listening...")
                print("🗣️ Start speaking! The AI will respond through your speakers.")
                print("🛑 Press Ctrl+C to stop the conversation")
                
                # Give audio system a moment to initialize
                await asyncio.sleep(0.5)
                
                # Main conversation loop
                while self.is_active:
                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                    
                    # Check if we should end the call based on duration (10 minutes default)
                    max_duration_seconds = 600  # 10 minutes default
                    if (self.stats['start_time'] and 
                        time_module.time() - self.stats['start_time'] > max_duration_seconds):
                        print(f"\n⏰ Maximum call duration reached ({max_duration_seconds/60:.0f} minutes)")
                        break
            
        except KeyboardInterrupt:
            print("\n\n🛑 Conversation interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during conversation: {str(e)}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up resources...")
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
                        print("🔊 Output stream closed")
                    except Exception as e:
                        print(f"⚠️ Error closing output stream: {e}")
                self.tts_audio_buffer.clear()
                self.tts_playback_active = False
                self.audio_complete = True
                self.audio_session_active = False
            print("🔇 Audio streaming stopped and buffers cleared")
        except Exception as e:
            print(f"⚠️ Error stopping audio: {e}")
        
        if self.connection:
            try:
                self.connection.finish()
                print("✅ Voice Agent connection closed")
            except Exception as e:
                print(f"⚠️ Error closing connection: {e}")
        
        print("✅ Cleanup completed")

###############################################################################
# Gradio one-shot wrapper
###############################################################################

import io as _io
import numpy as _np
import soundfile as _sf
import requests as _requests

async def run_voice_agent(user_pcm: bytes) -> tuple[str, bytes]:
    """Single-utterance helper for the Gradio demo.

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
        print(f"❌ Deepgram STT failed: {exc}")

    # 3) Generate response with Groq LLM ---------------------------------------
    try:
        system_prompt = load_prompt(PROMPT_PATH)
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
        print(f"❌ Groq completion failed: {exc}")

    # 4) Text-to-Speech with Deepgram Aura-2 -----------------------------------
    try:
        tts_url = "https://api.deepgram.com/v1/speak"
        params = {
            "model": "aura-2-cora-en",
            "encoding": "linear16",
            "sample_rate": "48000"
        }
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        tts_payload = {"text": reply_text}
        loop = asyncio.get_event_loop()
        # Run blocking request in a thread so we don't block the event loop
        def _blocking_tts():
            resp = _requests.post(tts_url, params=params, headers=headers, json=tts_payload)
            resp.raise_for_status()
            return resp.content
        reply_pcm = await loop.run_in_executor(None, _blocking_tts)
    except Exception as exc:
        reply_pcm = b""  # empty audio
        print(f"❌ Deepgram TTS failed: {exc}")

    return reply_text, reply_pcm

async def main():
    """Main function"""
    print("🚀 Deepgram Voice Agent + Groq Strategic Planning Demo")
    print("=" * 60)
    
    # Initialize and start the pipeline
    pipeline = DeepgramVoiceAgentPipeline()
    await pipeline.start_voice_conversation()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo terminated by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1) 