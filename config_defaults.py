"""
Centralized Configuration for AI Cold Calls Voice Pipeline
=========================================================

All magic numbers, thresholds, and configuration values in one place.
Easier audits, faster experiments, fewer "forgotten constant" bugs.

Environment variables override these defaults for runtime tuning.
"""

import os
from typing import Dict, Any, List

# =============================================================================
# AUDIO SYSTEM CONFIGURATION
# =============================================================================

class AudioConfig:
    """Audio system settings for optimal performance"""
    
    # Core audio parameters
    SAMPLE_RATE = 16000          # 16kHz standard for speech recognition
    CHANNELS = 1                 # Mono audio for speech
    BLOCKSIZE = 256             # 16ms frames (256 samples at 16kHz) - ultra-low latency
    DTYPE = 'int16'             # Direct int16 processing (no float conversion)
    
    # Frame timing calculations
    FRAME_DURATION_MS = (BLOCKSIZE / SAMPLE_RATE) * 1000  # 16ms per frame
    BYTES_PER_FRAME = BLOCKSIZE * 2  # 512 bytes per frame (int16 = 2 bytes/sample)
    
    # Audio stream management
    AUDIO_CLEANUP_WAIT = 0.1    # 100ms wait for audio system cleanup
    STREAM_STOP_WAIT = 0.05     # 50ms wait after stopping streams


# =============================================================================
# VOICE ACTIVITY DETECTION (VAD) CONFIGURATION
# =============================================================================

class VADConfig:
    """Voice Activity Detection parameters - field-tunable via environment variables"""
    
    # Environment-controlled VAD settings (balanced to prevent echo while allowing natural speech)
    STRICTNESS = int(os.getenv("VAD_STRICTNESS", "2"))           # 0-5 scale, 2=relaxed (reduced from 4 to allow natural speech onset)
    WEBRTC_SENSITIVITY = int(os.getenv("VAD_WEBRTC_SENSITIVITY", "2"))  # 0-3, 2=aggressive (reduced from 3 due to compatibility issues)
    NOISE_MULTIPLIER = float(os.getenv("VAD_NOISE_MULTIPLIER", "0.6"))  # 0.5-2.0 scaling (reduced from 0.8 for better tolerance)
    DEBUG_ENABLED = os.getenv("VAD_DEBUG", "true").lower() == "true"   # Debug logging
    
    # Two-stage VAD gating (WebRTC + Energy + Spectral)
    ENABLE_TWO_STAGE_VAD = False           # Enable advanced two-stage gating
    STAGE1_WEBRTC_AGGRESSIVE = 0          # Stage 1: Most aggressive WebRTC VAD
    STAGE2_ENABLE_ENERGY_GATE = True      # Stage 2: Energy gate
    STAGE2_ENABLE_CORRELATION_REJECT = True  # Stage 2: Correlation rejection
    STAGE2_ENABLE_SPECTRAL_GATE = True    # Stage 2: Spectral flatness gate
    
    # Dynamic threshold adaptation
    ENABLE_DYNAMIC_THRESHOLDS = True      # Adapt thresholds to noise floor changes
    NOISE_FLOOR_DRIFT_THRESHOLD = 5.0     # dB change to trigger adaptation
    THRESHOLD_ADAPTATION_RATE = 0.15      # How fast to adapt (15% per update)
    
    # VAD strictness level descriptions
    STRICTNESS_DESCRIPTIONS = {
        0: "Very Lenient (whispers)",
        1: "Lenient (soft speech)",
        2: "Relaxed",
        3: "Balanced (default)",
        4: "Strict",
        5: "Very Strict (noisy env)"
    }
    
    # WebRTC VAD sensitivity descriptions  
    WEBRTC_DESCRIPTIONS = {
        0: "Least aggressive",
        1: "Moderate",
        2: "Aggressive",
        3: "Most aggressive"
    }
    
    # Strictness level to multiplier mapping (enhanced for echo prevention)
    STRICTNESS_MULTIPLIERS = {
        0: (1.5, 1.2),  # (amplitude_multiplier, rms_multiplier)
        1: (2.0, 1.5),
        2: (2.5, 2.0),
        3: (3.0, 2.5),  # Default balanced settings
        4: (3.5, 3.0),  # Strict - new default for echo prevention
        5: (4.0, 3.5),  # Very strict for very noisy environments
    }
    
    # WebRTC VAD frame requirements
    WEBRTC_FRAME_SIZE = 320     # 20ms at 16kHz (160 samples * 2 bytes)
    WEBRTC_MIN_FRAME_SIZE = 160 # Minimum 10ms worth of data
    
    # Safety limits for extreme environments
    MAX_NOISE_FLOOR_AMPLITUDE = 1000   # Prevent excessive thresholds
    MAX_NOISE_FLOOR_RMS = 400          # Prevent excessive thresholds


# =============================================================================
# VOICE DETECTION AND PROCESSING TIMING
# =============================================================================

class TimingConfig:
    """Timing parameters for voice detection and processing"""
    
    # Initial calibration phase timeouts (made more responsive - closer to post-calibration)
    INITIAL_MAX_SILENCE = 0.35           # 350ms silence detection (was 400ms - now closer to post-cal)
    INITIAL_MIN_SPEECH_DURATION = 0.12   # 120ms minimum speech (was 150ms - now matches post-cal)
    INITIAL_MAX_LISTENING_TIME = 8.0     # 8s maximum listening time
    INITIAL_CHECK_INTERVAL = 0.013       # 13ms polling interval (was 15ms - much closer to 12ms post-cal)
    INITIAL_PREDICTIVE_SILENCE = 0.23    # 230ms predictive end-of-speech (was 250ms - closer to 220ms)
    
    # Post-calibration optimized timeouts (adjusted for better end-of-speech detection)
    POST_CALIBRATION_MAX_SILENCE = 0.8         # 800ms silence for proper end-of-speech detection
    POST_CALIBRATION_MIN_SPEECH_DURATION = 0.12 # 120ms minimum (same)
    POST_CALIBRATION_MAX_LISTENING_TIME = 6.0   # 6s maximum (same)
    POST_CALIBRATION_CHECK_INTERVAL = 0.012     # 12ms polling (same)
    POST_CALIBRATION_PREDICTIVE_SILENCE = 0.6   # 600ms predictive for natural pauses
    
    # Noise floor measurement
    NOISE_FLOOR_DURATION = 1.0          # 1 second silence measurement
    NOISE_FLOOR_CHECK_INTERVAL = 0.02   # 20ms for precise noise measurement
    NOISE_FLOOR_PERCENTILE = 0.95       # Use 95th percentile to ignore spikes
    
    # Dynamic threshold adaptation
    NOISE_UPDATE_INTERVAL = 0.5         # Update noise floor every 500ms
    NOISE_BUFFER_SIZE = 100             # Keep last 100 samples (~2 seconds)
    ADAPTATION_RATE = 0.1               # 10% adaptation rate per update
    ENVIRONMENT_STABILITY_THRESHOLD = 0.3 # Threshold for environment changes
    
    # Updated for streaming mode
    STREAMING_CHECK_INTERVAL = float(os.getenv("STREAMING_CHECK_INTERVAL", "0.008"))  # 8ms polling for tighter VAD


# =============================================================================
# INTERRUPTION AND BARGE-IN SYSTEM
# =============================================================================

class InterruptionConfig:
    """Settings for ultra-responsive interruption handling"""
    
    # Grace period and detection timing (reduced for faster response throughout call)
    GRACE_PERIOD = 1.2                  # 1.2s grace period after AI starts speaking (was 1.5s)
    MIN_INTERRUPTION_DURATION = 0.08    # 80ms minimum to confirm interruption (was 100ms - faster)
    INTERRUPTION_CHAIN_WINDOW = 1.5     # 1.5s window to allow continuation
    
    # Audio size requirements for interruption processing (stricter to reject AI echo)
    MIN_INTERRUPTION_AUDIO_SIZE = 8192  # 16 chunks minimum (was calculated from duration)
    MIN_INTERRUPTION_CHUNKS = 6         # Minimum chunks to consider as real interruption (reduced from 8 - faster)
    
    # TTS monitoring during playback
    TTS_MONITORING_INTERVAL = 0.01     # 10ms polling during TTS for instant detection
    BARGE_IN_RESPONSIVENESS = True     # Enable ultra-fast barge-in detection


# =============================================================================
# PERSONALIZED VOICE CALIBRATION
# =============================================================================

class PersonalizationConfig:
    """Personalized voice range and caller profile settings"""
    
    # Voice range expansion for personalization
    PERSONALIZED_RANGE_EXPANSION = 0.30    # ±30% of caller's baseline
    CONFIDENCE_BOOST_FACTOR = 1.3          # 30% boost for known voice patterns
    
    # Personalized detection multipliers (more sensitive for known callers)
    PERSONALIZED_AMP_MULTIPLIER = 2.0      # Lower than default 3.0x
    PERSONALIZED_RMS_MULTIPLIER = 1.8      # Lower than default 2.5x
    
    # Emergency adaptation for connection issues
    EMERGENCY_RANGE_EXPANSION_MAX = 2.0    # Maximum 200% expansion
    EMERGENCY_EXPANSION_RATE = 1.5         # 50% increase per adaptation
    
    # Voice profile calibration
    MIN_CALIBRATION_SAMPLES = 2            # Minimum samples for basic profile
    CALIBRATION_UPDATE_FREQUENCY = 5       # Show status every 5 successful transcriptions


# =============================================================================
# ASR (AUTOMATIC SPEECH RECOGNITION) CONFIGURATION
# =============================================================================

class ASRConfig:
    """Auto-fallback ASR system configuration"""
    
    # ASR model definitions with performance characteristics
    MODELS = [
        {
            'name': 'whisper-large-v3-turbo',
            'description': 'Primary - Balanced speed/accuracy',
            'multilingual': True,
            'wer': 12.0,        # Word Error Rate percentage
            'speed_factor': 247, # Relative speed factor
            'cost_per_hour': 0.04  # USD per hour
        },
        {
            'name': 'distil-whisper',
            'description': 'Fallback - Ultra-fast English-only',
            'multilingual': False,
            'wer': 13.0,
            'speed_factor': 262,
            'cost_per_hour': 0.02
        },
        {
            'name': 'whisper-large-v3',
            'description': 'Fallback - Highest accuracy (if needed)',
            'multilingual': True,
            'wer': 10.3,
            'speed_factor': 299,
            'cost_per_hour': 0.111
        }
    ]
    
    # Auto-fallback trigger conditions
    SLOW_TRANSCRIPTION_THRESHOLD = 1.0     # 1 second threshold for "slow"
    MAX_SLOW_BEFORE_FALLBACK = 2          # Switch after 2 slow transcriptions
    PERFORMANCE_WINDOW_SIZE = 10           # Keep last 10 transcription times
    UPGRADE_THRESHOLD = 5                  # Upgrade after 5 fast transcriptions
    UPGRADE_HEADROOM_FACTOR = 0.7         # Only upgrade if performance <70% of threshold
    
    # Connection issue handling
    MAX_FAILED_BEFORE_ADAPTATION = 2      # Adapt thresholds after 2 failures
    
    # Updated thresholds for streaming mode with micro-files
    STREAMING_SLOW_TRANSCRIPTION_THRESHOLD = float(os.getenv("STREAMING_SLOW_TRANSCRIPTION_THRESHOLD", "0.4"))
    STREAMING_MAX_SLOW_BEFORE_FALLBACK = int(os.getenv("STREAMING_MAX_SLOW_BEFORE_FALLBACK", "4"))
    
    # Whisper API parameters to reduce hallucinations (per OpenAI recommendations)
    WHISPER_TEMPERATURE = 0.0                  # Deterministic mode reduces hallucinated filler words
    WHISPER_INITIAL_PROMPT = " "               # Disable language-model priming when no context needed


# =============================================================================
# UTTERANCE RECOVERY SYSTEM
# =============================================================================

class RecoveryConfig:
    """Settings for recovering truncated utterances"""
    
    # Recovery timing windows
    MIN_RECOVERY_DURATION = 0.10          # 100ms minimum for recovery attempt
    MAX_RECOVERY_DURATION = 0.14          # 140ms maximum (just under normal minimum)
    
    # Recovery audio requirements (made more lenient)
    MIN_AUDIO_SIZE_FOR_RECOVERY = 1024    # 2 frames * 512 bytes = ~32ms minimum (was 1536)
    MIN_AUDIO_SIZE_FOR_NORMAL = 2048      # 4 frames * 512 bytes for normal processing (was 2560)
    
    # Meaningful speech detection
    MEANINGFUL_WORDS = {
        # Basic responses
        'yes', 'no', 'ok', 'okay', 'sure', 'right', 'correct', 'wrong',
        'good', 'bad', 'fine', 'great', 'nice', 'cool',
        
        # Interruptions/objections
        'wait', 'stop', 'hold', 'pause', 'hang', 'actually', 'but',
        'however', 'though', 'hmm', 'uh', 'um', 'oh', 'ah',
        
        # Questions
        'what', 'when', 'where', 'who', 'why', 'how', 'which',
        'really', 'seriously', 'true', 'sure',
        
        # Common short phrases  
        'hold on', 'hang on', 'wait up', 'one sec', 'just a', 'not now',
        'go on', 'continue', 'keep going', 'i see', 'got it', 'makes sense',
    }
    
    # Content analysis thresholds
    MIN_MEANINGFUL_LENGTH = 3             # 3+ characters for meaningful content
    QUESTION_STARTERS = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'can', 'will', 'should']
    NEGATIONS = ['no', 'not', 'never', 'nothing', 'none', "don't", "won't", "can't"]


# =============================================================================
# DEAD CALL WATCHDOG SYSTEM
# =============================================================================

class WatchdogConfig:
    """Dead call detection and prevention settings"""
    
    # Timeout settings (increased for realistic phone call duration)
    DEAD_CALL_TIMEOUT = float(os.getenv("DEAD_CALL_TIMEOUT", "180.0"))  # 3 minutes default (was 30s)
    WARNING_THRESHOLD_FACTOR = 0.75       # Warn at 75% of timeout (2m 15s)
    MIN_CALL_TIME_FOR_WARNING = 60        # Only warn after 1 minute into call (was 15s)
    
    # Rapid detection thresholds (for obvious dead calls)
    MAX_CONSECUTIVE_FAILURES = 8          # Trigger after 8 consecutive failures (was 5 - more lenient)
    FAILURE_LOG_FREQUENCY = 3             # Log every 3 failures (was 2 - less spam)
    
    # Professional exit message
    EXIT_MESSAGE = ("I haven't heard anything for a while, so I'll let you go for now. "
                    "Feel free to call back anytime if you'd like to learn more about our AI solutions. "
                    "Have a great day!")


# =============================================================================
# PERFORMANCE MONITORING AND OPTIMIZATION
# =============================================================================

class PerformanceConfig:
    """Performance tracking and bottleneck analysis settings"""
    
    # Timing system configuration
    DETAILED_TIMING_ENABLED = os.getenv("SHOW_DETAILED_TIMING", "true").lower() == "true"
    TIMING_STATS_INTERVAL = 5             # Show stats every 5 operations
    PERFORMANCE_WINDOW_SIZE = 20          # Keep last 20 measurements for rolling averages
    
    # Performance thresholds (for regression detection)
    MAX_VAD_TIME_PER_CALL = 0.005        # 5ms maximum per VAD call
    MAX_TRANSCRIPTION_TIME = 3.0          # 3s maximum transcription time
    MAX_LLM_GENERATION_TIME = 5.0        # 5s maximum LLM response time
    MAX_TTS_SYNTHESIS_TIME = 10.0        # 10s maximum TTS synthesis time
    
    # Component tracking
    TRACKED_COMPONENTS = [
        'transcription',
        'llm_generation', 
        'tts_synthesis',
        'voice_detection',
        'end_to_end'
    ]


# =============================================================================
# ECHO PREVENTION AND FEEDBACK PROTECTION
# =============================================================================

class EchoPreventionConfig:
    """Settings for preventing AI self-interruption from TTS feedback"""
    
    # Enhanced echo detection parameters using normalized cross-correlation
    ECHO_CORRELATION_THRESHOLD = 0.8      # Increased from 0.6 - stricter correlation gate per WebRTC recommendations
    ECHO_THRESHOLD_MULTIPLIER = 6.0       # Raise thresholds during AI speech (increased to stop TTS bleed-through completely)
    AI_AUDIO_BUFFER_SIZE = 100            # Keep last 100 frames of AI audio
    
    # Hard-mute settings for complete TTS isolation
    ENABLE_HARD_MUTE = True               # Hard-mute mic input during TTS playback
    TAIL_MUTE_MS = 150                    # Extended from 50ms to catch TTS fade-out (per roadmap)
    
    # Protection timing
    ECHO_PROTECTION_ENABLED = True        # Enable echo prevention system
    TTS_FEEDBACK_BUFFER_SIZE = 10         # Track last 10 TTS frames for correlation
    
    # Enhanced detection using neural AEC principles (based on research)
    USE_NORMALIZED_CROSS_CORRELATION = True  # Use proper cross-correlation instead of energy ratio
    
    # Advanced gating for stationary noise (fans, AC units)
    ENABLE_SPECTRAL_FLATNESS_GATE = True     # Enable spectral flatness gating for fan noise
    SPECTRAL_FLATNESS_THRESHOLD = 0.2        # Reject frames with flatness < 0.2 (quasi-stationary noise)
    SPECTRAL_FLATNESS_DURATION_MS = 100      # Minimum duration for noise classification
    
    # Adaptive correlation thresholding
    USE_ADAPTIVE_CORRELATION_THRESHOLD = True  # Use mean + 2σ of background correlation
    CORRELATION_ADAPTATION_WINDOW = 50         # Frames to compute background correlation stats
    
    # Enhanced echo prevention using multiple correlation methods
    USE_PEARSON_CORRELATION = True             # Enable Pearson correlation for better echo detection
    ENABLE_TRANSCRIPT_ECHO_FILTERING = True    # Filter AI echo patterns at transcript level
    
    # Final safety net for residual duplicates (based on Google Assistant Duplex approach)
    ENABLE_REGEX_DEDUP_PASS = True             # Apply regex deduplication as final safety net
    
    # VAD tuning profiles (users can choose based on their environment)
    VAD_PROFILE = os.getenv("VAD_PROFILE", "strict")  # "strict", "balanced", "relaxed"


# =============================================================================
# PHONE CALL SIMULATION SETTINGS
# =============================================================================

class PhoneCallConfig:
    """Settings for realistic phone call simulation"""
    
    # Call initialization timing
    INITIAL_GREETING_WINDOW = 3.0         # 3s window for initial greeting
    GREETING_MIN_SPEECH_DURATION = 0.3    # 300ms minimum for greeting
    GREETING_MAX_SILENCE = 0.6            # 600ms silence for greeting
    GREETING_CHECK_INTERVAL = 0.015       # 15ms polling for greeting
    
    # Static phone range fallbacks (if noise floor measurement fails)
    STATIC_PHONE_RANGES = {
        'min_amplitude': 300,      # Quiet speech
        'max_amplitude': 6000,     # Loud but not shouting
        'min_rms': 80,             # Whisper level  
        'max_rms': 1200,           # Normal loud speech
        'threshold_amplitude': 500, # Standard phone speech
        'threshold_rms': 150       # Comfortable detection level
    }
    
    # Quit phrases for ending calls
    QUIT_PHRASES = ['goodbye', 'quit', 'exit', 'stop', 'end call', 'hang up']


# =============================================================================
# LLM GENERATION CONFIGURATION
# =============================================================================

class LLMConfig:
    # LLM Configuration
    MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"  # Groq model for BYO LLM
    TEMPERATURE = 0.7
    MAX_TOKENS = 1000  # Reduced for voice conversations
    STREAM = True
    
    # BYO LLM Configuration
    USE_BYO_LLM = True  # Set to False to use Deepgram's standard LLM
    GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    DEEPGRAM_LLM_MODEL = "gpt-4o-mini"  # Used when USE_BYO_LLM is False
    
    # Pricing information (for reference)
    BYO_LLM_RATE_PER_MINUTE = 0.07  # $0.07/minute with BYO LLM
    STANDARD_RATE_PER_MINUTE = 0.08  # $0.08/minute with Deepgram LLM


# =============================================================================
# TTS (TEXT-TO-SPEECH) CONFIGURATION
# =============================================================================

class TTSConfig:
    """Text-to-Speech synthesis settings"""
    
    # GROQ TTS settings
    MODEL_NAME = "playai-tts"              # GROQ's TTS model
    VOICE_NAME = "Fritz-PlayAI"            # Professional male voice
    RESPONSE_FORMAT = "wav"                # Audio format
    
    # Playback settings
    WORD_DURATION_SIMULATION = 0.15        # 150ms per word for text simulation fallback
    PROGRESS_UPDATE_INTERVAL = 0.1         # Update speech progress every 100ms
    
    # Audio file management
    TEMP_FILE_CLEANUP_WAIT = 0.1          # Wait before cleaning up temp files
    TEMP_FILE_PREFIX = "temp_audio_"       # Prefix for temporary audio files


# =============================================================================
# CONVERSATION PLANNING INTEGRATION
# =============================================================================

class PlanningConfig:
    """Strategic conversation planning settings"""
    
    # Planning system
    PLANNING_TYPE = "cold_call"            # Type of conversation planning
    PLANNING_INTERVAL = 2.0                # Background planning every 2 seconds
    PLANNING_TIMEOUT = 2.0                 # Timeout for planning system shutdown
    
    # Context building
    MAX_CONTEXT_MESSAGES = 14             # Maximum messages in planning context
    STRATEGIC_CONTEXT_PRIORITY = True      # Include strategic guidance in responses


# =============================================================================
# DEBUGGING AND LOGGING CONFIGURATION  
# =============================================================================

class DebugConfig:
    """Settings for debugging and development"""
    
    # Environment-controlled debug settings
    DEBUG_LOGGING = bool(os.getenv("VAD_DEBUG", "true").lower() == "true")  # Temporarily enabled
    LOG_AUDIO_DEVICES = True              # Log available audio devices on startup
    
    # Debug output control
    AUDIO_LEVEL_DEBUG_FREQUENCY = 50      # Show audio levels every 50th frame
    VAD_DEBUG_FREQUENCY = 50              # Show VAD details every 50th frame  
    PERFORMANCE_DEBUG_ENABLED = True      # Enable performance timing logs
    
    # Progress reporting
    CALIBRATION_STATUS_FREQUENCY = 5      # Show calibration status every 5 samples
    CONNECTION_STATUS_FREQUENCY = 8       # Show connection status every 8 samples (static mode)


# =============================================================================
# PHYSICAL CONTACT DETECTION CONFIGURATION
# =============================================================================

class PhysicalContactDetectionConfig:
    """Settings for detecting and filtering physical mic contact/brushing"""
    
    # Physical contact characteristics 
    CONTACT_DETECTION_ENABLED = True      # Enable physical contact filtering
    CONTACT_ANALYSIS_WINDOW = 0.08        # 80ms analysis window before stopping TTS
    CONTACT_MIN_SUSTAINED_SPEECH = 0.12   # 120ms minimum for real speech
    
    # Audio characteristics of physical contact vs speech
    CONTACT_SPIKE_THRESHOLD = 5.0         # Physical contact causes sharp amplitude spikes
    CONTACT_DURATION_MAX = 0.15           # Physical contact rarely sustains >150ms  
    CONTACT_FREQUENCY_PATTERN = True      # Check for non-speech frequency patterns
    
    # Pre-interruption validation
    PRE_INTERRUPT_BUFFER_SIZE = 6         # Buffer 6 chunks (~96ms) before stopping TTS
    REQUIRE_SUSTAINED_ACTIVITY = True     # Require sustained activity, not just spikes


# =============================================================================
# STREAMING CONFIGURATION
# =============================================================================

class StreamingConfig:
    """Configuration for streaming transcription mode"""
    # Feature flag to enable streaming mode
    ENABLE_STREAMING_MODE = os.getenv("ENABLE_STREAMING_MODE", "false").lower() == "true"
    
    # Ring buffer configuration
    RING_BUFFER_SECONDS = float(os.getenv("RING_BUFFER_SECONDS", "10.0"))  # Total buffer size
    SNAPSHOT_INTERVAL_MS = int(os.getenv("SNAPSHOT_INTERVAL_MS", "250"))    # How often to take snapshots
    WINDOW_SIZE_SECONDS = float(os.getenv("WINDOW_SIZE_SECONDS", "1.2"))    # Size of each transcription window (reduced per analysis)
    OVERLAP_SECONDS = float(os.getenv("OVERLAP_SECONDS", "0.2"))            # Overlap between windows (reduced per analysis)
    
    # Transcription pool configuration
    MAX_CONCURRENT_TRANSCRIPTIONS = int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "3"))
    TRANSCRIPTION_TIMEOUT_SECONDS = float(os.getenv("TRANSCRIPTION_TIMEOUT_SECONDS", "2.0"))
    
    # Transcript assembly configuration
    MIN_OVERLAP_CHARS = int(os.getenv("MIN_OVERLAP_CHARS", "25"))           # Min chars for overlap detection (increased to reduce duplicates)
    MAX_OVERLAP_CHARS = int(os.getenv("MAX_OVERLAP_CHARS", "50"))           # Max chars to check for overlap
    
    # Clause boundary detection
    CLAUSE_MIN_PAUSE_MS = int(os.getenv("CLAUSE_MIN_PAUSE_MS", "300"))      # Min pause to consider clause end
    PUNCTUATION_MARKS = ['.', '?', '!', ',', ';', ':']                     # Marks that indicate clause boundaries
    
    # Adaptive chunk sizing (capped to prevent overlap duplication issues)
    TARGET_THROUGHPUT_RATIO = float(os.getenv("TARGET_THROUGHPUT_RATIO", "200"))  # Target real-time multiplier
    MIN_CHUNK_SIZE_SECONDS = float(os.getenv("MIN_CHUNK_SIZE_SECONDS", "1.0"))
    MAX_CHUNK_SIZE_SECONDS = float(os.getenv("MAX_CHUNK_SIZE_SECONDS", "2.5"))    # Capped at 2.5s to prevent massive overlap windows
    BATCH_SILENCE_THRESHOLD = float(os.getenv("BATCH_SILENCE_THRESHOLD", "1.0"))  # Batch silent periods


# =============================================================================
# ACOUSTIC ECHO CANCELLATION (AEC) INTEGRATION
# =============================================================================

class AECConfig:
    """Advanced AEC integration settings based on WebRTC AEC3 and SpeexDSP"""
    
    # AEC backend selection
    ENABLE_AEC = os.getenv("ENABLE_AEC", "true").lower() == "true"  # Enable AEC front-end
    AEC_BACKEND = os.getenv("AEC_BACKEND", "webrtc")  # "webrtc", "speex", or "disabled"
    
    # WebRTC AEC3 settings
    WEBRTC_AEC_SAMPLE_RATE = 16000        # Must match audio config
    WEBRTC_AEC_CHANNELS = 1               # Mono processing
    WEBRTC_AEC_BLOCK_SIZE = 160           # 10ms blocks (160 samples at 16kHz)
    WEBRTC_AEC_DELAY_AGNOSTIC = True      # Handle variable delays
    WEBRTC_AEC_EXTENDED_FILTER = True     # Better for longer echo paths
    
    # SpeexDSP fallback settings
    SPEEX_AEC_FRAME_SIZE = 256            # 16ms frames to match current blocksize
    SPEEX_AEC_FILTER_LENGTH = 1024        # Echo cancellation filter length
    SPEEX_AEC_SUPPRESSION_LEVEL = -50     # Suppression in dB
    
    # Residual Echo Suppression (RES)
    ENABLE_RESIDUAL_ECHO_SUPPRESSION = True  # Post-AEC residual suppression
    RES_SUPPRESSION_FACTOR = 0.1             # Residual echo suppression factor
    
    # Double-talk detection
    ENABLE_DOUBLE_TALK_DETECTION = True   # WebRTC residual echo detector (RED)
    DTD_HANGOVER_MAX = 10                 # Frames to maintain double-talk state
    DTD_THRESHOLD = 0.5                   # Double-talk detection sensitivity


# =============================================================================
# NEURAL NOISE SUPPRESSION
# =============================================================================

class NoiseSuppressionConfig:
    """Neural noise suppression for stationary noise (fans, AC units)"""
    
    # RNNoise integration
    ENABLE_RNNOISE = os.getenv("ENABLE_RNNOISE", "false").lower() == "true"  # Optional neural denoising
    RNNOISE_MODEL_PATH = os.getenv("RNNOISE_MODEL_PATH", "")  # Path to custom RNNoise model
    
    # Spectral subtraction for stationary noise
    ENABLE_SPECTRAL_SUBTRACTION = True    # Classic spectral subtraction
    NOISE_ESTIMATION_FRAMES = 25          # Frames for noise profile (500ms)
    SPECTRAL_SUBTRACTION_ALPHA = 2.0      # Over-subtraction factor
    SPECTRAL_SUBTRACTION_BETA = 0.01      # Spectral floor
    
    # Adaptive noise gate
    ADAPTIVE_NOISE_GATE = True            # Learn per-session noise fingerprint
    NOISE_PROFILE_UPDATE_RATE = 0.1       # How fast to adapt noise profile
    NOISE_PROFILE_FRAMES = 25             # Frames to build initial noise profile


# =============================================================================
# CONFIGURATION VALIDATION AND UTILITIES
# =============================================================================

def get_vad_strictness_info(level: int) -> Dict[str, Any]:
    """Get human-readable information about VAD strictness level"""
    level = max(0, min(5, level))  # Clamp to valid range
    
    return {
        'level': level,
        'description': VADConfig.STRICTNESS_DESCRIPTIONS[level],
        'multipliers': VADConfig.STRICTNESS_MULTIPLIERS[level],
        'amp_multiplier': VADConfig.STRICTNESS_MULTIPLIERS[level][0],
        'rms_multiplier': VADConfig.STRICTNESS_MULTIPLIERS[level][1]
    }

def get_vad_profile_settings(profile: str) -> Dict[str, Any]:
    """Get VAD settings for different environment profiles
    
    Based on analysis of echo vs natural speech tradeoffs:
    - strict: Better for echo-prone environments, may clip natural speech
    - balanced: Compromise between echo prevention and natural speech
    - relaxed: Better for natural speech, may allow some echo
    """
    profiles = {
        "strict": {
            "strictness": 4,
            "noise_multiplier": 0.8,
            "description": "Echo-prone environments (recommended for initial analysis)",
            "use_case": "Prevents TTS echo leakage, may clip soft speech onset"
        },
        "balanced": {
            "strictness": 3,
            "noise_multiplier": 0.7,
            "description": "Default compromise between echo prevention and natural speech",
            "use_case": "Good starting point for most environments"
        },
        "relaxed": {
            "strictness": 2,
            "noise_multiplier": 0.6,
            "description": "Natural speech environments (recommended for general use)",
            "use_case": "Allows natural speech onset, requires good echo suppression"
        },
        "recovery": {
            "strictness": 1,
            "noise_multiplier": 0.4,
            "description": "Emergency voice detection recovery mode",
            "use_case": "Very sensitive detection for troubleshooting voice issues"
        }
    }
    
    return profiles.get(profile, profiles["balanced"])

def get_webrtc_sensitivity_info(sensitivity: int) -> Dict[str, str]:
    """Get human-readable information about WebRTC VAD sensitivity"""
    sensitivity = max(0, min(3, sensitivity))  # Clamp to valid range
    
    return {
        'sensitivity': sensitivity,
        'description': VADConfig.WEBRTC_DESCRIPTIONS[sensitivity]
    }

def validate_configuration() -> List[str]:
    """Validate configuration values and return any warnings"""
    warnings = []
    
    # Check VAD configuration ranges
    if not (0 <= VADConfig.STRICTNESS <= 5):
        warnings.append(f"VAD_STRICTNESS={VADConfig.STRICTNESS} outside valid range [0-5]")
    
    if not (0 <= VADConfig.WEBRTC_SENSITIVITY <= 3):
        warnings.append(f"VAD_WEBRTC_SENSITIVITY={VADConfig.WEBRTC_SENSITIVITY} outside valid range [0-3]")
    
    if not (0.5 <= VADConfig.NOISE_MULTIPLIER <= 2.0):
        warnings.append(f"VAD_NOISE_MULTIPLIER={VADConfig.NOISE_MULTIPLIER} outside recommended range [0.5-2.0]")
    
    # Check timing configuration
    if TimingConfig.POST_CALIBRATION_MAX_SILENCE >= TimingConfig.INITIAL_MAX_SILENCE:
        warnings.append("Post-calibration silence timeout should be faster than initial timeout")
    
    # Check performance thresholds
    if PerformanceConfig.MAX_VAD_TIME_PER_CALL > 0.01:  # 10ms
        warnings.append(f"VAD time threshold {PerformanceConfig.MAX_VAD_TIME_PER_CALL*1000:.1f}ms may be too slow")
    
    return warnings

def get_configuration_summary() -> Dict[str, Any]:
    """Get a summary of key configuration values for display"""
    vad_info = get_vad_strictness_info(VADConfig.STRICTNESS)
    webrtc_info = get_webrtc_sensitivity_info(VADConfig.WEBRTC_SENSITIVITY)
    
    return {
        'audio': {
            'sample_rate': AudioConfig.SAMPLE_RATE,
            'frame_duration_ms': AudioConfig.FRAME_DURATION_MS,
            'blocksize': AudioConfig.BLOCKSIZE
        },
        'vad': {
            'strictness': vad_info,
            'webrtc': webrtc_info,
            'noise_multiplier': VADConfig.NOISE_MULTIPLIER,
            'debug_enabled': VADConfig.DEBUG_ENABLED
        },
        'timing': {
            'initial_silence_ms': TimingConfig.INITIAL_MAX_SILENCE * 1000,
            'post_calibration_silence_ms': TimingConfig.POST_CALIBRATION_MAX_SILENCE * 1000,
            'min_speech_duration_ms': TimingConfig.INITIAL_MIN_SPEECH_DURATION * 1000
        },
        'performance': {
            'max_vad_time_ms': PerformanceConfig.MAX_VAD_TIME_PER_CALL * 1000,
            'detailed_timing': PerformanceConfig.DETAILED_TIMING_ENABLED
        },
        'watchdog': {
            'timeout_seconds': WatchdogConfig.DEAD_CALL_TIMEOUT,
            'max_consecutive_failures': WatchdogConfig.MAX_CONSECUTIVE_FAILURES
        }
    }

# =============================================================================
# EXPORT CONFIGURATION CLASSES FOR EASY IMPORTING
# =============================================================================

__all__ = [
    'AudioConfig',
    'VADConfig', 
    'TimingConfig',
    'InterruptionConfig',
    'PersonalizationConfig',
    'ASRConfig',
    'RecoveryConfig',
    'WatchdogConfig',
    'PerformanceConfig',
    'EchoPreventionConfig',
    'PhoneCallConfig',
    'LLMConfig',
    'TTSConfig',
    'PlanningConfig',
    'DebugConfig',
    'PhysicalContactDetectionConfig',
    'StreamingConfig',
    'get_vad_strictness_info',
    'get_webrtc_sensitivity_info',
    'validate_configuration',
    'get_configuration_summary'
] 