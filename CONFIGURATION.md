# Re:MEMBER AI Voice Agent Configuration Guide

## BYO LLM vs Standard LLM

This voice agent supports two LLM configurations:

### BYO LLM (Bring Your Own) - $0.07/minute
- Uses **Groq** as the LLM provider through Deepgram's BYO LLM feature
- Leverages Groq's ultra-fast LPU inference (<200ms latency)
- **Model**: `meta-llama/llama-4-maverick-17b-128e-instruct`
- **Pricing**: $0.07 per minute
- **Concurrency**: Up to 5 simultaneous sessions

### Standard LLM - $0.08/minute  
- Uses **Deepgram's** native LLM
- **Model**: `gpt-4o-mini`
- **Pricing**: $0.08 per minute
- **Concurrency**: Up to 5 simultaneous sessions

## Configuration Methods

### Method 1: Environment Variables (.env file)

Create a `.env` file in your project root:

```bash
# Required API Keys
DEEPGRAM_API_KEY=your_deepgram_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# BYO LLM Configuration
USE_BYO_LLM=true  # Set to false for Deepgram's standard LLM

# Voice Configuration
AGENT_VOICE_NAME=Cheyenne-PlayAI
AGENT_PROMPT_PATH=prompts/system_prompt.txt

# Audio Configuration (optional)
SD_INPUT_DEVICE=Your Microphone Name
SD_OUTPUT_DEVICE=Your Speaker Name

# Audio Buffering (optional)
DG_START_THRESHOLD=24
DG_WRITE_CHUNKS=2
```

### Method 2: Modify config_defaults.py

Edit the `LLMConfig` class in `config_defaults.py`:

```python
class LLMConfig:
    # Set USE_BYO_LLM to False to use Deepgram's standard LLM
    USE_BYO_LLM = True  # Change this line
    
    # Groq model for BYO LLM
    MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"
    
    # Deepgram model for standard LLM
    DEEPGRAM_LLM_MODEL = "gpt-4o-mini"
```

## Quick Switch Commands

### Enable BYO LLM (Groq):
```bash
export USE_BYO_LLM=true
python main.py
```

### Enable Standard LLM (Deepgram):
```bash
export USE_BYO_LLM=false
python main.py
```

## API Key Setup

### Deepgram API Key
1. Sign up at [deepgram.com](https://deepgram.com)
2. Navigate to your dashboard
3. Create a new API key
4. Add to your `.env` file: `DEEPGRAM_API_KEY=your_key_here`

### Groq API Key (for BYO LLM)
1. Sign up at [console.groq.com](https://console.groq.com)
2. Create a new API key
3. Add to your `.env` file: `GROQ_API_KEY=your_key_here`

## Available Voice Models

The agent now supports Groq Play-AI voices:
- `Arista-PlayAI`
- `Atlas-PlayAI`
- `Basil-PlayAI`
- `Briggs-PlayAI`
- `Calum-PlayAI`
- `Celeste-PlayAI`
- `Cheyenne-PlayAI` *(default)*
- `Chip-PlayAI`
- `Cillian-PlayAI`
- `Deedee-PlayAI`
- `Fritz-PlayAI`
- `Gail-PlayAI`
- `Indigo-PlayAI`
- `Mamaw-PlayAI`
- `Mason-PlayAI`
- `Mikail-PlayAI`
- `Mitch-PlayAI`
- `Quinn-PlayAI`
- `Thunder-PlayAI`

## Performance Comparison

| Feature | BYO LLM (Groq) | Standard (Deepgram) |
|---------|----------------|---------------------|
| LLM Latency | <200ms | ~300ms |
| Price/minute | $0.07 | $0.08 |
| Model | meta-llama/llama-4-maverick-17b-128e-instruct | gpt-4o-mini |
| Max Sessions | 5 | 5 |
| Setup Complexity | Moderate | Simple |

## Troubleshooting

### "GROQ_API_KEY not set" Error
- Ensure you have added `GROQ_API_KEY=your_key` to your `.env` file
- This is required even when using standard LLM mode

### Voice Agent Connection Failed
- Check your Deepgram API key is valid
- Verify your internet connection
- Ensure you haven't exceeded the 5 concurrent session limit

### Audio Issues
- Run the microphone test in the Gradio interface
- Check that your browser has microphone permissions
- Try different input/output devices in the configuration 