import os
from typing import Optional
from groq import Groq

from config_defaults import TTSConfig

_client: Optional[Groq] = None

def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable not set")
        _client = Groq(api_key=api_key)
    return _client


def synthesize_speech(text: str, voice: str = TTSConfig.VOICE_NAME,
                      model: str = TTSConfig.MODEL_NAME,
                      response_format: str = TTSConfig.RESPONSE_FORMAT) -> bytes:
    """Generate speech audio bytes from text using Groq Play-AI TTS."""
    client = get_client()
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=response_format,
    )
    return response.content
