# config.py
import os
import torch

class Settings:
    # CORS
    CORS_ORIGINS: list = ["http://localhost", "http://localhost:3000"]

    # Whisper Model Settings
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
    WHISPER_COMPUTE_TYPE: str = "float32"

    # Ollama Settings
    OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
    OLLAMA_MODEL_NAME: str = "gemma:2b"
    
    # Kokoro TTS Settings
    KOKORO_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    KOKORO_SAMPLE_RATE: int = 24000
    KOKORO_VOICE: str = "af_heart"
    KOKORO_SPLIT_PATTERN: str = r'\n+'

settings = Settings()