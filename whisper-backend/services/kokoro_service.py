# services/kokoro_service.py
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import torch
from io import BytesIO
from config import settings

# Global variable to store the loaded pipeline
kokoro_pipeline = None

def load_kokoro_pipeline():
    """
    Loads the Kokoro TTS pipeline. This function should ideally be called once at startup.
    """
    global kokoro_pipeline
    if kokoro_pipeline is None:
        try:
            kokoro_pipeline = KPipeline(lang_code='a', device=settings.KOKORO_DEVICE)
            print("Kokoro TTS pipeline loaded successfully.")
        except Exception as e:
            print(f"Failed to load Kokoro TTS pipeline: {e}")
            # You might want to raise an exception here or handle it more robustly
            kokoro_pipeline = None # Ensure it's None if loading fails

class KokoroService:
    def __init__(self):
        # The pipeline is loaded globally, so we just check its existence here.
        if kokoro_pipeline is None:
            raise RuntimeError("Kokoro TTS pipeline not loaded. Call load_kokoro_pipeline() at startup.")

    def synthesize_speech(self, text: str) -> BytesIO:
        audio_bytes_io = BytesIO()
        all_audio_segments = []

        for _, _, audio_array in kokoro_pipeline(
            text,
            voice=settings.KOKORO_VOICE,
            split_pattern=settings.KOKORO_SPLIT_PATTERN
        ):
            all_audio_segments.append(audio_array)

        if all_audio_segments:
            combined_audio = np.concatenate(all_audio_segments)
            sf.write(audio_bytes_io, combined_audio, settings.KOKORO_SAMPLE_RATE, format='WAV')
        else:
            print("Warning: Kokoro TTS generated no audio for the given text.")
            # Create a minimal silent WAV header if no audio is generated
            num_channels = 1
            bits_per_sample = 16
            data_size = 0 # No actual audio data
            file_size = 36 + data_size
            byte_rate = settings.KOKORO_SAMPLE_RATE * num_channels * bits_per_sample // 8
            block_align = num_channels * bits_per_sample // 8

            wav_header = BytesIO()
            wav_header.write(b'RIFF')
            wav_header.write(file_size.to_bytes(4, 'little'))
            wav_header.write(b'WAVE')
            wav_header.write(b'fmt ')
            wav_header.write(b'\x10\x00\x00\x00')
            wav_header.write(b'\x01\x00')
            wav_header.write(num_channels.to_bytes(2, 'little'))
            wav_header.write(settings.KOKORO_SAMPLE_RATE.to_bytes(4, 'little'))
            wav_header.write(byte_rate.to_bytes(4, 'little'))
            wav_header.write(block_align.to_bytes(2, 'little'))
            wav_header.write(bits_per_sample.to_bytes(2, 'little'))
            wav_header.write(b'data')
            wav_header.write(data_size.to_bytes(4, 'little'))
            audio_bytes_io.write(wav_header.getvalue())

        audio_bytes_io.seek(0)
        return audio_bytes_io