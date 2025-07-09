# services/whisper_service.py
from faster_whisper import WhisperModel
from config import settings

class WhisperService:
    def __init__(self):
        try:
            self.model = WhisperModel(
                settings.WHISPER_MODEL_SIZE,
                device=settings.WHISPER_DEVICE,
                compute_type=settings.WHISPER_COMPUTE_TYPE
            )
            print(f"Loaded the Whisper offline model: size={settings.WHISPER_MODEL_SIZE}, device={settings.WHISPER_DEVICE}, compute_type={settings.WHISPER_COMPUTE_TYPE}.")
        except Exception as e:
            print(f"Error in loading Whisper model: {e}")
            print("Ensure FFmpeg is installed and in your PATH, and PyTorch/CUDA is correctly set up for GPU.")
            raise

    def transcribe(self, audio_path: str) -> str:
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        transcript = "".join(segment.text for segment in segments)
        return transcript