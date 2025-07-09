# api/tts.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from io import BytesIO
from models.request_models import TextRequest
from services.kokoro_service import KokoroService

router = APIRouter()
kokoro_service = KokoroService()

@router.post("/")
async def text_to_speech(request: TextRequest):
    text_to_speak = request.text
    print(f"Received text for TTS: '{text_to_speak}'")

    try:
        audio_bytes_io = kokoro_service.synthesize_speech(text_to_speak)
        return StreamingResponse(audio_bytes_io, media_type="audio/wav")
    except Exception as e:
        print(f"Error during TTS generation: {e}")
        return {"error": f"An error occurred during text-to-speech: {e}"}