# api/tts.py
from fastapi import APIRouter, Request # Import Request
from fastapi.responses import StreamingResponse
from io import BytesIO
from models.request_models import TextRequest
# from services.kokoro_service import KokoroService # Removed direct import and instantiation

router = APIRouter()

@router.post("/")
async def text_to_speech(request: Request, text_request: TextRequest): # Use text_request as parameter name
    text_to_speak = text_request.text # Access text from text_request
    print(f"Received text for TTS: '{text_to_speak}'")

    # Get the KokoroService instance from app.state
    # CORRECTED LINE: Use 'request' instead of 'text_request'
    kokoro_service = request.app.state.kokoro_service

    try:
        audio_bytes_io = kokoro_service.synthesize_speech(text_to_speak)
        return StreamingResponse(audio_bytes_io, media_type="audio/wav")
    except Exception as e:
        print(f"Error during TTS generation: {e}")
        return {"error": f"An error occurred during text-to-speech: {e}"}

