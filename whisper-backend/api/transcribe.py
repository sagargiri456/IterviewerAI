# api/transcribe.py
import os
import tempfile
import shutil
from fastapi import APIRouter, UploadFile, File
from services.whisper_service import WhisperService

router = APIRouter()
whisper_service = WhisperService()

@router.post("/")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    if not audio_file.filename.endswith((".mp3", ".wav", ".flac", ".m4a", ".webm")):
        return {"error": "Unsupported file format. Please upload MP3, WAV, FLAC, or M4A."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        shutil.copyfileobj(audio_file.file, temp_audio)
        temp_audio_path = temp_audio.name

    try:
        print(f"Transcribing {audio_file.filename}...")
        transcript = whisper_service.transcribe(temp_audio_path)
        print("Transcript generated")
        print(transcript)
        return {"filename": audio_file.filename, "transcript": transcript}
    except Exception as e:
        print(f"Transcription error: {e}")
        return {"error": f"An error occurred during transcription: {e}"}
    finally:
        os.remove(temp_audio_path)