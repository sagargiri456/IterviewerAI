# api/generate.py
from fastapi import APIRouter
from models.request_models import TranscriptRequest
from services.ollama_service import OllamaService

router = APIRouter()
ollama_service = OllamaService()

@router.post("/")
async def generate_next_question(request: TranscriptRequest):
    user_transcript = request.transcript
    print(f"Received transcript for question generation: '{user_transcript}'")

    generated_question = ollama_service.generate_question(user_transcript)

    print(f"Generated question: '{generated_question}'")
    return {"question": generated_question}