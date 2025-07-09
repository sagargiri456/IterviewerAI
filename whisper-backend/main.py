# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from api import transcribe, generate, tts
from services.kokoro_service import load_kokoro_pipeline,KokoroService
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(transcribe.router, prefix="/transcribe", tags=["transcription"])
app.include_router(generate.router, prefix="/generate", tags=["question_generation"])
app.include_router(tts.router, prefix="/tts", tags=["text_to_speech"])

@app.on_event("startup")
async def startup_event():
    """
    Initializes the Kokoro TTS pipeline on application startup.
    """
    print("Application starting up...")
    load_kokoro_pipeline()
    app.state.kokoro_service = KokoroService()
    print("Application startup complete.")


@app.get("/")
async def root():
    return {"message": "Whisper Backend is running. Check /docs for API documentation."}