import os 
from fastapi import FastAPI
from faster_whisper import WhisperModel 
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import tempfile
import shutil
from pydantic import BaseModel
from typing import Dict, Any
from io import BytesIO
from fastapi.responses import StreamingResponse
import requests
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import torch

kokoro_device = "cuda" if torch.cuda.is_available() else "cpu"
kokoro_pipeline = None
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],    
)
class TranscriptRequest(BaseModel):
    transcript: str

class TextRequest(BaseModel):
    text: str

model_size = "base"
device = "cpu" 
compute_type = "float32"

try:
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"Loaded the Whisper offline model: the model size is {model_size} runnning on {device} with {compute_type} compute type.")
except Exception as e:
    print(f"Error in loading Whisper model: {e}")
    print("Ensure FFmpeg is installed and in your PATH, and PyTorch/CUDA is correctly set up for GPU.")
    raise




@app.post("/transcribe/")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    if not audio_file.filename.endswith((".mp3", ".wav", ".flac", ".m4a", ".webm")):
        return {"error": "Unsupported file format. Please upload MP3, WAV, FLAC, or M4A."}

    # Create a temporary file to save the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        shutil.copyfileobj(audio_file.file, temp_audio)
        temp_audio_path = temp_audio.name

    try:
        print(f"Transcribing {audio_file.filename}...")
        segments, info = model.transcribe(
            temp_audio_path,
            beam_size=5,

        )

        transcript = ""
        for segment in segments:
            transcript += segment.text
        print("transcript is generted")
        print(transcript)
        return {"filename": audio_file.filename, "transcript": transcript}

    except Exception as e:
        print(f"Transcription error: {e}")
        return {"error": f"An error occurred during transcription: {e}"}
    finally:
        # Clean up the temporary file
        os.remove(temp_audio_path)

@app.get("/")
async def root():
    return {"message": "Whisper Backend is running. Send audio to /transcribe/"}

@app.post("/generate")
async def generate_next_question(request: TranscriptRequest):
    """
    Generates the next interview question based on the user's transcript using a local LLM (Ollama).
    """
    user_transcript = request.transcript
    print(f"Received transcript for question generation: '{user_transcript}'")

    # --- Ollama LLM Integration ---
    ollama_api_url = "http://localhost:11434/api/generate" 
    ollama_model_name = "gemma:2b" 

    prompt = f"""You are an AI interviewer. Based on the candidate's previous answer, generate a concise and relevant follow-up interview question.
    Candidate's last answer: "{user_transcript}"
    Your question:"""

    payload = {
        "model": ollama_model_name,
        "prompt": prompt,
        "stream": False # We want the full response at once
    }

    try:
        response = requests.post(ollama_api_url, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        # Ollama's /api/generate returns a JSON object with a 'response' field for non-streaming
        print(response)
        llm_response_data = response.json()
        generated_question = llm_response_data.get("response", "Could you please elaborate on that?")
        
        # Basic cleanup: remove leading/trailing whitespace or quotes if the model adds them
        generated_question = generated_question.strip().strip('"')

    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        generated_question = "I apologize, I'm having trouble generating a question right now. Could you please rephrase your last answer?"
    # --- End Ollama LLM Integration ---

    print(f"Generated question: '{generated_question}'")
    return {"question": generated_question}
@app.post("/tts")
async def text_to_speech(request: TextRequest):
    """
    Converts text to speech and returns the audio using Hexgrad/Kokoro-TTS.
    """
    global kokoro_pipeline # This ensures we use the globally loaded pipeline
    text_to_speak = request.text
    print(f"Received text for TTS: '{text_to_speak}'")

    if kokoro_pipeline is None:
        print("Error: Kokoro TTS pipeline not loaded. Attempting to load now...")
        # This fallback is for safety, but @app.on_event("startup") should handle it.
        try:
            kokoro_pipeline = KPipeline(lang_code='a', device=kokoro_device)
            print("Kokoro TTS pipeline loaded successfully (fallback).")
        except Exception as e:
            print(f"Failed to load Kokoro TTS pipeline in fallback: {e}")
            return {"error": "TTS model could not be loaded."}

    audio_bytes_io = BytesIO()

    try:
        all_audio_segments = []
        sample_rate = 24000 # Kokoro's default sample rate

        # Iterate through the generator yielded by kokoro_pipeline
        for _, _, audio_array in kokoro_pipeline(text_to_speak, voice='af_heart', split_pattern=r'\n+'):
            all_audio_segments.append(audio_array)
        
        if all_audio_segments:
            combined_audio = np.concatenate(all_audio_segments)
            sf.write(audio_bytes_io, combined_audio, sample_rate, format='WAV')
        else:
            print("Warning: Kokoro TTS generated no audio for the given text.")
            # Create a minimal silent WAV header if no audio is generated
            num_channels = 1
            bits_per_sample = 16
            data_size = 0 # No actual audio data
            file_size = 36 + data_size
            byte_rate = sample_rate * num_channels * bits_per_sample // 8
            block_align = num_channels * bits_per_sample // 8

            wav_header = BytesIO()
            wav_header.write(b'RIFF')
            wav_header.write(file_size.to_bytes(4, 'little'))
            wav_header.write(b'WAVE')
            wav_header.write(b'fmt ')
            wav_header.write(b'\x10\x00\x00\x00')
            wav_header.write(b'\x01\x00')
            wav_header.write(num_channels.to_bytes(2, 'little'))
            wav_header.write(sample_rate.to_bytes(4, 'little'))
            wav_header.write(byte_rate.to_bytes(4, 'little'))
            wav_header.write(block_align.to_bytes(2, 'little'))
            wav_header.write(bits_per_sample.to_bytes(2, 'little'))
            wav_header.write(b'data')
            wav_header.write(data_size.to_bytes(4, 'little'))
            audio_bytes_io.write(wav_header.getvalue())

        audio_bytes_io.seek(0) # Rewind the stream to the beginning

        # Return the audio as a streaming response with the correct media type
        return StreamingResponse(audio_bytes_io, media_type="audio/wav")

    except Exception as e:
        print(f"Error during Kokoro TTS generation: {e}")
        # Return an error response if TTS fails
        return {"error": f"An error occurred during text-to-speech: {e}"}