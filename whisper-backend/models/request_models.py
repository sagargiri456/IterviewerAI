# models/request_models.py
from pydantic import BaseModel

class TranscriptRequest(BaseModel):
    transcript: str

class TextRequest(BaseModel):
    text: str