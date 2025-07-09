# services/ollama_service.py
import requests
from config import settings

class OllamaService:
    def __init__(self):
        self.ollama_api_url = settings.OLLAMA_API_URL
        self.ollama_model_name = settings.OLLAMA_MODEL_NAME

    def generate_question(self, user_transcript: str) -> str:
        prompt = f"""You are an AI interviewer. Based on the candidate's previous answer, generate a concise and relevant follow-up interview question.
        Candidate's last answer: "{user_transcript}"
        Your question:"""

        payload = {
            "model": self.ollama_model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.ollama_api_url, json=payload)
            response.raise_for_status()
            llm_response_data = response.json()
            generated_question = llm_response_data.get("response", "Could you please elaborate on that?")
            return generated_question.strip().strip('"')
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return "I apologize, I'm having trouble generating a question right now. Could you please rephrase your last answer?"