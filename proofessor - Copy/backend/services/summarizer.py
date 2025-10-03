

# IMPORTANT: You must set your Gemini API key in the .env file as GEMINI_API_KEY=...
import os
import requests
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # <-- Put your Gemini API key in .env
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + GEMINI_API_KEY

def summarize_text(text: str, max_length: int = 300, min_length: int = 100) -> str:
    """
    Calls Gemini API to summarize the input text.
    """
    prompt = f"Summarize the following academic text in {min_length}-{max_length} words:\n{text}"
    response = requests.post(
        GEMINI_API_URL,
        json={
            "contents": [{"parts": [{"text": prompt}]}]
        },
        timeout=30
    )
    if response.status_code == 200:
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return "[Gemini API: Unexpected response format]"
    else:
        return f"[Gemini API Error: {response.status_code}] {response.text}"
