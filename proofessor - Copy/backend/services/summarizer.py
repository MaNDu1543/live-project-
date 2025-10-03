# summarize.py

# IMPORTANT: You must set your Gemini API key in the .env file as:
# GEMINI_API_KEY=your_api_key_here

import os
import requests
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# Create FastAPI router
router = APIRouter()

# --------- API CALL FUNCTION ----------
def call_gemini_api(prompt: str) -> str:
    """Call Gemini API with a given prompt"""
    try:
        response = requests.post(
            GEMINI_API_URL,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "[No summary returned]")
            )
        else:
            return f"[Gemini API Error: {response.status_code}] {response.text}"
    except requests.exceptions.RequestException as e:
        return f"[Gemini API Exception] {str(e)}"


# --------- SUMMARIZATION HELPERS ----------
def chunk_text(text, max_words=500):
    """Split long text into smaller chunks"""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


def summarize_text(text: str, max_length: int = 300, min_length: int = 100) -> str:
    """Summarize small/medium text directly with Gemini"""
    prompt = f"Summarize the following academic text in {min_length}-{max_length} words:\n{text}"
    return call_gemini_api(prompt)


def summarize_large_text(full_text: str) -> str:
    """Handle large texts by summarizing in chunks and combining"""
    partial_summaries = []
    for chunk in chunk_text(full_text):
        partial_summaries.append(summarize_text(chunk, max_length=150, min_length=50))
    # Final summary of all partial summaries
    return summarize_text(" ".join(partial_summaries), max_length=300, min_length=100)


# --------- FASTAPI ENDPOINT ----------
class SummarizeRequest(BaseModel):
    text: str


@router.post("/summarize/")
async def summarize_api(req: SummarizeRequest):
    """API endpoint to summarize text (handles large input automatically)"""
    text = req.text.strip()
    if len(text.split()) > 600:  # if too long, chunk it
        summary = summarize_large_text(text)
    else:
        summary = summarize_text(text)
    return {"summary": summary}
