import requests
from typing import List, Dict, Optional
import os

from .config import get_openrouter_api_key, get_openai_api_key

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm_response(
    messages: List[Dict[str, str]],
    model_name: str = "google/gemini-2.5-flash-lite",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """Send a chat completion request to OpenRouter and return the assistant content.

    Messages must be in OpenAI-compatible format:
    [{"role": "system|user|assistant", "content": "..."}, ...]
    """
    api_key = get_openrouter_api_key()
    if not api_key:
        raise ValueError("OpenRouter API key not available. Configure .env for local or Secrets Manager for non-local envs.")

    payload: Dict[str, object] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        response = requests.post(
            url=f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://example.com",
                "X-Title": "med-sim-app",
            },
            json=payload,
            timeout=60,
        )
    except requests.exceptions.RequestException as exc:
        return f"Connection error with OpenRouter API: {exc}"

    # Try to parse JSON either on success or error to surface details
    content: Optional[str] = None
    try:
        data = response.json()
        if response.ok:
            return data["choices"][0]["message"]["content"]
        # Error path: surface message if present
        err_msg = (
            data.get("error", {}).get("message")
            if isinstance(data, dict)
            else None
        )
        content = err_msg or str(data)
    except Exception:
        content = response.text

    return f"OpenRouter API error {response.status_code}: {content}"


def transcribe_audio(
    file_path: str,
    model_name: str = "whisper-1",
    prompt: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Transcribe audio using OpenAI Whisper.

    Tries direct OpenAI API first, then OpenRouter as fallback.
    Returns text or error string.
    """
    if OpenAI is None:
        return "OpenAI library not installed. Run: uv sync"

    # Try direct OpenAI API first (if key available)
    openai_key = get_openai_api_key()
    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            with open(file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=model_name,
                    file=audio_file,
                    prompt=prompt,
                    language=language,
                )
                return transcript.text
        except Exception as exc:
            error_msg = str(exc)
            # Check for invalid API key error
            if "401" in error_msg or "invalid_api_key" in error_msg.lower() or "incorrect api key" in error_msg.lower():
                return "Błąd: Nieprawidłowy klucz OpenAI API. Sprawdź OPENAI_API_KEY w .env lub Secrets Manager."
            # Check for other common errors
            if "429" in error_msg or "rate limit" in error_msg.lower():
                return "Błąd: Przekroczono limit zapytań do OpenAI. Spróbuj ponownie za chwilę."
            # Return detailed error (sanitized)
            return f"Błąd transkrypcji OpenAI: {error_msg[:200]}"  # Limit message length

    # Fallback: Try OpenRouter (may not support audio)
    api_key = get_openrouter_api_key()
    if not api_key:
        return "Błąd: Brak klucza API. Ustaw OPENAI_API_KEY dla transkrypcji audio (OpenRouter może nie obsługiwać transkrypcji)."

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://example.com",
                "X-Title": "med-sim-app",
            },
        )
        # OpenRouter uses model name with prefix
        openrouter_model = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=openrouter_model,
                file=audio_file,
                prompt=prompt,
                language=language,
            )
            return transcript.text
    except Exception as exc:
        error_msg = str(exc)
        # Check for 405 error specifically
        if "405" in error_msg or "Method Not Allowed" in error_msg:
            return "Błąd 405: OpenRouter nie obsługuje transkrypcji audio. Dodaj OPENAI_API_KEY do .env lub Secrets Manager."
        return f"Błąd transkrypcji przez OpenRouter: {error_msg}"


