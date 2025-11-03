import logging
import requests
from typing import List, Dict, Optional

from .config import get_openrouter_api_key, get_openai_api_key, get_groq_api_key

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None

try:
    from groq import Groq  # type: ignore
except ImportError:
    Groq = None


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
    try:
        api_key = get_openrouter_api_key()
    except ValueError as e:
        # Re-raise with more context if needed
        raise e
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
    model_name: str = "whisper-large-v3",
    prompt: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Transcribe audio using Groq Whisper Large V3 Turbo.

    Tries Groq API first, then OpenAI as fallback, then OpenRouter.
    Returns text or error string.
    """
    # Try Groq API first (primary method)
    if Groq is None:
        logger.warning("Groq library not installed. Install with: uv add groq")
    else:
        groq_key = get_groq_api_key()
        if groq_key:
            try:
                client = Groq(api_key=groq_key)
                with open(file_path, "rb") as file:
                    file_content = file.read()
                    create_kwargs = {
                        "file": (file_path, file_content),
                        "model": model_name,
                        "temperature": 0,
                        "response_format": "verbose_json",
                    }
                    # Add language parameter if provided
                    # Groq Whisper uses ISO 639-1 codes, "pl" for Polish
                    if language is not None:
                        create_kwargs["language"] = language
                        logger.info(f"Transcribing with language: {language}")
                    else:
                        logger.warning("No language specified for transcription - accuracy may be lower")
                    
                    # Add prompt hint for Polish language to improve accuracy
                    # Prompt helps Whisper with context and medical terminology
                    if language == "pl" or language == "polish":
                        # Polish language hint - helps Whisper understand context
                        # Use medical context hint for better accuracy with medical terms
                        create_kwargs["prompt"] = "Rozmowa medyczna w języku polskim. Lekarz i pacjent rozmawiają po polsku o objawach, diagnozie i leczeniu."
                        logger.info("Added Polish medical context prompt")
                    elif prompt is not None:
                        # Use provided prompt if available
                        create_kwargs["prompt"] = prompt
                    
                    logger.debug(f"Transcription parameters: model={model_name}, language={create_kwargs.get('language')}, has_prompt={'prompt' in create_kwargs}")
                    transcription = client.audio.transcriptions.create(**create_kwargs)
                    logger.info(f"Transcription successful, length: {len(transcription.text) if transcription.text else 0}")
                    return transcription.text
            except Exception as exc:
                error_msg = str(exc)
                # Check for invalid API key error
                if "401" in error_msg or "invalid_api_key" in error_msg.lower() or "incorrect api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    logger.warning(f"Groq API key error: {error_msg[:100]}")
                    # Fall through to OpenAI fallback
                # Check for rate limit
                elif "429" in error_msg or "rate limit" in error_msg.lower():
                    logger.warning(f"Groq rate limit: {error_msg[:100]}")
                    # Fall through to OpenAI fallback
                else:
                    logger.warning(f"Groq transcription error: {error_msg[:200]}")
                    # Fall through to OpenAI fallback

    # Fallback: Try OpenAI API (if key available)
    if OpenAI is not None:
        openai_key = get_openai_api_key()
        if openai_key:
            try:
                client = OpenAI(api_key=openai_key)
                # OpenAI API uses "whisper-1" for their model
                openai_model = "whisper-1"  # OpenAI only supports whisper-1
                with open(file_path, "rb") as audio_file:
                    create_kwargs = {
                        "model": openai_model,
                        "file": audio_file,
                    }
                    if prompt is not None:
                        create_kwargs["prompt"] = prompt
                    if language is not None:
                        create_kwargs["language"] = language
                    transcript = client.audio.transcriptions.create(**create_kwargs)
                    return transcript.text
            except Exception as exc:
                error_msg = str(exc)
                logger.warning(f"OpenAI transcription error: {error_msg[:200]}")
                # Fall through to final error message

    # Final fallback: Try OpenRouter (may not support audio)
    api_key = get_openrouter_api_key()
    if not api_key:
        return "Błąd: Brak klucza API. Ustaw GROQ_API_KEY dla transkrypcji audio (Groq Whisper Large V3 Turbo)."

    if OpenAI is None:
        return "Błąd: Brak biblioteki OpenAI/OpenRouter. Ustaw GROQ_API_KEY dla transkrypcji audio."

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
            create_kwargs = {
                "model": openrouter_model,
                "file": audio_file,
            }
            if prompt is not None:
                create_kwargs["prompt"] = prompt
            if language is not None:
                create_kwargs["language"] = language
            transcript = client.audio.transcriptions.create(**create_kwargs)
            return transcript.text
    except Exception as exc:
        error_msg = str(exc)
        # Check for 405 error specifically
        if "405" in error_msg or "Method Not Allowed" in error_msg:
            return "Błąd 405: OpenRouter nie obsługuje transkrypcji audio. Ustaw GROQ_API_KEY dla transkrypcji."
        return f"Błąd transkrypcji przez OpenRouter: {error_msg}"


