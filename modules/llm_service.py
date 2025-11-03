import requests
from typing import List, Dict, Optional

from .config import get_openrouter_api_key


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
    model_name: str = "openai/whisper-1",
    prompt: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Transcribe audio using OpenRouter Whisper-compatible endpoint.

    Returns text or error string. This is a convenience wrapper; see
    modules.audio_processor for additional utilities.
    """
    api_key = get_openrouter_api_key()
    if not api_key:
        raise ValueError("OpenRouter API key not available.")

    files = {
        "file": (file_path, open(file_path, "rb"), "application/octet-stream"),
    }
    data = {"model": model_name}
    if prompt:
        data["prompt"] = prompt
    if language:
        data["language"] = language

    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://example.com",
                "X-Title": "med-sim-app",
            },
            data=data,
            files=files,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")
    except requests.exceptions.RequestException as exc:
        return f"Transcription request failed: {exc}"
    except Exception as exc:
        return f"Unexpected transcription error: {exc}"


