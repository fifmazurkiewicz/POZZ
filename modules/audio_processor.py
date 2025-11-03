import os
from typing import Optional

from .llm_service import transcribe_audio as _transcribe_via_openrouter


def transcribe_audio_file(
    file_path: str,
    model_name: str = "openai/whisper-1",
    language: Optional[str] = None,
 ) -> str:
     """Transcribe an audio file using OpenRouter Whisper-compatible endpoint.

     Returns plaintext transcription or error string.
     """
     if not os.path.isfile(file_path):
         return f"Audio file not found: {file_path}"

     return _transcribe_via_openrouter(
         file_path=file_path,
         model_name=model_name,
         language=language,
     )


