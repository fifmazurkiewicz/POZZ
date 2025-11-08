"""
Live Transcription + Role Recognition z użyciem Gradio
Transkrypcja przez Groq Whisper + rozpoznawanie ról przez LLM (lekarz/pacjent)
"""

import os
import io
import time
import queue
import threading
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Ładowanie .env.test jeśli istnieje
try:
    from dotenv import load_dotenv
    if os.path.exists(".env.test"):
        load_dotenv(".env.test", override=True)
        print("✓ Załadowano .env.test")
except ImportError:
    pass

import numpy as np
import gradio as gr
from openai import OpenAI

# Groq dla transkrypcji
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq nie jest zainstalowany - zainstaluj: pip install groq")

# SoundDevice dla streamingu audio w czasie rzeczywistym
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️ sounddevice nie jest zainstalowany - streaming audio nie będzie działał")

# ==================== KONFIG ====================
CHUNK_SECONDS = 10          # długość kawałka dla transkrypcji
TARGET_SR = 16000           # docelowa próbka WAV
MAX_RENDER_LINES = 120

# Pobierz konfigurację z env vars
def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    """Pobiera wartość z env vars"""
    return os.getenv(key, default)

GROQ_API_KEY = get_config("GROQ_API_KEY")
OPENROUTER_API_KEY = get_config("OPENROUTER_API_KEY")
GEMINI_FLASH_MODEL = get_config("GEMINI_FLASH_MODEL", "google/gemini-2.5-flash-lite") or "google/gemini-2.5-flash-lite"
WHISPER_MODEL = "whisper-large-v3"  # Groq obsługuje tylko large-v3
FORCED_LANG = get_config("LANG", "pl") or "pl"  # Polski domyślnie

# Sprawdź klucze API
if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY nie jest ustawiony! Sprawdź .env.test")
if not OPENROUTER_API_KEY:
    raise ValueError("⚠️ OPENROUTER_API_KEY nie jest ustawiony! Sprawdź .env.test")

# Klienci API
if GROQ_AVAILABLE:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

or_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ==================== POMOCNICZE ====================
def to_wav_bytes(pcm_float32: np.ndarray, sample_rate: int = TARGET_SR) -> bytes:
    import soundfile as sf
    # mono
    if pcm_float32.ndim > 1:
        pcm_float32 = pcm_float32.mean(axis=0)
    pcm = np.clip(pcm_float32, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm16, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    role: Optional[str] = None  # "lekarz" lub "pacjent"

@dataclass
class ConversationState:
    started_at: float = field(default_factory=time.time)
    transcript: List[TranscriptSegment] = field(default_factory=list)
    role_mapping: Dict[str, str] = field(default_factory=dict)  # Mapowanie mówców na role

# Globalny stan rozmowy
conversation_state = ConversationState()

# Bezpieczne przechowywanie danych do UI
_ui_data_lock = threading.Lock()
_ui_data = {
    "transcript_text": "",
    "roles_text": "_Oczekiwanie na transkrypcję..._",
    "chunks_created": 0,
    "chunks_processed": 0,
    "processing_status": "⏸️ Oczekiwanie na chunki...",
    "processing_progress": 0.0,
    "current_chunk": 0,
    "total_chunks": 0,
}

# Kolejka audio
audio_q: "queue.Queue[tuple[bytes, float]]" = queue.Queue()

# ==================== TRANSKRYPCJA ====================
def transcribe_chunk_with_groq(wav_bytes: bytes, lang: str = "pl") -> str:
    """Transkrybuje chunk audio używając Groq Whisper API"""
    if not GROQ_AVAILABLE or not groq_client:
        return ""
    
    # Zapisz do tymczasowego pliku
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(wav_bytes)
        tmp_path = tmp_file.name
    
    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = groq_client.audio.transcriptions.create(
                file=audio_file,
                model=WHISPER_MODEL,
                language=lang,
                response_format="text"
            )
        if transcript is None:
            return ""
        return transcript if isinstance(transcript, str) else (transcript.text if hasattr(transcript, 'text') else str(transcript))
    except Exception as e:
        print(f"[Transcribe] ❌ Błąd transkrypcji: {e}")
        return ""
    finally:
        # Usuń tymczasowy plik
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ==================== ROZPOZNAWANIE RÓL ====================
def recognize_roles_with_llm(transcript_text: str) -> Dict[str, str]:
    """
    Wysyła transkrypcję do LLM i prosi o rozpoznanie ról (lekarz/pacjent).
    Zwraca dict z mapowaniem mówców na role.
    """
    if not transcript_text or not transcript_text.strip():
        return {}
    
    prompt = f"""Analizujesz transkrypcję rozmowy między lekarzem a pacjentem. 
Rozpoznaj kto jest lekarzem, a kto pacjentem na podstawie treści wypowiedzi.

Transkrypcja:
{transcript_text}

Zwróć odpowiedź w formacie JSON:
{{
    "lekarz": "opis jak rozpoznałeś lekarza (np. 'Mówi o diagnozie, przepisuje leki, zadaje pytania medyczne')",
    "pacjent": "opis jak rozpoznałeś pacjenta (np. 'Opisuje objawy, odpowiada na pytania medyczne')",
    "pewnosc": "wysoka" lub "średnia" lub "niska"
}}

Jeśli nie jesteś pewien, zaznacz pewność jako "niska"."""

    try:
        resp = or_client.chat.completions.create(
            model=GEMINI_FLASH_MODEL,
            messages=[
                {"role": "system", "content": "Jesteś ekspertem w analizie rozmów medycznych. Rozpoznaj role mówców na podstawie treści."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        if not content:
            return {"error": "Brak odpowiedzi z LLM"}
        
        # Spróbuj wyciągnąć JSON z odpowiedzi
        import json
        import re
        
        # Usuń markdown code blocks jeśli są
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:].strip()
        elif content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()
        
        # Spróbuj sparsować JSON
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            # Spróbuj wyciągnąć JSON z tekstu
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            return {"error": "Nie udało się sparsować odpowiedzi LLM"}
    except Exception as e:
        print(f"[LLM] ❌ Błąd rozpoznawania ról: {e}")
        return {"error": str(e)}

# ==================== WORKER ====================
def worker_loop():
    """Worker w tle - przetwarza chunki audio"""
    while True:
        try:
            wav_bytes, chunk_offset = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        print(f"[Worker] 🎯 OTRZYMANO CHUNK! Offset: {chunk_offset:.2f}s, rozmiar: {len(wav_bytes)} bajtów")
        print("[Worker] Rozpoczynam transkrypcję...")
        
        # Aktualizuj status - rozpoczęcie przetwarzania
        with _ui_data_lock:
            _ui_data["current_chunk"] = _ui_data.get("chunks_processed", 0) + 1
            _ui_data["total_chunks"] = _ui_data.get("chunks_created", 0)
            _ui_data["processing_status"] = f"🔄 Przetwarzanie chunka {_ui_data['current_chunk']}/{_ui_data['total_chunks']}..."
            _ui_data["processing_progress"] = 0.1
        
        try:
            import time
            start_time = time.time()
            
            # Aktualizuj progress - transkrypcja w toku
            with _ui_data_lock:
                _ui_data["processing_progress"] = 0.3
            
            # Transkrypcja przez Groq
            transcript_text = transcribe_chunk_with_groq(wav_bytes, lang=FORCED_LANG)
            elapsed_time = time.time() - start_time
            print(f"[Worker] ✅ Transkrypcja zakończona! Tekst: {transcript_text[:100]}... (czas: {elapsed_time:.1f}s)")
            
            if not transcript_text or not transcript_text.strip():
                print("[Worker] ⚠️ Brak tekstu w transkrypcji!")
                with _ui_data_lock:
                    _ui_data["processing_progress"] = 0.0
                continue
            
            # Aktualizuj progress - transkrypcja zakończona, rozpoznawanie ról
            with _ui_data_lock:
                _ui_data["processing_progress"] = 0.6
            
            # Dodaj segment do transkrypcji
            segment = TranscriptSegment(
                start=chunk_offset,
                end=chunk_offset + CHUNK_SECONDS,
                text=transcript_text
            )
            conversation_state.transcript.append(segment)
            
            # Rozpoznaj role przez LLM (co kilka chunków lub gdy mamy wystarczająco tekstu)
            total_text = " ".join([s.text for s in conversation_state.transcript])
            if len(conversation_state.transcript) >= 2 or len(total_text) > 200:
                print("[Worker] 🤖 Rozpoznaję role przez LLM...")
                roles = recognize_roles_with_llm(total_text)
                conversation_state.role_mapping.update(roles)
            
            # Aktualizuj progress - zakończone
            with _ui_data_lock:
                _ui_data["processing_progress"] = 0.9
            
            # Zaktualizuj UI - transkrypcja
            lines = []
            for seg in conversation_state.transcript[-MAX_RENDER_LINES:]:
                role_label = seg.role or "?"
                lines.append(f"[{seg.start:6.1f}–{seg.end:6.1f}] {role_label}: {seg.text}")
            
            with _ui_data_lock:
                _ui_data["transcript_text"] = "\n".join(lines)
                _ui_data["chunks_processed"] += 1
                _ui_data["processing_progress"] = 1.0
                _ui_data["processing_status"] = f"✅ Chunek {_ui_data['chunks_processed']}/{_ui_data.get('chunks_created', 0)} przetworzony"
                print(f"[Worker] 📝 Zaktualizowano transkrypcję: {len(lines)} linii")
            
            # Zaktualizuj UI - role
            roles_display = []
            if conversation_state.role_mapping:
                roles_display.append("**Rozpoznane role:**\n")
                if "lekarz" in conversation_state.role_mapping:
                    roles_display.append(f"👨‍⚕️ **Lekarz:** {conversation_state.role_mapping['lekarz']}")
                if "pacjent" in conversation_state.role_mapping:
                    roles_display.append(f"👤 **Pacjent:** {conversation_state.role_mapping['pacjent']}")
                if "pewnosc" in conversation_state.role_mapping:
                    roles_display.append(f"\n**Pewność:** {conversation_state.role_mapping['pewnosc']}")
            else:
                roles_display.append("_Oczekiwanie na rozpoznanie ról..._")
            
            with _ui_data_lock:
                _ui_data["roles_text"] = "\n".join(roles_display)
            
            # Reset progress po krótkiej chwili
            time.sleep(0.5)
            with _ui_data_lock:
                if _ui_data.get("chunks_processed", 0) >= _ui_data.get("chunks_created", 0):
                    _ui_data["processing_status"] = "✅ Wszystkie chunki przetworzone"
                    _ui_data["processing_progress"] = 1.0
                else:
                    _ui_data["processing_progress"] = 0.0
            
        except Exception as e:
            print(f"[Worker] ❌ BŁĄD podczas przetwarzania chunka: {e}")
            import traceback
            traceback.print_exc()
            with _ui_data_lock:
                _ui_data["processing_status"] = f"❌ Błąd przetwarzania chunka {_ui_data.get('current_chunk', 0)}"
                _ui_data["processing_progress"] = 0.0

# Start worker thread
worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()
print("[Main] ✅ Worker thread uruchomiony")

# ==================== REAL-TIME AUDIO STREAMING ====================
_recording_active = False
_audio_buffer_lock = threading.Lock()
_audio_buffer = np.array([], dtype=np.float32)
_audio_stream = None
_recording_start_time = None
_chunk_counter = 0

def audio_callback(indata, frames, time_info, status):
    """Callback wywoływany przez sounddevice podczas nagrywania"""
    global _audio_buffer, _recording_active
    
    if not _recording_active:
        return
    
    if status:
        print(f"[Audio] ⚠️ Status: {status}")
    
    audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
    audio_chunk = audio_chunk.astype(np.float32)
    
    with _audio_buffer_lock:
        _audio_buffer = np.concatenate([_audio_buffer, audio_chunk])
        
        chunk_samples = CHUNK_SECONDS * TARGET_SR
        
        while len(_audio_buffer) >= chunk_samples:
            chunk = _audio_buffer[:chunk_samples]
            _audio_buffer = _audio_buffer[chunk_samples:]
            
            wav_bytes = to_wav_bytes(chunk, TARGET_SR)
            global _chunk_counter
            offset = _chunk_counter * CHUNK_SECONDS
            _chunk_counter += 1
            audio_q.put((wav_bytes, offset))
            
            print(f"[Stream] ⏰ CHUNK GOTOWY! {CHUNK_SECONDS}s, offset: {offset:.1f}s, bufor: {len(_audio_buffer)/TARGET_SR:.1f}s")
            with _ui_data_lock:
                _ui_data["chunks_created"] += 1

def start_recording():
    """Rozpoczyna nagrywanie z mikrofonu"""
    global _recording_active, _audio_stream, _recording_start_time, _chunk_counter, _audio_buffer
    
    if not SOUNDDEVICE_AVAILABLE:
        return "❌ sounddevice nie jest zainstalowany! Zainstaluj: pip install sounddevice", "", ""
    
    if _recording_active:
        return "⚠️ Nagrywanie już trwa!", "", ""
    
    try:
        _recording_active = True
        _recording_start_time = time.time()
        _chunk_counter = 0
        with _audio_buffer_lock:
            _audio_buffer = np.array([], dtype=np.float32)
        
        conversation_state.started_at = _recording_start_time
        conversation_state.transcript = []
        conversation_state.role_mapping = {}
        
        _audio_stream = sd.InputStream(
            samplerate=TARGET_SR,
            channels=1,
            dtype=np.float32,
            callback=audio_callback,
            blocksize=int(TARGET_SR * 0.1)
        )
        _audio_stream.start()
        
        print(f"[Stream] 🎵 ROZPOCZĘTO NAGRYWANIE! Sample rate: {TARGET_SR}Hz")
        
        with _ui_data_lock:
            _ui_data["chunks_created"] = 0
            _ui_data["chunks_processed"] = 0
            _ui_data["processing_status"] = "⏸️ Oczekiwanie na chunki..."
            _ui_data["processing_progress"] = 0.0
        
        transcript, roles, status, proc_status, proc_progress = get_current_status()
        return "✅ Nagrywanie rozpoczęte! Mów do mikrofonu...", transcript, roles
    except Exception as e:
        _recording_active = False
        error_msg = f"❌ Błąd rozpoczęcia nagrywania: {e}"
        print(f"[Stream] {error_msg}")
        return error_msg, "", ""

def stop_recording():
    """Zatrzymuje nagrywanie"""
    global _recording_active, _audio_stream
    
    if not _recording_active:
        transcript, roles, status, proc_status, proc_progress = get_current_status()
        return "⚠️ Nagrywanie nie jest aktywne!", transcript, roles, status, proc_status, proc_progress
    
    try:
        _recording_active = False
        
        if _audio_stream is not None:
            _audio_stream.stop()
            _audio_stream.close()
            _audio_stream = None
        
        with _audio_buffer_lock:
            if len(_audio_buffer) > 0:
                remaining_seconds = len(_audio_buffer) / TARGET_SR
                print(f"[Stream] ⏹️ Nagranie zakończone. Reszta w buforze: {remaining_seconds:.2f}s")
        
        print("[Stream] ⏹️ Nagrywanie zatrzymane")
        
        transcript, roles, status, proc_status, proc_progress = get_current_status()
        return "⏹️ Nagrywanie zatrzymane", transcript, roles, status, proc_status, proc_progress
    except Exception as e:
        error_msg = f"❌ Błąd zatrzymania nagrywania: {e}"
        print(f"[Stream] {error_msg}")
        return error_msg, "", "", "", "❌ Błąd", 0

def reset_audio_buffer():
    """Reset stanu nagrywania"""
    global _recording_active, _audio_stream, _audio_buffer, _chunk_counter
    
    if _recording_active:
        stop_recording()
    
    conversation_state.transcript = []
    conversation_state.role_mapping = {}
    conversation_state.started_at = time.time()
    _chunk_counter = 0
    
    with _audio_buffer_lock:
        _audio_buffer = np.array([], dtype=np.float32)
    
    with _ui_data_lock:
        _ui_data["transcript_text"] = ""
        _ui_data["roles_text"] = "_Oczekiwanie na transkrypcję..._"
        _ui_data["chunks_created"] = 0
        _ui_data["chunks_processed"] = 0
        _ui_data["processing_status"] = "⏸️ Oczekiwanie na chunki..."
        _ui_data["processing_progress"] = 0.0
    
    print("[Reset] 🔄 Stan zresetowany")

def get_current_status():
    """Zwraca aktualny status dla auto-refresh"""
    with _ui_data_lock:
        transcript = _ui_data.get("transcript_text", "")
        roles = _ui_data.get("roles_text", "_Oczekiwanie na transkrypcję..._")
        chunks_created = _ui_data.get("chunks_created", 0)
        chunks_processed = _ui_data.get("chunks_processed", 0)
        processing_status = _ui_data.get("processing_status", "⏸️ Oczekiwanie na chunki...")
        processing_progress = _ui_data.get("processing_progress", 0.0)
    
    status = f"""
**📊 Statystyki:**
- Chunki utworzone: {chunks_created}
- Chunki przetworzone: {chunks_processed}
- Segmenty: {len(conversation_state.transcript)}
- Kolejka: {audio_q.qsize()}
"""
    progress_percent = int(processing_progress * 100)
    return transcript, roles, status, processing_status, progress_percent

# Tworzenie UI
with gr.Blocks(title="Live Transcription + Role Recognition") as demo:
    gr.Markdown("# 🎙️ Transkrypcja na żywo + Rozpoznawanie ról")
    gr.Markdown("**Streaming w czasie rzeczywistym:** Kliknij 'Start nagrywania' i mów do mikrofonu. Co 10 sekund audio jest automatycznie transkrybowane przez Groq Whisper i analizowane przez LLM w celu rozpoznania ról (lekarz/pacjent).")
    
    with gr.Row():
        with gr.Column(scale=1):
            transcript_output = gr.Textbox(
                label="📝 Transkrypcja",
                lines=20,
                max_lines=30,
                interactive=False,
                placeholder="_Oczekiwanie na audio..._"
            )
        with gr.Column(scale=1):
            roles_output = gr.Markdown(
                label="👥 Rozpoznane role",
                value="_Oczekiwanie na transkrypcję..._"
            )
            status_output = gr.Markdown(
                label="📊 Status",
                value="_Oczekiwanie..._"
            )
    
    # Progress bar dla przetwarzania chunków
    with gr.Row():
        processing_status_text = gr.Textbox(
            label="🔄 Status przetwarzania",
            value="⏸️ Oczekiwanie na chunki...",
            interactive=False
        )
        processing_progress_bar = gr.Slider(
            label="Postęp",
            minimum=0,
            maximum=100,
            value=0,
            interactive=False,
            info="Postęp przetwarzania chunka (0-100%)"
        )
    
    # Przyciski kontroli nagrywania
    with gr.Row():
        start_btn = gr.Button("▶️ Start nagrywania", variant="primary")
        stop_btn = gr.Button("⏹️ Stop nagrywania", variant="stop")
        reset_btn = gr.Button("🔄 Reset", variant="secondary")
        refresh_btn = gr.Button("🔄 Odśwież status", variant="secondary")
    
    # Status nagrywania
    recording_status = gr.Textbox(
        label="📊 Status nagrywania",
        value="⏸️ Nagrywanie nieaktywne",
        interactive=False
    )
    
    # Inicjalizacja przy starcie
    demo.load(
        fn=get_current_status,
        inputs=None,
        outputs=[transcript_output, roles_output, status_output, processing_status_text, processing_progress_bar]
    )
    
    # Automatyczne odświeżanie - użyjmy prostego podejścia z JavaScript
    # Dodajemy skrypt JavaScript który będzie wywoływał funkcję odświeżającą co 2 sekundy
    demo.load(
        fn=None,
        js="""
        () => {
            setInterval(() => {
                // Znajdź przycisk refresh po tekście
                const buttons = Array.from(document.querySelectorAll('button'));
                const refreshBtn = buttons.find(btn => btn.textContent && btn.textContent.includes('Odśwież status'));
                if (refreshBtn && !refreshBtn.disabled) {
                    refreshBtn.click();
                }
            }, 2000);
            return [];
        }
        """
    )
    
    # Ręczne odświeżanie statusu
    refresh_btn.click(
        fn=get_current_status,
        inputs=None,
        outputs=[transcript_output, roles_output, status_output, processing_status_text, processing_progress_bar]
    )
    
    # Kontrola nagrywania
    start_btn.click(
        fn=start_recording,
        inputs=None,
        outputs=[recording_status, transcript_output, roles_output]
    )
    
    stop_btn.click(
        fn=stop_recording,
        inputs=None,
        outputs=[recording_status, transcript_output, roles_output, status_output, processing_status_text, processing_progress_bar]
    )
    
    # Reset bufora
    reset_btn.click(
        fn=reset_audio_buffer,
        inputs=None,
        outputs=None
    )
    
    gr.Markdown("---")
    gr.Markdown("**Silnik:** Groq Whisper Large V3 | **Rozpoznawanie ról:** Gemini Flash przez OpenRouter")

if __name__ == "__main__":
    print("[Main] 🚀 Uruchamiam Gradio...")
    demo.queue()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
