"""
Live Transcription + Diarization z użyciem Gradio
Transkrypcja przez Groq Whisper + rozpoznawanie mówców przez whisper-diarization (CUDA)
"""

import os
import io
import re
import sys
import time
import glob
import queue
import shutil
import threading
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

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
import requests

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
CHUNK_OVERLAP_SECONDS = 2.0  # overlap między chunkami (aby nie ucinać zdań) - 2 sekundy
TARGET_SR = 16000           # docelowa próbka WAV
MAX_RENDER_LINES = 120

# Pobierz konfigurację z env vars
def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    """Pobiera wartość z env vars"""
    return os.getenv(key, default)

GROQ_API_KEY = get_config("GROQ_API_KEY")
DEEPGRAM_API_KEY = get_config("DEEPGRAM_API_KEY")
WHISPER_DIARIZATION_DIR = get_config("WHISPER_DIARIZATION_DIR", "./whisper-diarization") or "./whisper-diarization"
FORCED_LANG = get_config("LANG", "pl") or "pl"  # Polski domyślnie
DEVICE_CONFIG = get_config("DEVICE", "cuda") or "cuda"  # CUDA domyślnie
WHISPER_MODEL = get_config("WHISPER_MODEL", "small") or "small"  # Domyślnie small (dla GTX 1050 Ti), można zmienić na medium/large-v3

# Wybór silnika diarization
USE_DEEPGRAM = get_config("USE_DEEPGRAM", "true") or "true"
USE_DEEPGRAM = USE_DEEPGRAM.lower() == "true"

# Sprawdź klucze API
if USE_DEEPGRAM:
    if not DEEPGRAM_API_KEY:
        raise ValueError("⚠️ DEEPGRAM_API_KEY nie jest ustawiony! Sprawdź .env.test")
    print("[Config] ✅ Używam Deepgram API do diarization")
else:
    if not GROQ_API_KEY:
        raise ValueError("⚠️ GROQ_API_KEY nie jest ustawiony! Sprawdź .env.test")
    print("[Config] ✅ Używam whisper-diarization (lokalne)")

# Sprawdź czy CUDA jest dostępne
def check_cuda_available():
    """Sprawdza czy CUDA jest dostępne i działa"""
    if DEVICE_CONFIG.lower() != "cuda":
        return False, "DEVICE nie jest ustawiony na 'cuda'"
    
    # Sprawdź torch
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            driver_version = torch.cuda.get_driver_version()
            print(f"[Config] ✓ PyTorch widzi CUDA:")
            print(f"    Urządzenie: {device_name}")
            print(f"    CUDA Runtime: {cuda_version}")
            print(f"    Sterownik CUDA: {driver_version}")
            
            # Spróbuj utworzyć tensor na CUDA (sprawdzi czy sterownik działa)
            try:
                x = torch.tensor([1.0]).cuda()
                del x
                torch.cuda.empty_cache()
                print(f"[Config] ✓ Test CUDA zakończony pomyślnie")
                return True, "PyTorch CUDA działa"
            except RuntimeError as e:
                error_msg = str(e)
                if "driver version is insufficient" in error_msg.lower():
                    print(f"[Config] ❌ Sterownik CUDA jest za stary!")
                    print(f"    Wymagana wersja: {cuda_version}")
                    print(f"    Zainstalowana wersja: {driver_version}")
                    return False, f"Sterownik CUDA za stary (wymagane: {cuda_version}, masz: {driver_version})"
                else:
                    print(f"[Config] ⚠️ Błąd testu CUDA: {e}")
                    return False, f"Błąd testu CUDA: {e}"
        else:
            return False, "PyTorch zainstalowany, ale CUDA nie jest dostępne"
    except ImportError:
        print("[Config] ⚠️ PyTorch nie jest zainstalowany - sprawdzam ctranslate2...")
    except Exception as e:
        print(f"[Config] ⚠️ Błąd sprawdzania PyTorch CUDA: {e}")
    
    # Sprawdź ctranslate2
    try:
        import ctranslate2
        devices = ctranslate2.get_supported_compute_types("cuda")
        if len(devices) > 0:
            print(f"[Config] ✓ ctranslate2 widzi CUDA: {devices}")
            return True, "ctranslate2 CUDA działa"
        else:
            return False, "ctranslate2 zainstalowany, ale CUDA nie jest dostępne"
    except ImportError:
        print("[Config] ⚠️ ctranslate2 nie jest zainstalowany")
    except Exception as e:
        print(f"[Config] ⚠️ Błąd sprawdzania ctranslate2 CUDA: {e}")
    
    return False, "Brak zainstalowanych bibliotek CUDA (torch lub ctranslate2)"

# Opcja wymuszenia CUDA (pomiń sprawdzanie)
FORCE_CUDA_STR = get_config("FORCE_CUDA", "false") or "false"
FORCE_CUDA = FORCE_CUDA_STR.lower() == "true"

# Globalna flaga - czy CUDA nie działa (aby nie próbować za każdym razem)
CUDA_FAILED = False

# Automatyczny fallback na CPU jeśli CUDA nie działa
DEVICE = DEVICE_CONFIG
if DEVICE_CONFIG.lower() == "cuda":
    if FORCE_CUDA:
        print("[Config] ⚠️ FORCE_CUDA=true - używam CUDA bez sprawdzania")
        DEVICE = "cuda"
    else:
        cuda_available, reason = check_cuda_available()
        if not cuda_available:
            print(f"[Config] ⚠️ CUDA nie jest dostępne: {reason}")
            print("[Config] 💡 Rozwiązania:")
            print("   1. Zainstaluj PyTorch z CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("   2. LUB zainstaluj ctranslate2 z CUDA")
            print("   3. LUB ustaw FORCE_CUDA=true w .env.test aby wymusić CUDA (może nie działać)")
            print("   4. LUB ustaw DEVICE=cpu w .env.test")
            print("[Config] ⚠️ Przełączam na CPU...")
            DEVICE = "cpu"
        else:
            print("[Config] ✓ CUDA jest dostępne - używam GPU")
elif DEVICE_CONFIG.lower() == "cpu":
    print("[Config] ✓ Używam CPU (może być wolniejsze niż CUDA, ale zawsze działa)")
else:
    print(f"[Config] ⚠️ Nieznane urządzenie '{DEVICE_CONFIG}' - używam CPU")
    DEVICE = "cpu"

# Klienci API
if GROQ_AVAILABLE:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

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

def parse_srt(srt_path: str):
    """
    Parsuje prosty plik SRT zwracając listę: [(start_sec, end_sec, speaker_label, text), ...]
    """
    out = []
    time_pat = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d\d\d)")
    speaker_pat = re.compile(r"^(?:Speaker\s*(\d+)|SPEAKER[_\s]?(\d+))\s*:\s*(.*)$", re.IGNORECASE)
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        block = []
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if block and len(block) >= 2:
                    m = time_pat.search(block[1])
                    if m:
                        h1,m1,s1,ms1,h2,m2,s2,ms2 = map(int, m.groups())
                        start = h1*3600 + m1*60 + s1 + ms1/1000.0
                        end   = h2*3600 + m2*60 + s2 + ms2/1000.0
                        text_lines = block[2:] if len(block) > 2 else []
                        text = " ".join(t.strip() for t in text_lines if t.strip())
                        spk = None
                        msp = speaker_pat.match(text)
                        if msp:
                            spk = msp.group(1) or msp.group(2)
                            text = msp.group(3).strip()
                            spk = f"Speaker {spk}"
                        out.append((start, end, spk, text))
                block = []
            else:
                block.append(line)
        # ostatni blok
        if block and len(block) >= 2:
            m = time_pat.search(block[1])
            if m:
                h1,m1,s1,ms1,h2,m2,s2,ms2 = map(int, m.groups())
                start = h1*3600 + m1*60 + s1 + ms1/1000.0
                end   = h2*3600 + m2*60 + s2 + ms2/1000.0
                text_lines = block[2:] if len(block) > 2 else []
                text = " ".join(t.strip() for t in text_lines if t.strip())
                spk = None
                msp = speaker_pat.match(text)
                if msp:
                    spk = msp.group(1) or msp.group(2)
                    text = msp.group(3).strip()
                    spk = f"Speaker {spk}"
                out.append((start, end, spk, text))
    return out

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

@dataclass
class ConversationState:
    started_at: float = field(default_factory=time.time)
    transcript: List[Segment] = field(default_factory=list)

# Globalny stan rozmowy
conversation_state = ConversationState()

# Bezpieczne przechowywanie danych do UI
_ui_data_lock = threading.Lock()
_ui_data = {
    "transcript_text": "",
    "speakers_text": "_Oczekiwanie na transkrypcję..._",
    "chunks_created": 0,
    "chunks_processed": 0,
    "processing_status": "⏸️ Oczekiwanie na chunki...",
    "processing_progress": 0.0,
    "current_chunk": 0,
    "total_chunks": 0,
}

# Kolejka audio
audio_q: "queue.Queue[tuple[bytes, float]]" = queue.Queue()

# Globalna flaga - czy model został wczytany
MODEL_PRELOADED = False
MODEL_PRELOAD_LOCK = threading.Lock()

# ==================== PRELOAD MODEL ====================
def preload_whisper_model(device: str, compute_type: str, model_name: str) -> bool:
    """
    Wstępnie wczytuje model Whisper do pamięci GPU/CPU z progress barem.
    Zwraca True jeśli sukces, False jeśli błąd.
    """
    global MODEL_PRELOADED
    
    with MODEL_PRELOAD_LOCK:
        if MODEL_PRELOADED:
            print("[Preload] ✅ Model już został wczytany wcześniej")
            return True
        
        print("\n" + "=" * 80)
        print("[Preload] 🚀 ROZPOCZYNAM WCZYTYWANIE MODELU WHISPER")
        print("=" * 80)
        print(f"[Preload] 📋 Model: {model_name}")
        print(f"[Preload] 🔧 Device: {device}")
        print(f"[Preload] 🔧 Compute Type: {compute_type}")
        print(f"[Preload] ⏳ To może zająć 10-60 sekund...")
        print()
        
        try:
            import time
            import faster_whisper
            
            # Sprawdź użycie GPU przed wczytaniem
            if device.lower() == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        mem_before = torch.cuda.memory_allocated(0) / 1024**2  # MB
                        print(f"[Preload] 📊 GPU Memory przed wczytaniem: {mem_before:.1f} MB")
                        print(f"[Preload] 💡 Sprawdź nvidia-smi - powinno pokazać użycie GPU podczas wczytywania")
                except Exception as e:
                    print(f"[Preload] ⚠️ Nie można sprawdzić GPU memory: {e}")
            
            # Progress bar w logach
            start_time = time.time()
            steps = [
                "Inicjalizacja faster-whisper...",
                "Pobieranie modelu (jeśli potrzebne)...",
                "Wczytywanie wag do pamięci...",
                "Inicjalizacja CUDA/CPU...",
                "Gotowe!"
            ]
            
            for i, step in enumerate(steps):
                progress = (i + 1) / len(steps) * 100
                bar_length = 40
                filled = int(bar_length * (i + 1) / len(steps))
                bar = "█" * filled + "░" * (bar_length - filled)
                print(f"[Preload] [{bar}] {progress:.0f}% - {step}")
                time.sleep(0.3)  # Małe opóźnienie dla efektu
            
            # Rzeczywiste wczytywanie modelu
            print(f"[Preload] 🔄 Wczytywanie modelu {model_name}...")
            load_start = time.time()
            
            model = faster_whisper.WhisperModel(
                model_size_or_path=model_name,
                device=device,
                compute_type=compute_type
            )
            
            load_time = time.time() - load_start
            print(f"[Preload] ✅ Model wczytany w {load_time:.1f} sekund!")
            
            # Sprawdź użycie GPU po wczytaniu
            if device.lower() == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        mem_after = torch.cuda.memory_allocated(0) / 1024**2  # MB
                        mem_used = mem_after - mem_before
                        print(f"[Preload] 📊 GPU Memory po wczytaniu: {mem_after:.1f} MB")
                        print(f"[Preload] 📊 GPU Memory użyte przez model: {mem_used:.1f} MB")
                        print(f"[Preload] 💡 Sprawdź nvidia-smi - powinno pokazać użycie pamięci GPU")
                except Exception as e:
                    print(f"[Preload] ⚠️ Nie można sprawdzić GPU memory: {e}")
            
            # Test transkrypcji (krótki test)
            print(f"[Preload] 🧪 Testowanie modelu (krótki test)...")
            test_start = time.time()
            try:
                # Tworzymy krótki test audio (1 sekunda ciszy)
                import numpy as np
                test_audio = np.zeros(16000, dtype=np.float32)  # 1 sekunda @ 16kHz
                segments, info = model.transcribe(test_audio, language="pl", vad_filter=False)
                # Pobierz pierwszy segment (może być pusty, to OK)
                list(segments)  # Wymuszenie przetworzenia
                test_time = time.time() - test_start
                print(f"[Preload] ✅ Test zakończony w {test_time:.1f} sekund")
            except Exception as e:
                print(f"[Preload] ⚠️ Test nie powiódł się (ale model jest wczytany): {e}")
            
            # Zwolnij model z pamięci (zostanie wczytany ponownie w diarize.py, ale szybciej)
            del model
            if device.lower() == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
            
            total_time = time.time() - start_time
            print()
            print("=" * 80)
            print(f"[Preload] ✅ MODEL WCZYTANY POMYŚLNIE!")
            print(f"[Preload] ⏱️  Całkowity czas: {total_time:.1f} sekund")
            print("=" * 80)
            print()
            
            MODEL_PRELOADED = True
            return True
            
        except Exception as e:
            print()
            print("=" * 80)
            print(f"[Preload] ❌ BŁĄD WCZYTYWANIA MODELU!")
            print("=" * 80)
            print(f"Błąd: {e}")
            print()
            print("💡 Możliwe przyczyny:")
            print("   - Brak połączenia internetowego (pierwsze wczytywanie)")
            print("   - Za mało pamięci GPU/CPU")
            print("   - Nieprawidłowa konfiguracja CUDA")
            print()
            print("⚠️  Aplikacja będzie próbować wczytać model przy pierwszym użyciu")
            print("=" * 80)
            print()
            return False

# ==================== DEEPGRAM DIARIZATION ====================
def run_deepgram_diarization_on_chunk(wav_bytes: bytes, lang: str) -> List[Segment]:
    """
    Używa Deepgram API do transkrypcji i diarization.
    Zwraca listę Segmentów w czasie LOKALNYM chunku (0..N sek).
    """
    print(f"[DEEPGRAM] 🚀 Rozpoczynam transkrypcję i diarization przez Deepgram API...")
    
    try:
        import time
        start_time = time.time()
        
        # Przygotuj URL z parametrami
        url = "https://api.deepgram.com/v1/listen"
        params = {
            "diarize": "true",
            "punctuate": "true",
            "utterances": "true",
            "smart_format": "true",  # Lepsze formatowanie może pomóc w diarization
            "model": "nova-2",  # Najnowszy model - lepsze diarization
            "detect_language": "false",  # Wyłącz auto-detekcję, używamy wymuszonego języka
        }
        
        # Dodaj język jeśli podano
        if lang and lang.lower() != "auto":
            # Deepgram używa kodów ISO 639-1 (pl, en, de, fr, etc.)
            # Dla polskiego używamy "pl"
            lang_map = {
                "pl": "pl",
                "polish": "pl",
                "en": "en",
                "english": "en",
                "de": "de",
                "german": "de",
                "fr": "fr",
                "french": "fr",
            }
            lang_code = lang_map.get(lang.lower(), lang.lower())
            params["language"] = lang_code
            print(f"[DEEPGRAM] 🌐 Ustawiono język: {lang_code} (z {lang})")
        
        # Przygotuj nagłówki
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav",
        }
        
        print(f"[DEEPGRAM] 📤 Wysyłam {len(wav_bytes)} bajtów audio do Deepgram API...")
        
        # Wyślij request
        response = requests.post(
            url,
            params=params,
            headers=headers,
            data=wav_bytes,
            timeout=120  # 2 minuty timeout
        )
        
        response.raise_for_status()
        
        elapsed = time.time() - start_time
        print(f"[DEEPGRAM] ✅ Odpowiedź otrzymana w {elapsed:.1f}s")
        
        # Parsuj odpowiedź JSON
        result = response.json()
        
        # Debug: wyświetl strukturę odpowiedzi
        print(f"[DEEPGRAM] 🔍 Debug - struktura odpowiedzi:")
        if "results" in result:
            if "utterances" in result["results"]:
                print(f"[DEEPGRAM]   - utterances: {len(result['results']['utterances'])}")
            if "channels" in result["results"]:
                print(f"[DEEPGRAM]   - channels: {len(result['results']['channels'])}")
                for i, channel in enumerate(result["results"]["channels"]):
                    if "alternatives" in channel:
                        for j, alt in enumerate(channel["alternatives"]):
                            if "words" in alt:
                                print(f"[DEEPGRAM]   - channel[{i}].alternatives[{j}].words: {len(alt['words'])}")
                                # Sprawdź unikalnych mówców w words (WSZYSTKIE słowa, nie tylko pierwsze 10)
                                speakers = set()
                                for word in alt["words"]:
                                    if "speaker" in word:
                                        speakers.add(word["speaker"])
                                print(f"[DEEPGRAM]   - Unikalni mówcy w words (wszystkie {len(alt['words'])} słów): {speakers}")
        
        segments = []
        
        # Sprawdź najpierw words - mogą mieć lepsze informacje o mówcach
        words_with_speakers = []
        if "results" in result and "channels" in result["results"]:
            for channel in result["results"]["channels"]:
                if "alternatives" in channel:
                    for alt in channel["alternatives"]:
                        if "words" in alt:
                            for word in alt["words"]:
                                if "speaker" in word:
                                    words_with_speakers.append(word)
        
        # Sprawdź unikalnych mówców w words
        speakers_in_words = set()
        for word in words_with_speakers:
            speakers_in_words.add(word.get("speaker", 0))
        print(f"[DEEPGRAM] 🔍 Unikalni mówcy w words: {speakers_in_words} (łącznie {len(words_with_speakers)} słów z informacją o mówcy)")
        
        # Spróbuj użyć utterances (jeśli dostępne) - są już pogrupowane
        if "results" in result and "utterances" in result["results"]:
            utterances = result["results"]["utterances"]
            print(f"[DEEPGRAM] 📝 Otrzymano {len(utterances)} utterances z diarization")
            
            # Sprawdź unikalnych mówców w utterances
            speakers_in_utterances = set()
            for utterance in utterances:
                speaker = utterance.get("speaker", 0)
                speakers_in_utterances.add(speaker)
            print(f"[DEEPGRAM] 🔍 Unikalni mówcy w utterances: {speakers_in_utterances}")
            
            # Jeśli words mają więcej mówców niż utterances, użyj words zamiast utterances
            use_words_instead = False
            if len(speakers_in_words) > len(speakers_in_utterances) and len(speakers_in_words) > 1:
                print(f"[DEEPGRAM] ⚠️ Words mają więcej mówców ({len(speakers_in_words)}) niż utterances ({len(speakers_in_utterances)}) - używam words!")
                use_words_instead = True
            elif len(speakers_in_utterances) == 1 and len(speakers_in_words) > 1:
                print(f"[DEEPGRAM] ⚠️ Utterances mają tylko 1 mówcę, ale words mają {len(speakers_in_words)} - używam words!")
                use_words_instead = True
            
            if not use_words_instead:
                # Użyj utterances
                for utterance in utterances:
                    start = utterance.get("start", 0.0)
                    end = utterance.get("end", 0.0)
                    transcript = utterance.get("transcript", "").strip()
                    speaker = utterance.get("speaker", 0)
                    
                    if transcript:  # Tylko jeśli jest tekst
                        speaker_str = f"Speaker {speaker}"
                        segments.append(Segment(
                            start=start,
                            end=end,
                            text=transcript,
                            speaker=speaker_str
                        ))
                
                # Jeśli mamy segmenty z utterances, zwróć je
                if segments:
                    print(f"[DEEPGRAM] ✅ Utworzono {len(segments)} segmentów z utterances")
                    return segments
        
        # Fallback: użyj words i grupuj ręcznie (lub jeśli words mają więcej mówców)
        words = []
        if "results" in result and "channels" in result["results"]:
            for channel in result["results"]["channels"]:
                if "alternatives" in channel:
                    for alt in channel["alternatives"]:
                        if "words" in alt:
                            words.extend(alt["words"])
        
        if not words:
            print("[DEEPGRAM] ⚠️ Brak słów w odpowiedzi!")
            return []
        
        print(f"[DEEPGRAM] 📝 Otrzymano {len(words)} słów z diarization (grupowanie ręczne)")
        
        # Grupuj słowa w segmenty według mówcy
        current_segment = None
        
        for word_data in words:
            word_text = word_data.get("word", "")
            start = word_data.get("start", 0.0)
            end = word_data.get("end", 0.0)
            speaker = word_data.get("speaker", 0)
            
            # Konwertuj speaker na string (Deepgram używa liczb: 0, 1, 2, ...)
            speaker_str = f"Speaker {speaker}"
            
            # Jeśli to pierwsze słowo lub mówca się zmienił, rozpocznij nowy segment
            if current_segment is None or current_segment.speaker != speaker_str:
                # Zapisz poprzedni segment jeśli istnieje
                if current_segment is not None:
                    segments.append(current_segment)
                
                # Rozpocznij nowy segment
                current_segment = Segment(
                    start=start,
                    end=end,
                    text=word_text,
                    speaker=speaker_str
                )
            else:
                # Dodaj słowo do obecnego segmentu
                current_segment.text += " " + word_text
                current_segment.end = end  # Aktualizuj koniec segmentu
        
        # Dodaj ostatni segment
        if current_segment is not None:
            segments.append(current_segment)
        
        print(f"[DEEPGRAM] ✅ Utworzono {len(segments)} segmentów z diarization")
        return segments
        
    except requests.exceptions.RequestException as e:
        print(f"[DEEPGRAM] ❌ Błąd request do Deepgram API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"[DEEPGRAM] 📄 Szczegóły błędu: {error_detail}")
            except:
                print(f"[DEEPGRAM] 📄 Response: {e.response.text[:200]}")
        return []
    except Exception as e:
        print(f"[DEEPGRAM] ❌ Nieoczekiwany błąd: {e}")
        import traceback
        traceback.print_exc()
        return []

# ==================== WHISPER DIARIZATION (STARE) ====================
def get_gpu_compute_type(device: str) -> str:
    """
    Wykrywa architekturę GPU i zwraca odpowiedni compute_type.
    Starsze GPU (Pascal i starsze) nie wspierają efektywnego float16.
    """
    if device.lower() != "cuda":
        return "float32"  # CPU zawsze używa float32
    
    try:
        import torch
        if not torch.cuda.is_available():
            return "float32"
        
        # Pobierz compute capability GPU
        device_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        major, minor = compute_capability
        
        print(f"[DIARIZE] 🔍 GPU: {device_name}, Compute Capability: {major}.{minor}")
        
        # Architektury GPU i ich compute capability:
        # Pascal (GTX 10xx): 6.0, 6.1, 6.2 - NIE wspiera efektywnego float16
        # Volta (V100): 7.0 - wspiera float16
        # Turing (RTX 20xx): 7.5 - wspiera float16
        # Ampere (RTX 30xx): 8.0, 8.6 - wspiera float16
        # Ada Lovelace (RTX 40xx): 8.9 - wspiera float16
        # Hopper (H100): 9.0 - wspiera float16
        
        if major < 7:
            print(f"[DIARIZE] ⚠️ GPU {device_name} (compute {major}.{minor}) nie wspiera efektywnego float16")
            print(f"[DIARIZE] 💡 Używam float32 zamiast float16")
            return "float32"
        else:
            print(f"[DIARIZE] ✅ GPU {device_name} (compute {major}.{minor}) wspiera float16")
            return "float16"
    except Exception as e:
        print(f"[DIARIZE] ⚠️ Nie można wykryć architektury GPU: {e}")
        print(f"[DIARIZE] 💡 Używam bezpiecznego float32")
        return "float32"

def run_whisper_diarization_on_chunk(wav_bytes: bytes, lang: str, device: str) -> List[Segment]:
    """
    Uruchamia 'python diarize.py -a file.wav ...' i parsuje .srt.
    """
    global CUDA_FAILED
    
    # Jeśli CUDA już nie działało, od razu użyj CPU
    if CUDA_FAILED and device.lower() == "cuda":
        print("[DIARIZE] ⚠️ CUDA już wcześniej nie działało - używam CPU")
        device = "cpu"
    
    # Wykryj odpowiedni compute_type dla GPU
    compute_type = get_gpu_compute_type(device)
    tmp_dir = tempfile.mkdtemp(prefix="wd_chunk_")
    wav_path = os.path.join(tmp_dir, "chunk.wav")
    with open(wav_path, "wb") as f:
        f.write(wav_bytes)

    diarize_dir = WHISPER_DIARIZATION_DIR or "./whisper-diarization"
    diarize_script = os.path.join(diarize_dir, "diarize.py")
    
    # Sprawdź czy katalog i skrypt istnieją
    if not os.path.isdir(diarize_dir):
        print(f"[DIARIZE] ERROR: Katalog nie istnieje: {diarize_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return []
    
    if not os.path.isfile(diarize_script):
        print(f"[DIARIZE] ERROR: Skrypt nie istnieje: {diarize_script}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return []
    
    # Użyj absolutnej ścieżki dla cwd (Windows wymaga tego)
    diarize_dir_abs = os.path.abspath(diarize_dir)
    
    # Użyj tego samego interpretera Python co aplikacja
    python_executable = sys.executable
    
    # Dodaj katalog whisper-diarization do PYTHONPATH
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    
    parent_dir = os.path.dirname(diarize_dir_abs)
    ctc_aligner_dir = os.path.join(parent_dir, "ctc-forced-aligner")
    ctc_aligner_dir_alt = r"C:\Users\MSI\PycharmProjects\ctc-forced-aligner"
    
    separator = ";" if os.name == 'nt' else ":"
    paths_to_add = [diarize_dir_abs, parent_dir]
    
    # Sprawdź czy ctc-forced-aligner istnieje
    ctc_found_dir = None
    if os.path.isdir(ctc_aligner_dir):
        ctc_found_dir = ctc_aligner_dir
        paths_to_add.append(ctc_aligner_dir)
    elif os.path.isdir(ctc_aligner_dir_alt):
        ctc_found_dir = ctc_aligner_dir_alt
        paths_to_add.append(ctc_aligner_dir_alt)
    
    if ctc_found_dir:
        ctc_module_dir = os.path.join(ctc_found_dir, "ctc_forced_aligner")
        if os.path.isdir(ctc_module_dir):
            paths_to_add.append(ctc_module_dir)
    
    new_paths = separator.join(paths_to_add)
    if pythonpath:
        pythonpath = f"{new_paths}{separator}{pythonpath}"
    else:
        pythonpath = new_paths
    
    env["PYTHONPATH"] = pythonpath
    
    cmd = [
        python_executable, "-u", diarize_script,
        "-a", wav_path,
        "--no-stem",
        "--whisper-model", WHISPER_MODEL,
        "--device", device
    ]
    if lang and lang.lower() != "auto":
        cmd += ["--language", lang]
    
    # Ustaw compute_type dla starszych GPU (Pascal i starsze nie wspierają float16)
    # Próbujemy dwa sposoby:
    # 1. Parametr --compute-type (jeśli diarize.py go wspiera)
    # 2. Zmienna środowiskowa (jako fallback)
    if compute_type == "float32" and device.lower() == "cuda":
        # Sposób 1: Parametr --compute-type
        cmd += ["--compute-type", "float32"]
        # Sposób 2: Zmienna środowiskowa (dla faster-whisper)
        env["WHISPER_COMPUTE_TYPE"] = "float32"
        print(f"[DIARIZE] 🔧 Ustawiam compute_type=float32 dla kompatybilności z GPU (GTX 1050 Ti)")

    print(f"[DIARIZE] 🚀 Uruchamiam: {' '.join(cmd)}")
    print(f"[DIARIZE] 📁 Katalog roboczy: {diarize_dir_abs}")
    print(f"[DIARIZE] 🔧 Device: {device}, Compute Type: {compute_type}")
    print(f"[DIARIZE] ⏳ To może zająć 10-60 sekund (zależnie od {device})...")
    if device.lower() == "cuda":
        print(f"[DIARIZE] 💡 Sprawdź nvidia-smi w osobnym oknie, aby zobaczyć użycie GPU")

    try:
        import time
        diarize_start = time.time()
        result = subprocess.run(
            cmd, 
            cwd=diarize_dir_abs, 
            env=env,
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=600
        )
        diarize_elapsed = time.time() - diarize_start
        print(f"[DIARIZE] ✅ Proces zakończony w {diarize_elapsed:.1f}s")
        
        if result.stdout:
            stdout_lines = result.stdout.strip().split('\n')
            if len(stdout_lines) > 0:
                print(f"[DIARIZE] 📄 Stdout ({len(stdout_lines)} linii):")
                for line in stdout_lines[-10:]:
                    if line.strip():
                        print(f"[DIARIZE]    {line}")
        
        if result.stderr:
            stderr_lines = result.stderr.strip().split('\n')
            if len(stderr_lines) > 0:
                print(f"[DIARIZE] ⚠️ Stderr ({len(stderr_lines)} linii):")
                for line in stderr_lines[-10:]:
                    if line.strip():
                        print(f"[DIARIZE]    {line}")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else (e.stdout if e.stdout else 'Brak szczegółów')
        print("=" * 80)
        print("[DIARIZE] ❌ BŁĄD - Pełny komunikat błędu:")
        print("=" * 80)
        print(error_msg)
        print("=" * 80)
        
        # Sprawdź czy to błąd brakującego modułu
        missing_module = "ModuleNotFoundError" in error_msg or "No module named" in error_msg
        
        if missing_module:
            # Sprawdź który moduł brakuje
            if "nemo" in error_msg.lower():
                print("\n" + "=" * 80)
                print("[DIARIZE] ⚠️ BRAKUJE MODUŁU: nemo (NVIDIA NeMo Toolkit)")
                print("=" * 80)
                print("❌ Problem: whisper-diarization wymaga NVIDIA NeMo Toolkit dla diarization")
                print()
                print("💡 Rozwiązanie - Zainstaluj nemo:")
                print("   1. Aktywuj środowisko Python (venv_test):")
                print("      venv_test\\Scripts\\activate")
                print()
                print("   2. Zainstaluj nemo:")
                print("      pip install nemo-toolkit[all]")
                print()
                print("   LUB tylko podstawowe (jeśli powyższe nie działa):")
                print("      pip install nemo-toolkit")
                print()
                print("   3. Jeśli masz problemy z instalacją, sprawdź:")
                print("      https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/getting_started/installation.html")
                print()
                print("⚠️  UWAGA: nemo-toolkit może być duży (~1-2GB) i wymagać dużo zależności")
                print("=" * 80)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return []
            elif "faster_whisper" in error_msg.lower():
                print("\n[DIARIZE] ⚠️ Brakuje faster-whisper - zainstaluj: pip install faster-whisper")
            elif "pyannote" in error_msg.lower():
                print("\n[DIARIZE] ⚠️ Brakuje pyannote.audio - zainstaluj: pip install pyannote.audio")
            else:
                # Wyciągnij nazwę modułu z błędu
                import re
                match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_msg)
                if match:
                    module_name = match.group(1)
                    print(f"\n[DIARIZE] ⚠️ Brakuje modułu: {module_name}")
                    print(f"[DIARIZE] 💡 Zainstaluj: pip install {module_name}")
        
        # Sprawdź czy to błąd "unrecognized arguments" (diarize.py nie wspiera --compute-type)
        unrecognized_arg = "unrecognized arguments" in error_msg.lower() and "--compute-type" in error_msg.lower()
        
        if unrecognized_arg:
            print("\n" + "=" * 80)
            print("[DIARIZE] ⚠️ diarize.py NIE WSPIERA parametru --compute-type!")
            print("=" * 80)
            print("❌ Problem: diarize.py nie akceptuje --compute-type z linii poleceń")
            print("💡 Rozwiązanie: Musisz zmodyfikować diarize.py aby używał compute_type='float32'")
            print()
            print("📝 Instrukcja:")
            print("   1. Otwórz plik: C:\\Users\\MSI\\PycharmProjects\\whisper-diarization\\diarize.py")
            print("   2. Znajdź linię z: faster_whisper.WhisperModel(...)")
            print("   3. Dodaj parametr: compute_type='float32'")
            print("   4. Przykład:")
            print("      whisper_model = faster_whisper.WhisperModel(")
            print("          model_size_or_path=args.whisper_model,")
            print("          device=args.device,")
            print("          compute_type='float32',  # <-- DODAJ TO dla GTX 1050 Ti")
            print("      )")
            print()
            print("💡 Alternatywnie, możesz dodać argument do parsera w diarize.py:")
            print("      parser.add_argument('--compute-type', default='float16', ...)")
            print("=" * 80)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return []
        
        # Sprawdź czy to błąd float16 (starsze GPU nie wspierają)
        float16_error = "float16 compute type" in error_msg.lower() and "do not support" in error_msg.lower()
        
        if float16_error and device.lower() == "cuda":
            print("\n" + "=" * 80)
            print("[DIARIZE] ⚠️ BŁĄD FLOAT16 WYKRYTY!")
            print("=" * 80)
            print("❌ Problem: GPU nie wspiera efektywnego float16 (GTX 1050 Ti to Pascal)")
            print("💡 Rozwiązanie: Musisz zmodyfikować diarize.py aby używał compute_type='float32'")
            print()
            print("📝 Instrukcja:")
            print("   1. Otwórz plik: C:\\Users\\MSI\\PycharmProjects\\whisper-diarization\\diarize.py")
            print("   2. Znajdź linię z: faster_whisper.WhisperModel(...)")
            print("   3. Dodaj parametr: compute_type='float32'")
            print("   4. Przykład:")
            print("      whisper_model = faster_whisper.WhisperModel(")
            print("          model_size_or_path=args.whisper_model,")
            print("          device=args.device,")
            print("          compute_type='float32',  # <-- DODAJ TO")
            print("      )")
            print("=" * 80)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return []
        
        # Sprawdź czy to błąd "out of memory" (GPU ma za mało VRAM)
        out_of_memory = "out of memory" in error_msg.lower() or "cuda oom" in error_msg.lower()
        
        if out_of_memory and device.lower() == "cuda":
            print("\n" + "=" * 80)
            print("[DIARIZE] ⚠️ BŁĄD: CUDA OUT OF MEMORY!")
            print("=" * 80)
            print(f"❌ Problem: Model '{WHISPER_MODEL}' jest za duży dla GTX 1050 Ti (4GB VRAM)")
            print()
            print("💡 Rozwiązania (w kolejności zalecanej):")
            print("   1. ✅ Użyj mniejszego modelu Whisper (NAJLEPSZE dla GTX 1050 Ti):")
            print("      - Otwórz plik .env.test")
            print("      - Zmień: WHISPER_MODEL=medium")
            print("      - Lub: WHISPER_MODEL=small (jeszcze mniejszy)")
            print("      - Zrestartuj aplikację")
            print()
            print("   2. 🔄 LUB użyj CPU (wolniejsze, ale działa z large-v3):")
            print("      - W pliku .env.test ustaw: DEVICE=cpu")
            print("      - Zrestartuj aplikację")
            print()
            print("   3. 🔧 LUB zmniejsz batch_size (mniej efektywne):")
            print("      - Dodaj do wywołania: --batch-size 4")
            print("      - Wymaga modyfikacji kodu aplikacji")
            print()
            print("📊 Porównanie modeli:")
            print("   - base: ~74MB, najszybszy, najgorsza jakość")
            print("   - small: ~244MB, szybki, dobra jakość")
            print("   - medium: ~769MB, średnia prędkość, bardzo dobra jakość")
            print("   - large-v3: ~1550MB, wolny, najlepsza jakość (ZA DUŻY dla 4GB VRAM)")
            print("=" * 80)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return []
        
        # Sprawdź czy to błąd CUDA
        cuda_errors = [
            "CUDA driver version is insufficient",
            "CUDA error",
            "CUDA runtime",
        ]
        is_cuda_error = any(err.lower() in error_msg.lower() for err in cuda_errors)
        
        if is_cuda_error and device.lower() == "cuda":
            CUDA_FAILED = True
            
            print("\n" + "=" * 80)
            print("[DIARIZE] ⚠️ BŁĄD CUDA WYKRYTY!")
            print("=" * 80)
            if "driver version is insufficient" in error_msg.lower():
                print("❌ Problem: Sterownik CUDA jest za stary dla CUDA runtime!")
                print("💡 Rozwiązania:")
                print("   1. Zaktualizuj sterowniki NVIDIA GPU:")
                print("      - Pobierz najnowsze sterowniki z: https://www.nvidia.com/Download/index.aspx")
                print("      - Lub użyj GeForce Experience do automatycznej aktualizacji")
                print("   2. Tymczasowo użyj CPU: ustaw DEVICE=cpu w .env.test")
                print("   3. LUB zainstaluj starszą wersję PyTorch z CUDA (zgodną z twoim sterownikiem)")
            else:
                print("💡 Rozwiązania:")
                print("   1. Zaktualizuj sterowniki GPU (NVIDIA)")
                print("   2. Ustaw DEVICE=cpu w .env.test")
            print("=" * 80)
            print("[DIARIZE] 🔄 Próbuję ponownie z CPU...")
            print("[DIARIZE] ⚠️ UWAGA: Wszystkie kolejne chunki będą przetwarzane na CPU (będzie wolniej)")
            print("=" * 80)
            
            # Retry z CPU
            cmd_cpu = cmd.copy()
            try:
                device_idx = cmd_cpu.index("--device")
                if device_idx + 1 < len(cmd_cpu):
                    cmd_cpu[device_idx + 1] = "cpu"
            except ValueError:
                cmd_cpu.extend(["--device", "cpu"])
            
            try:
                result = subprocess.run(
                    cmd_cpu,
                    cwd=diarize_dir_abs,
                    env=env,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600
                )
                diarize_elapsed = time.time() - diarize_start
                print(f"[DIARIZE] ✅ Proces zakończony w {diarize_elapsed:.1f}s (z CPU)")
            except Exception as retry_error:
                print(f"[DIARIZE] ❌ Retry z CPU też nie powiódł się: {retry_error}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return []
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return []
    except subprocess.TimeoutExpired:
        print("[DIARIZE] ❌ Przekroczono limit czasu (600s = 10 minut)")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return []
    except (OSError, NotADirectoryError) as e:
        print(f"[DIARIZE] ❌ Błąd systemowy: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return []

    candidates = glob.glob(os.path.join(tmp_dir, "*.srt"))
    if not candidates:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        candidates = glob.glob(os.path.join(diarize_dir, f"{base}*.srt"))
    if not candidates:
        print("[DIARIZE] ⚠️ Nie znaleziono pliku .srt po diarization!")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return []
    srt_path = max(candidates, key=os.path.getmtime)
    print(f"[DIARIZE] 📄 Znaleziono plik SRT: {srt_path}")

    entries = parse_srt(srt_path)
    segs = [Segment(start=s, end=e, text=t, speaker=spk) for (s, e, spk, t) in entries if t]
    print(f"[DIARIZE] 📝 Sparsowano {len(entries)} wpisów, {len(segs)} segmentów z tekstem")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return segs

# ==================== WORKER ====================
def worker_loop():
    """Worker w tle - przetwarza chunki audio"""
    while True:
        try:
            wav_bytes, chunk_offset = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        print(f"[Worker] 🎯 OTRZYMANO CHUNK! Offset: {chunk_offset:.2f}s, rozmiar: {len(wav_bytes)} bajtów")
        print("[Worker] Rozpoczynam diarization...")
        
        # Aktualizuj status - rozpoczęcie przetwarzania
        with _ui_data_lock:
            _ui_data["current_chunk"] = _ui_data.get("chunks_processed", 0) + 1
            _ui_data["total_chunks"] = _ui_data.get("chunks_created", 0)
            _ui_data["processing_status"] = f"🔄 Przetwarzanie chunka {_ui_data['current_chunk']}/{_ui_data['total_chunks']}..."
            _ui_data["processing_progress"] = 0.1
        
        try:
            import time
            start_time = time.time()
            
            # Wątek do symulacji postępu podczas diarization
            progress_stop_event = threading.Event()
            def simulate_progress():
                progress = 0.3
                while not progress_stop_event.is_set() and progress < 0.7:
                    time.sleep(1.0)
                    if not progress_stop_event.is_set():
                        progress += 0.02
                        if progress > 0.7:
                            progress = 0.7
                        with _ui_data_lock:
                            if _ui_data.get("processing_progress", 0) < 0.7:
                                _ui_data["processing_progress"] = progress
            
            progress_thread = threading.Thread(target=simulate_progress, daemon=True)
            progress_thread.start()
            
            try:
                # Diarization przez Deepgram API lub whisper-diarization
                if USE_DEEPGRAM:
                    segs_local = run_deepgram_diarization_on_chunk(
                        wav_bytes=wav_bytes,
                        lang=FORCED_LANG
                    )
                else:
                    segs_local = run_whisper_diarization_on_chunk(
                        wav_bytes=wav_bytes,
                        lang=FORCED_LANG,
                        device=DEVICE
                    )
            finally:
                progress_stop_event.set()
                progress_thread.join(timeout=0.1)
            
            elapsed_time = time.time() - start_time
            print(f"[Worker] ✅ Diarization zakończone! Zwrócono {len(segs_local)} segmentów (czas: {elapsed_time:.1f}s)")
            
            if len(segs_local) == 0:
                print("[Worker] ⚠️ Brak segmentów w wyniku diarization!")
                with _ui_data_lock:
                    _ui_data["processing_progress"] = 0.0
                continue
            
            # Aktualizuj progress - diarization zakończone, parsowanie
            with _ui_data_lock:
                _ui_data["processing_progress"] = 0.7
            
            # Dodaj segmenty do transkrypcji (z filtrowaniem duplikatów z overlap)
            last_segment_end = 0.0
            if len(conversation_state.transcript) > 0:
                last_segment_end = conversation_state.transcript[-1].end
            
            # Filtruj segmenty, które są w obszarze overlap (pierwsze 2 sekundy chunka)
            overlap_start = chunk_offset
            overlap_end = chunk_offset + CHUNK_OVERLAP_SECONDS
            
            for s in segs_local:
                segment_start_global = s.start + chunk_offset
                segment_end_global = s.end + chunk_offset
                
                # Pomiń segmenty, które są w obszarze overlap (już były w poprzednim chunku)
                if segment_start_global < overlap_end and segment_start_global >= overlap_start:
                    # Segment zaczyna się w obszarze overlap - sprawdź czy nie jest duplikatem
                    if segment_start_global < last_segment_end:
                        # Ten segment już był w poprzednim chunku - pomiń
                        continue
                
                conversation_state.transcript.append(
                    Segment(
                        start=segment_start_global,
                        end=segment_end_global,
                        text=s.text,
                        speaker=s.speaker
                    )
                )
            
            # Aktualizuj progress - zakończone
            with _ui_data_lock:
                _ui_data["processing_progress"] = 0.9
            
            # Zaktualizuj UI - transkrypcja
            lines = []
            for seg in conversation_state.transcript[-MAX_RENDER_LINES:]:
                speaker_label = seg.speaker or "?"
                lines.append(f"[{seg.start:6.1f}–{seg.end:6.1f}] {speaker_label}: {seg.text}")
            
            with _ui_data_lock:
                _ui_data["transcript_text"] = "\n".join(lines)
                _ui_data["chunks_processed"] += 1
                _ui_data["processing_progress"] = 1.0
                _ui_data["processing_status"] = f"✅ Chunek {_ui_data['chunks_processed']}/{_ui_data.get('chunks_created', 0)} przetworzony"
                print(f"[Worker] 📝 Zaktualizowano transkrypcję: {len(lines)} linii, łącznie {len(conversation_state.transcript)} segmentów")
            
            # Zaktualizuj UI - mówcy
            speakers = set()
            for seg in conversation_state.transcript:
                if seg.speaker:
                    speakers.add(seg.speaker)
            
            speakers_display = []
            if speakers:
                speakers_display.append("**Rozpoznani mówcy:**\n")
                for speaker in sorted(speakers):
                    count = sum(1 for seg in conversation_state.transcript if seg.speaker == speaker)
                    speakers_display.append(f"🎤 **{speaker}:** {count} segmentów")
            else:
                speakers_display.append("_Oczekiwanie na rozpoznanie mówców..._")
            
            with _ui_data_lock:
                _ui_data["speakers_text"] = "\n".join(speakers_display)
            
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
        
        chunk_samples = int(CHUNK_SECONDS * TARGET_SR)
        overlap_samples = int(CHUNK_OVERLAP_SECONDS * TARGET_SR)
        
        # Tworzymy chunki z overlap - każdy chunk zaczyna się 2 sekundy wcześniej niż poprzedni
        while len(_audio_buffer) >= chunk_samples:
            # Weź chunk (10 sekund)
            chunk = _audio_buffer[:chunk_samples]
            
            # Zostaw overlap w buforze (2 sekundy) dla następnego chunka
            # Usuń tylko (chunk_samples - overlap_samples) z początku
            samples_to_remove = chunk_samples - overlap_samples
            _audio_buffer = _audio_buffer[samples_to_remove:]
            
            wav_bytes = to_wav_bytes(chunk, TARGET_SR)
            global _chunk_counter
            # Offset jest liczony bez overlap (każdy chunk zaczyna się 8 sekund po poprzednim zamiast 10)
            offset = _chunk_counter * (CHUNK_SECONDS - CHUNK_OVERLAP_SECONDS)
            _chunk_counter += 1
            audio_q.put((wav_bytes, offset))
            
            print(f"[Stream] ⏰ CHUNK GOTOWY! {CHUNK_SECONDS}s (z overlap {CHUNK_OVERLAP_SECONDS}s), offset: {offset:.1f}s, bufor: {len(_audio_buffer)/TARGET_SR:.1f}s")
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
        
        transcript, speakers, status, proc_status, proc_progress = get_current_status()
        return "✅ Nagrywanie rozpoczęte! Mów do mikrofonu...", transcript, speakers
    except Exception as e:
        _recording_active = False
        error_msg = f"❌ Błąd rozpoczęcia nagrywania: {e}"
        print(f"[Stream] {error_msg}")
        return error_msg, "", ""

def stop_recording():
    """Zatrzymuje nagrywanie"""
    global _recording_active, _audio_stream
    
    if not _recording_active:
        transcript, speakers, status, proc_status, proc_progress = get_current_status()
        return "⚠️ Nagrywanie nie jest aktywne!", transcript, speakers, status, proc_status, proc_progress
    
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
        
        transcript, speakers, status, proc_status, proc_progress = get_current_status()
        return "⏹️ Nagrywanie zatrzymane", transcript, speakers, status, proc_status, proc_progress
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
    conversation_state.started_at = time.time()
    _chunk_counter = 0
    
    with _audio_buffer_lock:
        _audio_buffer = np.array([], dtype=np.float32)
    
    with _ui_data_lock:
        _ui_data["transcript_text"] = ""
        _ui_data["speakers_text"] = "_Oczekiwanie na transkrypcję..._"
        _ui_data["chunks_created"] = 0
        _ui_data["chunks_processed"] = 0
        _ui_data["processing_status"] = "⏸️ Oczekiwanie na chunki..."
        _ui_data["processing_progress"] = 0.0
    
    print("[Reset] 🔄 Stan zresetowany")

def get_current_status():
    """Zwraca aktualny status dla auto-refresh"""
    with _ui_data_lock:
        transcript = _ui_data.get("transcript_text", "")
        speakers = _ui_data.get("speakers_text", "_Oczekiwanie na transkrypcję..._")
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
    return transcript, speakers, status, processing_status, progress_percent

# Tworzenie UI
with gr.Blocks(title="Live Transcription + Diarization") as demo:
    gr.Markdown("# 🎙️ Transkrypcja na żywo + Diarization")
    if USE_DEEPGRAM:
        gr.Markdown("**Streaming w czasie rzeczywistym:** Kliknij 'Start nagrywania' i mów do mikrofonu. Co 10 sekund audio jest automatycznie przetwarzane przez Deepgram API w celu transkrypcji i rozpoznania mówców.")
    else:
        gr.Markdown("**Streaming w czasie rzeczywistym:** Kliknij 'Start nagrywania' i mów do mikrofonu. Co 10 sekund audio jest automatycznie przetwarzane przez whisper-diarization (CUDA) w celu rozpoznania mówców.")
    
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
            speakers_output = gr.Markdown(
                label="👥 Rozpoznani mówcy",
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
        outputs=[transcript_output, speakers_output, status_output, processing_status_text, processing_progress_bar]
    )
    
    # Automatyczne odświeżanie - użyjmy prostego podejścia z JavaScript
    demo.load(
        fn=None,
        js="""
        () => {
            setInterval(() => {
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
        outputs=[transcript_output, speakers_output, status_output, processing_status_text, processing_progress_bar]
    )
    
    # Kontrola nagrywania
    start_btn.click(
        fn=start_recording,
        inputs=None,
        outputs=[recording_status, transcript_output, speakers_output]
    )
    
    stop_btn.click(
        fn=stop_recording,
        inputs=None,
        outputs=[recording_status, transcript_output, speakers_output, status_output, processing_status_text, processing_progress_bar]
    )
    
    # Reset bufora
    reset_btn.click(
        fn=reset_audio_buffer,
        inputs=None,
        outputs=None
    )
    
    gr.Markdown("---")
    if USE_DEEPGRAM:
        gr.Markdown("**Silnik:** Deepgram API (Cloud) | **Diarization:** Włączone")
    else:
        gr.Markdown("**Silnik:** whisper-diarization (CUDA) | **Model:** Whisper Large V3")

if __name__ == "__main__":
    print("[Main] 🚀 Inicjalizacja aplikacji...")
    
    # Wstępne wczytanie modelu przed uruchomieniem Gradio (tylko jeśli nie używamy Deepgram)
    if not USE_DEEPGRAM and DEVICE_CONFIG.lower() == "cuda":
        compute_type = get_gpu_compute_type(DEVICE_CONFIG)
        print(f"[Main] 📋 Konfiguracja: Device={DEVICE_CONFIG}, ComputeType={compute_type}, Model={WHISPER_MODEL}")
        print(f"[Main] 💡 Wczytuję model przed startem aplikacji...")
        print(f"[Main] 💡 Sprawdź nvidia-smi w osobnym oknie, aby zobaczyć użycie GPU podczas wczytywania")
        print()
        
        # Wczytaj model w tle (w osobnym wątku, aby nie blokować)
        def preload_in_background():
            preload_whisper_model(DEVICE_CONFIG, compute_type, WHISPER_MODEL)
        
        preload_thread = threading.Thread(target=preload_in_background, daemon=True)
        preload_thread.start()
        print("[Main] 💡 Wczytywanie modelu rozpoczęte w tle...")
        print("[Main] 💡 Aplikacja uruchomi się, ale model będzie wczytywany równolegle")
        print()
    else:
        print(f"[Main] 📋 Konfiguracja: Device={DEVICE_CONFIG}, Model={WHISPER_MODEL}")
        print(f"[Main] ⚠️  Używam CPU - wczytywanie modelu przy pierwszym użyciu")
        print()
    
    print("[Main] 🚀 Uruchamiam Gradio...")
    demo.queue()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)

