# Testy Diarization i Transkrypcji na Żywo

Ten folder zawiera testowe implementacje systemów transkrypcji i rozpoznawania mówców w czasie rzeczywistym.

## 📁 Pliki

### 1. `test_chunking_gradio_diarization.py`
**Transkrypcja + Diarization (rozpoznawanie mówców)**

- **Funkcjonalność:**
  - Streaming audio w czasie rzeczywistym z mikrofonu
  - Automatyczne chunkowanie nagrania (10 sekund z 2 sekundami overlap)
  - Transkrypcja i diarization przez Deepgram API LUB lokalne whisper-diarization
  - Wyświetlanie transkrypcji z przypisanymi mówcami (Speaker 0, Speaker 1, ...)
  - Automatyczne odświeżanie UI co 2 sekundy
  - Progress bar dla przetwarzania chunków

- **Mechanizm chunkowania:**
  - Każdy chunk: 10 sekund audio
  - Overlap: 2 sekundy między chunkami (aby nie ucinać zdań)
  - Przykład: Chunk 1: 0-10s, Chunk 2: 8-18s, Chunk 3: 16-26s
  - Automatyczne filtrowanie duplikatów z obszaru overlap

- **Diarization:**
  - **Deepgram API** (domyślnie, `USE_DEEPGRAM=true`):
    - Model: nova-2
    - Język: polski (pl)
    - Parametry: `diarize=true`, `utterances=true`, `smart_format=true`
    - Automatyczne wykrywanie mówców w odpowiedzi API
  - **whisper-diarization** (lokalne, `USE_DEEPGRAM=false`):
    - Wymaga CUDA GPU (lub CPU jako fallback)
    - Wykrywa architekturę GPU i dostosowuje `compute_type` (float16/float32)
    - Automatyczny fallback na CPU przy błędach CUDA
    - Wymaga zewnętrznych repozytoriów: `whisper-diarization` i `ctc-forced-aligner`

- **Port:** 7861

### 2. `test_chunking_gradio.py`
**Transkrypcja + Rozpoznawanie Ról przez LLM**

- **Funkcjonalność:**
  - Streaming audio w czasie rzeczywistym z mikrofonu
  - Automatyczne chunkowanie nagrania (10 sekund, bez overlap)
  - Transkrypcja przez Groq Whisper API (model: whisper-large-v3)
  - Rozpoznawanie ról (lekarz/pacjent) przez LLM (Gemini Flash przez OpenRouter)
  - Wyświetlanie transkrypcji z przypisanymi rolami
  - Automatyczne odświeżanie UI co 2 sekundy
  - Progress bar dla przetwarzania chunków

- **Mechanizm chunkowania:**
  - Każdy chunk: 10 sekund audio
  - Brak overlap (prostsze, ale może ucinać zdania)
  - Przykład: Chunk 1: 0-10s, Chunk 2: 10-20s, Chunk 3: 20-30s

- **Rozpoznawanie ról:**
  - LLM analizuje transkrypcję i rozpoznaje role na podstawie treści
  - Wykrywa lekarza i pacjenta w rozmowie medycznej
  - Zwraca pewność rozpoznania (wysoka/średnia/niska)

- **Port:** 7860

## 🚀 Instalacja

1. **Zainstaluj zależności:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Skonfiguruj zmienne środowiskowe:**
   Utwórz plik `.env.test` w głównym folderze projektu z:
   ```env
   # Dla test_chunking_gradio.py
   GROQ_API_KEY=your_groq_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   LANG=pl
   
   # Dla test_chunking_gradio_diarization.py (Deepgram)
   DEEPGRAM_API_KEY=your_deepgram_api_key
   USE_DEEPGRAM=true
   
   # Dla test_chunking_gradio_diarization.py (whisper-diarization lokalne)
   USE_DEEPGRAM=false
   WHISPER_DIARIZATION_DIR=./whisper-diarization
   DEVICE=cuda
   WHISPER_MODEL=small
   ```

3. **Uruchom aplikację:**
   ```bash
   # Diarization
   python test_chunking_gradio_diarization.py
   
   # Rozpoznawanie ról
   python test_chunking_gradio.py
   ```

## 🔧 Konfiguracja

### `test_chunking_gradio_diarization.py`
- `CHUNK_SECONDS=10` - długość chunka (sekundy)
- `CHUNK_OVERLAP_SECONDS=2.0` - overlap między chunkami (sekundy)
- `USE_DEEPGRAM=true/false` - wybór silnika diarization
- `DEVICE=cuda/cpu` - urządzenie dla whisper-diarization
- `WHISPER_MODEL=small/medium/large-v3` - model Whisper

### `test_chunking_gradio.py`
- `CHUNK_SECONDS=10` - długość chunka (sekundy)
- `GEMINI_FLASH_MODEL` - model LLM (domyślnie: google/gemini-2.5-flash-lite)

## 📝 Mechanizm Chunkowania

### Wersja z overlap (`test_chunking_gradio_diarization.py`):
```
Chunk 1: [0s -------- 10s]
Chunk 2:        [8s -------- 18s]  (2s overlap)
Chunk 3:               [16s -------- 26s]  (2s overlap)
```

**Zalety:**
- Nie ucina zdań w połowie
- Lepsze rozpoznawanie mówców na granicach chunków
- Płynniejsza transkrypcja

**Wady:**
- Większe zużycie zasobów (przetwarzanie overlap)
- Wymaga filtrowania duplikatów

### Wersja bez overlap (`test_chunking_gradio.py`):
```
Chunk 1: [0s -------- 10s]
Chunk 2:              [10s -------- 20s]
Chunk 3:                         [20s -------- 30s]
```

**Zalety:**
- Prostsze w implementacji
- Mniejsze zużycie zasobów

**Wady:**
- Może ucinać zdania w połowie
- Gorsze rozpoznawanie na granicach chunków

## 🎯 Diarization

### Deepgram API (zalecane)
- ✅ Szybkie (1-2 sekundy na chunk)
- ✅ Nie wymaga GPU
- ✅ Dobre rozpoznawanie mówców
- ❌ Wymaga klucza API (płatne)

### whisper-diarization (lokalne)
- ✅ Darmowe (lokalne przetwarzanie)
- ✅ Dobre rozpoznawanie mówców
- ❌ Wymaga CUDA GPU (lub wolne na CPU)
- ❌ Wymaga zewnętrznych repozytoriów
- ❌ Wolniejsze (10-60 sekund na chunk)

## 📊 Status

**To jest wersja testowa** - nie trafi do głównego projektu. Została utworzona w celu:
- Testowania mechanizmów chunkowania audio
- Testowania różnych silników diarization
- Eksperymentowania z rozpoznawaniem mówców i ról
- Oceny wydajności i jakości transkrypcji

## 🔍 Co zostało zaimplementowane

1. ✅ Streaming audio w czasie rzeczywistym (sounddevice)
2. ✅ Automatyczne chunkowanie z overlap (2 sekundy)
3. ✅ Integracja z Deepgram API dla diarization
4. ✅ Integracja z whisper-diarization (lokalne)
5. ✅ Automatyczne wykrywanie architektury GPU
6. ✅ Fallback na CPU przy błędach CUDA
7. ✅ Progress bar i status przetwarzania
8. ✅ Automatyczne odświeżanie UI
9. ✅ Filtrowanie duplikatów z overlap
10. ✅ Rozpoznawanie ról przez LLM (test_chunking_gradio.py)

## ⚠️ Uwagi

- Pliki wymagają pliku `.env.test` w głównym folderze projektu (nie w `diarization_test`)
- `whisper-diarization` wymaga dodatkowych zależności i zewnętrznych repozytoriów
- Deepgram API wymaga klucza API (może być płatne)
- Testy są przeznaczone do eksperymentowania, nie do produkcji

