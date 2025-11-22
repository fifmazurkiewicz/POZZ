"""
Moduł do zarządzania ciągłym nagrywaniem wywiadów z automatycznym zapisem chunków co 3 minuty.
"""
import os
import tempfile
import time
from datetime import datetime
from typing import List, Optional, Tuple
import threading
import queue

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcStreamerContext
    STREAMLIT_WEBRTC_AVAILABLE = True
except ImportError:
    STREAMLIT_WEBRTC_AVAILABLE = False
    WebRtcStreamerContext = None


class InterviewRecorder:
    """Zarządza ciągłym nagrywaniem wywiadu z automatycznym zapisem chunków co 3 minuty."""
    
    CHUNK_DURATION_SECONDS = 180  # 3 minuty
    
    def __init__(self):
        self.recording_active = False
        self.start_time: Optional[float] = None
        self.chunk_number = 0
        self.chunk_files: List[str] = []
        self.audio_queue: queue.Queue = queue.Queue()
        self.temp_dir = tempfile.mkdtemp(prefix="interview_audio_")
        self.last_chunk_save_time: Optional[float] = None
        self.recording_thread: Optional[threading.Thread] = None
        
    def start_recording(self) -> bool:
        """Rozpoczyna ciągłe nagrywanie."""
        if self.recording_active:
            return False
        
        self.recording_active = True
        self.start_time = time.time()
        self.chunk_number = 0
        self.chunk_files = []
        self.last_chunk_save_time = time.time()
        self.audio_queue = queue.Queue()
        
        return True
    
    def stop_recording(self) -> Tuple[Optional[str], List[str]]:
        """Zatrzymuje nagrywanie i zwraca ścieżkę do ostatniego fragmentu oraz listę wszystkich chunków."""
        if not self.recording_active:
            return None, []
        
        self.recording_active = False
        
        # Zapis ostatniego fragmentu (jeśli minęło > 10 sekund od ostatniego chunka)
        last_fragment = None
        if self.start_time:
            elapsed = time.time() - self.last_chunk_save_time if self.last_chunk_save_time else 0
            if elapsed > 10:  # Jeśli minęło więcej niż 10 sekund, zapisz ostatni fragment
                last_fragment = self._save_current_chunk()
        
        all_chunks = self.chunk_files.copy()
        if last_fragment:
            all_chunks.append(last_fragment)
        
        return last_fragment, all_chunks
    
    def is_recording(self) -> bool:
        """Sprawdza czy nagrywanie jest aktywne."""
        return self.recording_active
    
    def get_total_duration(self) -> float:
        """Zwraca całkowity czas nagrywania w sekundach."""
        if not self.recording_active or not self.start_time:
            return 0.0
        return time.time() - self.start_time
    
    def get_chunk_number(self) -> int:
        """Zwraca numer ostatniego zapisanego chunka."""
        return self.chunk_number
    
    def get_saved_chunks_count(self) -> int:
        """Zwraca liczbę zapisanych chunków."""
        return len(self.chunk_files)
    
    def can_continue(self) -> bool:
        """Sprawdza czy można kontynuować nagrywanie (max 5 chunków = 15 min)."""
        return self.get_saved_chunks_count() < 5
    
    def should_save_chunk(self) -> bool:
        """Sprawdza czy powinien zostać zapisany chunka (co 3 minuty)."""
        if not self.recording_active or not self.last_chunk_save_time:
            return False
        
        elapsed = time.time() - self.last_chunk_save_time
        return elapsed >= self.CHUNK_DURATION_SECONDS
    
    def save_chunk(self) -> Optional[str]:
        """Zapisuje chunka 3-minutowego (wywoływane co 3 minuty w tle)."""
        if not self.recording_active:
            return None
        
        chunk_file = self._save_current_chunk()
        if chunk_file:
            self.chunk_number += 1
            self.chunk_files.append(chunk_file)
            self.last_chunk_save_time = time.time()
        
        return chunk_file
    
    def _save_current_chunk(self) -> Optional[str]:
        """Wewnętrzna metoda do zapisu chunka.
        
        UWAGA: To jest placeholder. Rzeczywisty zapis audio wymaga integracji z streamlit-webrtc.
        W obecnej implementacji zwracamy None, aby uniknąć błędów z pustymi plikami.
        """
        # TODO: Implementacja zapisu audio z streamlit-webrtc
        # Na razie zwracamy None, aby uniknąć błędów z pustymi plikami
        # Rzeczywisty zapis audio będzie wymagał:
        # 1. Integracji z streamlit-webrtc do przechwytywania audio
        # 2. Buforowania audio w pamięci podczas nagrywania
        # 3. Zapisania bufora do pliku co 3 minuty
        
        # Placeholder - zwracamy None zamiast pustego pliku
        return None
    
    def get_all_chunks(self) -> List[str]:
        """Zwraca listę ścieżek do wszystkich zapisanych chunków."""
        return self.chunk_files.copy()
    
    def cleanup(self):
        """Czyści tymczasowe pliki."""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def format_duration(self, seconds: float) -> str:
        """Formatuje czas w sekundach do formatu MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


# Globalna instancja rejestratora (dla session state)
_recorder_instance: Optional[InterviewRecorder] = None


def get_recorder() -> InterviewRecorder:
    """Zwraca globalną instancję rejestratora."""
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = InterviewRecorder()
    return _recorder_instance

