import streamlit as st
import tempfile
import os
import hashlib
import random
import re
import json
import time
from typing import List, Optional, Dict, Any

from modules import llm_service, prompt_manager
from modules import db
from modules import audio_processor


# --- Page config ---
st.set_page_config(
    page_title="Symulator Pacjenta POZ",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Symulator Pacjenta POZ - Aplikacja treningowa dla lekarzy"
    }
)

# Add mobile-friendly viewport meta tag and microphone permission request
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
    /* Mobile-friendly styles */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        .stButton > button {
            width: 100%;
        }
    }
    .mic-permission-banner {
        background-color: #ff9800;
        color: white;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
        text-align: center;
    }
    .mic-permission-banner button {
        background-color: white;
        color: #ff9800;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 8px;
        font-weight: bold;
    }
</style>
<script>
// Request microphone permissions on page load
(function() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Check current permission status
        navigator.permissions.query({ name: 'microphone' }).then(function(result) {
            if (result.state === 'prompt') {
                // Permission not yet requested, request it
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(function(stream) {
                        // Permission granted, stop the stream
                        stream.getTracks().forEach(track => track.stop());
                        console.log('Microphone permission granted');
                    })
                    .catch(function(err) {
                        console.log('Microphone permission denied or error:', err);
                    });
            } else if (result.state === 'denied') {
                // Permission was denied, show a message
                console.log('Microphone permission was denied');
                // You can show a banner here if needed
            }
        }).catch(function(err) {
            // Permissions API not supported, try direct request
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function(stream) {
                    stream.getTracks().forEach(track => track.stop());
                    console.log('Microphone permission granted');
                })
                .catch(function(err) {
                    console.log('Microphone permission error:', err);
                });
        });
    }
})();
</script>
""", unsafe_allow_html=True)

st.title("🩺 Symulator Pacjenta POZ")

# --- Session state initialization ---
if "patient_scenario" not in st.session_state:
    st.session_state.patient_scenario = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # OpenAI-compatible messages
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "doctor_asks"  # doctor_asks | patient_asks | meta_ask
if "patient_id" not in st.session_state:
    st.session_state.patient_id = None
# Basic patient info card state
if "patient_basic_info" not in st.session_state:
    st.session_state.patient_basic_info = {
        "name": "",
        "age": "",
        "has_history_here": None,  # True/False/None
        "chronic_diseases": "",
        "operations": "",
        "allergies": "",
        "family_history": "",
    }
if "patient_first_time" not in st.session_state:
    st.session_state.patient_first_time = False
# Load treatment plan from DB when patient is loaded (if exists)
if st.session_state.patient_id and st.session_state.correct_treatment_plan is None:
    try:
        treatment_plan = db.get_patient_treatment_plan(st.session_state.patient_id)
        if treatment_plan:
            st.session_state.correct_treatment_plan = treatment_plan
    except Exception:
        pass  # Ignore errors, will generate if needed
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "manual_interview_history" not in st.session_state:
    st.session_state.manual_interview_history = []
if "manual_patient_scenario" not in st.session_state:
    st.session_state.manual_patient_scenario = ""
if "manual_interview_summary" not in st.session_state:
    st.session_state.manual_interview_summary = None
if "manual_interview_recommendations" not in st.session_state:
    st.session_state.manual_interview_recommendations = None
if "last_processed_audio_hash" not in st.session_state:
    st.session_state.last_processed_audio_hash = None
if "last_end_interview_audio_hash" not in st.session_state:
    st.session_state.last_end_interview_audio_hash = None
if "interview_end_mode" not in st.session_state:
    st.session_state.interview_end_mode = None  # None, "waiting_for_response", "evaluated"
if "correct_treatment_plan" not in st.session_state:
    st.session_state.correct_treatment_plan = None
if "user_treatment_response" not in st.session_state:
    st.session_state.user_treatment_response = None
if "diagnosis_evaluation" not in st.session_state:
    st.session_state.diagnosis_evaluation = None
# Interview recording state
if "interview_recording_active" not in st.session_state:
    st.session_state.interview_recording_active = False
if "interview_recording_start_time" not in st.session_state:
    st.session_state.interview_recording_start_time = None
if "interview_chunks" not in st.session_state:
    st.session_state.interview_chunks = []
if "interview_suggestions" not in st.session_state:
    st.session_state.interview_suggestions = []
if "interview_processing_chunks" not in st.session_state:
    st.session_state.interview_processing_chunks = set()
if "interview_summary" not in st.session_state:
    st.session_state.interview_summary = None
if "interview_extracted_info" not in st.session_state:
    st.session_state.interview_extracted_info = None

# --- Initialize DB schema (no-op if exists) ---
try:
    db.init_schema()
except Exception as exc:
    st.warning(f"Database not initialized: {exc}")


# --- Helper functions for patient generation ---
def generate_patient(keywords: Optional[str] = None) -> Optional[str]:
    """Generate a new patient and return patient_id, or None on error."""
    try:
        # Losowe określenie czy pacjent pierwszorazowy (20% szans, 80% stałych z kartą)
        patient_first_time = random.choices([True, False], weights=[20, 80])[0]
        prompt_messages = prompt_manager.generate_patient_scenario_prompt(
            keywords=keywords,
            first_time_missing_basics=patient_first_time,
        )
        scenario = llm_service.get_llm_response(
            prompt_messages,
            model_name="google/gemini-2.5-flash-lite",
        )
        
        # Generate summary via LLM
        summary = ""
        try:
            summary_prompt = prompt_manager.generate_patient_summary_prompt(scenario)
            summary = llm_service.get_llm_response(
                summary_prompt,
                model_name="google/gemini-2.5-flash-lite",
            ).strip()
            summary = summary.strip('"').strip("'").strip()
        except Exception:
            pass  # Summary is optional
        
        # Generate treatment plan via LLM
        treatment_plan = ""
        try:
            plan_prompt = prompt_manager.generate_treatment_plan_prompt(
                patient_scenario=scenario,
                chat_history=[],
            )
            treatment_plan = llm_service.get_llm_response(
                plan_prompt,
                model_name="google/gemini-2.5-flash-lite",
            )
        except Exception:
            pass  # Treatment plan is optional
        
        # Persist patient
        patient_id = db.create_patient(
            scenario=scenario,
            summary=summary,
            treatment_plan=treatment_plan
        )
        
        # Decide whether to reveal the card to the user (20% default);
        # dla generowania na żądanie nie znamy tu jeszcze flagi pierwszorazowego,
        # zostanie ustawiona w load_patient_to_session
        st.session_state.reveal_patient_card = bool(random.choices([True, False], weights=[20, 80])[0])
        
        return patient_id
    except Exception as exc:
        st.error(f"Błąd generowania pacjenta: {exc}")
        return None


def ensure_patient_pool(target_count: int = 20):
    """Ensure there are at least target_count unprocessed patients in the database."""
    try:
        current_count = db.count_unprocessed_patients()
        needed = target_count - current_count
        if needed > 0:
            with st.spinner(f"Generowanie {needed} pacjentów do puli..."):
                for i in range(needed):
                    generate_patient()
                    if (i + 1) % 5 == 0:
                        st.info(f"Wygenerowano {i + 1}/{needed} pacjentów...")
    except Exception as exc:
        st.warning(f"Błąd podczas uzupełniania puli pacjentów: {exc}")


def parse_patient_card_from_scenario(scenario: str) -> Optional[Dict[str, str]]:
    """Parse 'Karta pacjenta' section from scenario text, robust to Markdown and bullets.

    Accepts both plain and markdown/bulleted formats and returns a dict with:
    name, age, has_history_here, chronic_diseases, operations, allergies, family_history.
    """
    # Find the start of the card section (case-insensitive)
    header_match = re.search(r"karta\s+pacjenta\s*:?(?:\s*\(wymagana\))?\s*", scenario, re.IGNORECASE)
    if header_match:
        start_idx = header_match.end()
    else:
        # Fallback: very tolerant search (no regex anchors)
        low = scenario.lower()
        idx = low.find("karta pacjenta")
        if idx == -1:
            return None
        # Move to end of that line if present
        line_end = scenario.find("\n", idx)
        start_idx = (line_end + 1) if line_end != -1 else idx

    # Take a reasonable window after header to parse (to avoid picking later sections)
    after = scenario[start_idx:start_idx + 2000]
    # Truncate if app footer marker present
    for marker in ["\n🩺", "\nSymulator", "\nSłowa kluczowe", "\nSlowa kluczowe"]:
        pos = after.find(marker)
        if pos != -1:
            after = after[:pos]
            break
    lines = after.splitlines()

    # Helper to normalize a line (remove bullets/markdown and trim)
    def normalize_line(line: str) -> str:
        # Remove leading bullets and markdown markers
        line = re.sub(r"^[\s>*\-•\u2022]*", "", line)  # bullets and spaces
        line = line.strip()
        # Remove bold markers if present
        line = line.replace("**", "").replace("*", "")
        return line

    # Map of Polish labels to internal keys
    field_map = {
        "imię i nazwisko": "name",
        "imie i nazwisko": "name",
        "wiek": "age",
        "historia w punkcie": "has_history_here",
        "choroby przewlekłe": "chronic_diseases",
        "choroby przewlekle": "chronic_diseases",
        "operacje": "operations",
        "alergie": "allergies",
        "wywiad rodzinny": "family_history",
    }

    result: Dict[str, Any] = {
        "name": "",
        "age": "",
        "has_history_here": None,
        "chronic_diseases": "",
        "operations": "",
        "allergies": "",
        "family_history": "",
    }

    # Parse until we hit an empty line followed by a non-indented section or end
    for raw_line in lines:
        line = normalize_line(raw_line)
        if not line:
            # likely end of card block; continue to see if next lines are still fields
            continue
        # Split on the first colon
        if ":" not in line:
            # If we hit a non key:value line after we've started collecting, we can stop
            # but being safe, just skip
            continue
        label, value = line.split(":", 1)
        label = label.strip().lower()
        value = value.strip()
        key = field_map.get(label)
        if not key:
            continue
        if key == "has_history_here":
            low = value.lower()
            if "tak" in low:
                result["has_history_here"] = True
            elif "nie" in low:
                result["has_history_here"] = False
        else:
            result[key] = value

    # Validate: at least name or age must be present to accept card
    if (result["name"] or result["age"]) and result["has_history_here"] is not None:
        return result  # type: ignore[return-value]
    return None


def load_patient_to_session(patient_id: str):
    """Load patient data from database into session state."""
    try:
        patient_data = db.get_patient_by_id(patient_id)
        if not patient_data:
            return False
        
        scenario, summary, treatment_plan, _ = patient_data
        st.session_state.patient_id = patient_id
        st.session_state.patient_scenario = scenario
        st.session_state.correct_treatment_plan = treatment_plan
        st.session_state.chat_history = []
        
        # Reset interview end state (hide evaluation section)
        st.session_state.interview_end_mode = None
        st.session_state.user_treatment_response = None
        st.session_state.diagnosis_evaluation = None
        
        # Parse "Karta pacjenta" section directly from scenario (ONLY source)
        card_data = parse_patient_card_from_scenario(scenario)
        
        # Reset basic card info
        st.session_state.patient_basic_info = {
            "name": "",
            "age": "",
            "has_history_here": None,
            "chronic_diseases": "",
            "operations": "",
            "allergies": "",
            "family_history": "",
        }
        
        # Use parsed card data (must be in scenario)
        if card_data:
            st.session_state.patient_basic_info.update({
                "name": card_data.get("name", ""),
                "age": card_data.get("age", ""),
                "has_history_here": card_data.get("has_history_here"),
                "chronic_diseases": card_data.get("chronic_diseases", ""),
                "operations": card_data.get("operations", ""),
                "allergies": card_data.get("allergies", ""),
                "family_history": card_data.get("family_history", ""),
            })
            # Determine if first-time based on has_history_here
            is_first_time = (card_data.get("has_history_here") is False)
            st.session_state.patient_first_time = is_first_time
        else:
            # If card section not found, show warning
            st.warning("⚠️ Sekcja 'Karta pacjenta' nie została znaleziona w scenariuszu. Upewnij się, że LLM generuje tę sekcję.")
            st.session_state.patient_first_time = False
            st.session_state.patient_basic_info["has_history_here"] = None
        
        # Decide whether to reveal the card to the user (20% true)
        st.session_state.reveal_patient_card = bool(random.choices([True, False], weights=[20, 80])[0])
        
        # Create new conversation
        try:
            st.session_state.conversation_id = db.create_conversation(
                patient_id=patient_id, title="Simulation"
            )
        except Exception:
            pass
        
        return True
    except Exception as exc:
        st.error(f"Błąd ładowania pacjenta: {exc}")
        return False

# --- Sidebar: patient scenario and chat history ---
with st.sidebar:
    st.header("Pacjent")
    if st.session_state.patient_scenario:
        with st.expander("Scenariusz pacjenta", expanded=False):
            st.markdown(st.session_state.patient_scenario)
    else:
        st.info("Brak scenariusza pacjenta.")

# --- Tabs ---
tab_sim, tab_interview, tab_browse, tab_admin = st.tabs(["Symulacja", "Wywiad", "Przeglądanie", "Admin"])

with tab_sim:
    
    # Pole na słowa kluczowe (opcjonalnie)
    keywords_input = st.text_input(
        "Słowa kluczowe (opcjonalnie):",
        placeholder="np. kobieta, ból głowy, nadciśnienie",
    )

    # Show pool status
    try:
        pool_count = db.count_unprocessed_patients()
        st.caption(f"Dostępnych pacjentów w puli: {pool_count}")
    except Exception:
        pass
    
    btn_next_col, _ = st.columns([1, 5])
    with btn_next_col:
        if st.button("Następny pacjent", type="secondary"):
            # Opcjonalnie oznacz bieżącego pacjenta jako pominiętego/obsłużonego
            current_pid = st.session_state.get("patient_id")
            if current_pid:
                try:
                    db.mark_patient_skipped(current_pid)
                except Exception as exc:
                    st.warning(f"Nie udało się oznaczyć poprzedniego pacjenta jako pominiętego: {exc}")
            # Jeśli podano słowa kluczowe – generuj nowego pacjenta na ich podstawie
            if keywords_input and keywords_input.strip():
                new_patient_id = generate_patient(keywords_input.strip())
                if new_patient_id and load_patient_to_session(new_patient_id):
                    st.success("Nowy pacjent wygenerowany i załadowany.")
                    st.rerun()
                else:
                    st.error("Nie udało się wygenerować pacjenta na podstawie słów kluczowych.")
            else:
                # Brak słów kluczowych – ładuj z bazy
                patient_id = db.get_unprocessed_patient_id()
                if patient_id:
                    if load_patient_to_session(patient_id):
                        st.success("Pacjent załadowany. Rozpocznij wywiad.")
                        st.rerun()
                    else:
                        st.error("Błąd podczas ładowania pacjenta.")
                else:
                    # Gdy w bazie pusto – generujemy nowego pacjenta
                    new_patient_id = generate_patient()
                    if new_patient_id and load_patient_to_session(new_patient_id):
                        st.success("Nowy pacjent wygenerowany i załadowany.")
                        st.rerun()
                    else:
                        st.error("Nie udało się wygenerować pacjenta.")

    st.divider()
    if st.session_state.patient_scenario:
        # Karta pacjenta — możliwość ukrycia/pokazania
        col_card_title, col_card_btn = st.columns([3, 1])
        with col_card_title:
            st.subheader("Karta pacjenta")
        with col_card_btn:
            if "card_visible" not in st.session_state:
                st.session_state.card_visible = True
            if st.button("🔽 Ukryj kartę" if st.session_state.card_visible else "🔼 Pokaż kartę", 
                        use_container_width=True, type="secondary"):
                st.session_state.card_visible = not st.session_state.card_visible
                st.rerun()
        
        info = st.session_state.patient_basic_info
        def _val_or_missing(val: str) -> str:
            return val.strip() if isinstance(val, str) and val.strip() else "— brak danych (zbierz w wywiadzie)"
        
        if st.session_state.get("card_visible", True):
            # Ustal, czy pacjent pierwszy raz na podstawie stanu/parsowanej karty
            is_first = bool(st.session_state.patient_first_time or info.get("has_history_here") is False)

            colc1, colc2, colc3 = st.columns(3)
            with colc1:
                # Zawsze pokazuj imię i wiek
                st.markdown(f"**Imię i nazwisko:** {_val_or_missing(info.get('name',''))}")
                st.markdown(f"**Wiek:** {_val_or_missing(info.get('age',''))}")

            with colc2:
                hist = info.get("has_history_here")
                hist_label = (
                    "Nie (pacjent pierwszy raz)" if is_first else (
                        "— brak danych (zbierz w wywiadzie)" if hist is None else ("Tak" if hist else "Nie")
                    )
                )
                st.markdown(f"**Historia w punkcie:** {hist_label}")
                # Dodatowe pola tylko dla pacjentów z historią
                if not is_first:
                    st.markdown(f"**Choroby przewlekłe:** {_val_or_missing(info.get('chronic_diseases',''))}")

            with colc3:
                if not is_first:
                    st.markdown(f"**Operacje:** {_val_or_missing(info.get('operations',''))}")
                    st.markdown(f"**Uczulenia:** {_val_or_missing(info.get('allergies',''))}")

            if not is_first:
                st.markdown(f"**Wywiad rodzinny:** {_val_or_missing(info.get('family_history',''))}")

        # NIE pokazuj planu LLM na starcie — będzie widoczny po zakończeniu wywiadu

        st.divider()
        st.subheader("Rozmowa")
        
        # Chat history in tab
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                role = message.get("role", "assistant")
                if st.session_state.current_mode == "meta_ask":
                    speaker = "Lekarz" if role == "user" else "AI"
                else:
                    speaker = "Lekarz" if role == "user" else "Pacjent"
                with st.chat_message("user" if role == "user" else "assistant"):
                    st.markdown(f"**{speaker}:** {message.get('content', '')}")
        
        # Przełącznik trybu pod czatem
        col_mode, col_reset = st.columns([3, 1])
        with col_mode:
            mode = st.radio(
                "Kto zadaje pytanie?",
                ("Lekarz", "Pacjent", "Dopytaj AI"),
                index=0,
                horizontal=True,
            )
        with col_reset:
            st.write("")
            if st.button("🔄 Resetuj wywiad", use_container_width=True, type="secondary"):
                st.session_state.chat_history = []
                st.session_state.interview_end_mode = None
                st.session_state.correct_treatment_plan = None
                st.session_state.user_treatment_response = None
                st.session_state.diagnosis_evaluation = None
                if st.session_state.patient_id:
                    try:
                        st.session_state.conversation_id = db.create_conversation(
                            patient_id=st.session_state.patient_id, title="Reset interview"
                        )
                    except Exception as exc:
                        st.warning(f"Nie udało się utworzyć nowej konwersacji: {exc}")
                st.success("Wywiad zresetowany. Możesz rozpocząć od nowa.")
                st.rerun()
        st.session_state.current_mode = (
            "doctor_asks" if mode == "Lekarz" else "patient_asks" if mode == "Pacjent" else "meta_ask"
        )

        # Voice input section (only if not in end interview mode) — NAD przyciskiem Zakończ wywiad
        if st.session_state.interview_end_mode is None:
            st.caption("Możesz użyć mikrofonu lub wpisać pytanie")
            
            try:
                from streamlit_mic_recorder import mic_recorder
                
                # Mobile-friendly microphone button
                col_mic1, col_mic2, col_mic3 = st.columns([1, 2, 1])
                with col_mic2:
                    audio = mic_recorder(
                        start_prompt="🎤 Nagraj",
                        stop_prompt="⏹ Stop",
                        just_once=False,
                        use_container_width=True,
                        format="wav",
                        key="mic_recorder_main",
                    )
                
                if audio and audio.get("bytes"):
                    audio_bytes = audio["bytes"]
                    # Create hash of audio to avoid processing the same audio twice
                    audio_hash = hashlib.md5(audio_bytes).hexdigest()
                    
                    # Skip if this audio was already processed
                    if audio_hash == st.session_state.last_processed_audio_hash:
                        # Audio already processed, skip completely
                        pass
                    else:
                        # Mark this audio as processed BEFORE processing
                        st.session_state.last_processed_audio_hash = audio_hash
                        
                        # Save audio to temp file and transcribe
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_path = tmp_file.name
                        
                        try:
                            with st.spinner("Transkrypcja audio..."):
                                transcript = audio_processor.transcribe_audio_file(tmp_path, language="pl")
                                
                                # Automatically add transcript to chat and get response
                                if transcript and transcript.strip():
                                    transcript_text = transcript.strip()
                                    # Check if this message already exists to avoid duplicates
                                    if not st.session_state.chat_history or st.session_state.chat_history[-1].get("content") != transcript_text:
                                        # Add user message to chat
                                        st.session_state.chat_history.append({"role": "user", "content": transcript_text})
                                    
                                    # Save to database if conversation exists
                                    if st.session_state.conversation_id:
                                        try:
                                            db.add_message(st.session_state.conversation_id, role="user", content=transcript_text)
                                        except Exception as exc:
                                            st.warning(f"Nie udało się zapisać wiadomości: {exc}")
                                    
                                    # Get response based on current mode
                                    role_to_play = (
                                        "patient" if st.session_state.current_mode == "doctor_asks" else
                                        "doctor" if st.session_state.current_mode == "patient_asks" else
                                        "meta"
                                    )
                                    spinner_text = (
                                        "Pacjent się zastanawia..." if role_to_play == "patient" else
                                        "Lekarz się zastanawia..." if role_to_play == "doctor" else
                                        "AI analizuje..."
                                    )
                                    
                                    with st.spinner(spinner_text):
                                        full_prompt = prompt_manager.create_simulation_prompt(
                                            role_to_play=role_to_play,
                                            patient_scenario=st.session_state.patient_scenario,
                                            chat_history=st.session_state.chat_history,
                                            question=transcript_text,
                                        )
                                        response = llm_service.get_llm_response(
                                            full_prompt,
                                            model_name="google/gemini-2.5-flash-lite",
                                        )
                                        # Check if response already exists to avoid duplicates
                                        if not st.session_state.chat_history or st.session_state.chat_history[-1].get("content") != response:
                                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                                        
                                        # Save assistant response to database
                                        if st.session_state.conversation_id:
                                            try:
                                                db.add_message(st.session_state.conversation_id, role="assistant", content=response)
                                            except Exception as exc:
                                                st.warning(f"Nie udało się zapisać odpowiedzi: {exc}")
                                    
                                    st.rerun()
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
            except ImportError as e:
                st.error(f"❌ Biblioteka nagrywania audio nie jest zainstalowana: {e}")
                st.info("📦 Na serwerze AWS wykonaj: `uv add streamlit-mic-recorder` i zrestartuj aplikację")
                st.warning("⚠️ **Ważne:** Nagrywanie audio wymaga HTTPS. Upewnij się, że aplikacja działa przez HTTPS (nie HTTP).")
            except Exception as e:
                st.error(f"❌ Błąd nagrywania audio: {e}")
                st.info("💡 **Porady:**")
                st.markdown("""
                - Upewnij się, że aplikacja działa przez **HTTPS** (przeglądarki wymagają HTTPS do dostępu do mikrofonu)
                - Sprawdź, czy przeglądarka pozwala na dostęp do mikrofonu (sprawdź ikonę 🔒 w pasku adresu)
                - **Na telefonie:** Upewnij się, że przeglądarka ma uprawnienia do mikrofonu w ustawieniach telefonu
                - Sprawdź logi aplikacji na serwerze: `tail -f logs/app.err.log`
                """)

        # Input box (disabled during end interview mode)
        if st.session_state.interview_end_mode is None:
            input_placeholder = (
                "Zadaj pytanie pacjentowi..." if st.session_state.current_mode == "doctor_asks" else
                "Zadaj pytanie lekarzowi..." if st.session_state.current_mode == "patient_asks" else
                "Zadaj pytanie AI..."
            )
            if prompt := st.chat_input(input_placeholder):
                # Check if this message already exists to avoid duplicates
                if not st.session_state.chat_history or st.session_state.chat_history[-1].get("content") != prompt:
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Determine role to play based on mode
                role_to_play = (
                    "patient" if st.session_state.current_mode == "doctor_asks" else
                    "doctor" if st.session_state.current_mode == "patient_asks" else
                    "meta"
                )
                spinner_text = (
                    "Pacjent się zastanawia..." if role_to_play == "patient" else
                    "Lekarz się zastanawia..." if role_to_play == "doctor" else
                    "AI analizuje..."
                )
                with st.spinner(spinner_text):
                    full_prompt = prompt_manager.create_simulation_prompt(
                        role_to_play=role_to_play,
                        patient_scenario=st.session_state.patient_scenario,
                        chat_history=st.session_state.chat_history,
                        question=prompt,
                    )
                    response = llm_service.get_llm_response(
                        full_prompt,
                        model_name="google/gemini-2.5-flash-lite",
                    )
                    # Check if response already exists to avoid duplicates
                    if not st.session_state.chat_history or st.session_state.chat_history[-1].get("content") != response:
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    # Persist messages if we have a conversation
                    if st.session_state.conversation_id:
                        try:
                            db.add_message(st.session_state.conversation_id, role="user", content=prompt)
                            db.add_message(st.session_state.conversation_id, role="assistant", content=response)
                        except Exception as exc:
                            st.warning(f"Nie udało się zapisać wiadomości: {exc}")
                st.rerun()

        # Przycisk zakończ wywiad — POD polem tekstowym
        if st.session_state.interview_end_mode is None and st.session_state.chat_history:
            st.divider()
            col_end1, col_end2, col_end3 = st.columns([2, 1, 2])
            with col_end2:
                if st.button("✅ Zakończ wywiad", use_container_width=True, type="secondary"):
                    # Use pre-generated treatment plan from database if available
                    if st.session_state.correct_treatment_plan:
                        st.session_state.interview_end_mode = "waiting_for_response"
                    elif st.session_state.patient_id:
                        treatment_plan = db.get_patient_treatment_plan(st.session_state.patient_id)
                        if treatment_plan:
                            st.session_state.correct_treatment_plan = treatment_plan
                            st.session_state.interview_end_mode = "waiting_for_response"
                        else:
                            with st.spinner("Generowanie planu postępowania..."):
                                plan_prompt = prompt_manager.generate_treatment_plan_prompt(
                                    patient_scenario=st.session_state.patient_scenario,
                                    chat_history=st.session_state.chat_history,
                                )
                                correct_plan = llm_service.get_llm_response(
                                    plan_prompt,
                                    model_name="google/gemini-2.5-flash-lite",
                                )
                                st.session_state.correct_treatment_plan = correct_plan
                                st.session_state.interview_end_mode = "waiting_for_response"
                    else:
                        with st.spinner("Generowanie planu postępowania..."):
                            plan_prompt = prompt_manager.generate_treatment_plan_prompt(
                                patient_scenario=st.session_state.patient_scenario,
                                chat_history=st.session_state.chat_history,
                            )
                            correct_plan = llm_service.get_llm_response(
                                plan_prompt,
                                model_name="google/gemini-2.5-flash-lite",
                            )
                            st.session_state.correct_treatment_plan = correct_plan
                            st.session_state.interview_end_mode = "waiting_for_response"
                    st.rerun()

        # End interview section - waiting for user response
        if st.session_state.interview_end_mode == "waiting_for_response":
            st.divider()
            st.subheader("📋 Zakończenie wywiadu")
            st.info(
                "Co po takim wywiadzie przepiszesz, zalecisz lub jakie badania zlecisz?\n\n"
                "Opisz swoje leki, zalecenia i badania poniżej (możesz użyć tekstu lub głosu)."
            )
            
            # Text input for treatment plan
            # Display current response (will be updated by voice input)
            user_response = st.text_area(
                "Twoja odpowiedź:",
                value=st.session_state.user_treatment_response or "",
                height=150,
                placeholder="Np. Przepiszę ibuprofen 400mg 3x dziennie, zaleciłbym odpoczynek i obfite nawadnianie, zleciłbym morfologię krwi i CRP...",
                key="treatment_response_input"
            )
            
            # Sync text area changes to session state (for manual typing)
            if user_response != (st.session_state.user_treatment_response or ""):
                st.session_state.user_treatment_response = user_response
            
            # Voice input for treatment plan (if available)
            try:
                from streamlit_mic_recorder import mic_recorder
                col_voice1, col_voice2 = st.columns([1, 5])
                with col_voice1:
                    voice_audio = mic_recorder(
                        start_prompt="🎤 Nagraj",
                        stop_prompt="⏹ Stop",
                        just_once=False,
                        use_container_width=True,
                        format="wav",
                        key="mic_recorder_end_interview",
                    )
                
                if voice_audio and voice_audio.get("bytes"):
                    audio_bytes = voice_audio["bytes"]
                    audio_hash = hashlib.md5(audio_bytes).hexdigest()
                    
                    # Use separate hash for end interview audio to avoid conflicts
                    if audio_hash == st.session_state.last_end_interview_audio_hash:
                        # Audio already processed, skip completely
                        pass
                    else:
                        # Mark this audio as processed BEFORE processing
                        st.session_state.last_end_interview_audio_hash = audio_hash
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_path = tmp_file.name
                        
                        try:
                            with st.spinner("Transkrypcja odpowiedzi..."):
                                transcript = audio_processor.transcribe_audio_file(tmp_path, language="pl")
                                if transcript and transcript.strip():
                                    transcript_text = transcript.strip()
                                    # Set the response (replace existing if any)
                                    st.session_state.user_treatment_response = transcript_text
                                    
                                    # Automatically trigger evaluation if we have a treatment plan
                                    if st.session_state.correct_treatment_plan:
                                        with st.spinner("Analizowanie odpowiedzi i porównywanie z poprawnym planem..."):
                                            eval_prompt = prompt_manager.generate_diagnosis_evaluation_prompt(
                                                correct_plan=st.session_state.correct_treatment_plan,
                                                user_response=st.session_state.user_treatment_response,
                                                patient_scenario=st.session_state.patient_scenario,
                                                chat_history=st.session_state.chat_history,
                                            )
                                            evaluation = llm_service.get_llm_response(
                                                eval_prompt,
                                                model_name="google/gemini-2.5-flash-lite",
                                            )
                                            st.session_state.diagnosis_evaluation = evaluation
                                            st.session_state.interview_end_mode = "evaluated"
                                            
                                            # Save to database
                                            if st.session_state.conversation_id:
                                                try:
                                                    db.update_conversation_treatment_response(
                                                        conversation_id=st.session_state.conversation_id,
                                                        user_response=st.session_state.user_treatment_response,
                                                        evaluation=evaluation,
                                                    )
                                                except Exception as exc:
                                                    st.warning(f"Nie udało się zapisać odpowiedzi i oceny: {exc}")
                                        st.rerun()
                                    else:
                                        # No plan available, just show the transcript
                                        st.rerun()
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
            except ImportError as e:
                st.error(f"❌ Biblioteka nagrywania audio nie jest zainstalowana: {e}")
                st.info("📦 Na serwerze AWS wykonaj: `uv add streamlit-mic-recorder` i zrestartuj aplikację")
                st.warning("⚠️ **Ważne:** Nagrywanie audio wymaga HTTPS. Upewnij się, że aplikacja działa przez HTTPS (nie HTTP).")
            except Exception as e:
                st.error(f"❌ Błąd nagrywania audio: {e}")
                st.info("💡 **Porady:**")
                st.markdown("""
                - Upewnij się, że aplikacja działa przez **HTTPS** (przeglądarki wymagają HTTPS do dostępu do mikrofonu)
                - Sprawdź, czy przeglądarka pozwala na dostęp do mikrofonu (sprawdź ikonę 🔒 w pasku adresu)
                - **Na telefonie:** Upewnij się, że przeglądarka ma uprawnienia do mikrofonu w ustawieniach telefonu
                - Sprawdź logi aplikacji na serwerze: `tail -f logs/app.err.log`
                """)
            
            col_submit1, col_submit2 = st.columns([1, 5])
            with col_submit1:
                if st.button("✅ Wyślij odpowiedź", use_container_width=True, type="secondary"):
                    if not user_response.strip():
                        st.warning("Wprowadź odpowiedź przed wysłaniem.")
                    else:
                        st.session_state.user_treatment_response = user_response.strip()
                        if st.session_state.correct_treatment_plan:
                            with st.spinner("Analizowanie odpowiedzi i porównywanie z poprawnym planem..."):
                                eval_prompt = prompt_manager.generate_diagnosis_evaluation_prompt(
                                    correct_plan=st.session_state.correct_treatment_plan,
                                    user_response=st.session_state.user_treatment_response,
                                    patient_scenario=st.session_state.patient_scenario,
                                    chat_history=st.session_state.chat_history,
                                )
                                evaluation = llm_service.get_llm_response(
                                    eval_prompt,
                                    model_name="google/gemini-2.5-flash-lite",
                                )
                                st.session_state.diagnosis_evaluation = evaluation
                                st.session_state.interview_end_mode = "evaluated"
                                
                                # Save to database
                                if st.session_state.conversation_id:
                                    try:
                                        db.update_conversation_treatment_response(
                                            conversation_id=st.session_state.conversation_id,
                                            user_response=st.session_state.user_treatment_response,
                                            evaluation=evaluation,
                                        )
                                    except Exception as exc:
                                        st.warning(f"Nie udało się zapisać odpowiedzi i oceny: {exc}")
                            st.rerun()
                        else:
                            st.error("Błąd: Brak wygenerowanego planu postępowania. Spróbuj ponownie.")
        
        # End interview section - show evaluation results
        if st.session_state.interview_end_mode == "evaluated":
            st.divider()
            st.subheader("📊 Ocena Twojej diagnozy")
            
            # Show correct plan
            with st.expander("📋 Poprawny plan postępowania (do porównania)", expanded=False):
                st.markdown(st.session_state.correct_treatment_plan)
            
            # Show user's response
            with st.expander("💬 Twoja odpowiedź", expanded=False):
                st.markdown(st.session_state.user_treatment_response)
            
            # Show evaluation
            st.markdown("### 🎯 Ocena i uwagi")
            st.markdown(st.session_state.diagnosis_evaluation)
            
            # Option to retry
            if st.button("🔄 Spróbuj ponownie", use_container_width=True):
                st.session_state.interview_end_mode = "waiting_for_response"
                st.session_state.user_treatment_response = None
                st.session_state.diagnosis_evaluation = None
                st.rerun()
    else:
        st.subheader("Tryb wywiadu")
        st.info("Najpierw wygeneruj pacjenta, aby móc rozpocząć wywiad.")

# Helper function for processing a single chunk during recording
def process_chunk_async(audio_bytes: bytes, chunk_number: int, conversation_id: str) -> Optional[str]:
    """Process a single chunk: transcribe + LLM analysis + save suggestions.
    
    Returns:
        Suggestion text if successful, None otherwise
    """
    try:
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Transcribe chunk
        transcript = audio_processor.transcribe_audio_file(tmp_path, language="pl")
        
        if not transcript or not transcript.strip():
            os.unlink(tmp_path)
            return None
        
        # Get previous chunks for context
        previous_chunks = []
        try:
            prev_transcripts = db.get_interview_transcripts(conversation_id)
            for chunk_num, transcript_json in prev_transcripts:
                if chunk_num < chunk_number:
                    previous_chunks.append({
                        "chunk_number": chunk_num,
                        "transcript": transcript_json.get("transcript", [])
                    })
        except Exception:
            pass
        
        # LLM: assign roles and generate suggestions
        prompt = prompt_manager.assign_roles_and_suggest(
            transcript=transcript,
            context=None,  # No patient scenario for recorded interviews
            previous_chunks=previous_chunks if previous_chunks else None
        )
        
        response = llm_service.get_llm_response(
            prompt,
            model_name="google/gemini-2.5-flash-lite"
        )
        
        # Extract JSON from response
        def extract_json_from_response(text: str) -> str:
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:].strip()
            elif text.startswith("```"):
                text = text[3:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
            return text
        
        try:
            json_text = extract_json_from_response(response)
            result = json.loads(json_text)
            transcript_with_roles = result.get("transcript_with_roles", [])
            suggestions = result.get("suggestions", "")
            
            # Save transcript chunk
            db.save_interview_transcript(
                conversation_id=conversation_id,
                chunk_number=chunk_number,
                transcript_json={"transcript": transcript_with_roles}
            )
            
            # Save suggestions
            minute_number = chunk_number * 1  # Each chunk is 10 seconds (for testing)
            db.save_interview_suggestion(
                conversation_id=conversation_id,
                chunk_number=chunk_number,
                minute_number=minute_number,
                suggestions=suggestions
            )
            
            # Return suggestion text for immediate display
            return suggestions
        except json.JSONDecodeError:
            return None  # Skip if parsing fails
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        
        return None
    
    except Exception:
        return None  # Fail silently for background processing


# Helper function for processing interview audio
def process_interview_audio(audio_file: str):
    """Process complete interview audio (transcription + LLM analysis + summary)."""
    if not st.session_state.conversation_id:
        st.error("Brak conversation_id - nie można przetworzyć wywiadu")
        return
    
    try:
        # Step 1: Transcribe audio with Whisper
        with st.spinner("🎤 Transkrypcja audio (Whisper)..."):
            transcript = audio_processor.transcribe_audio_file(audio_file, language="pl")
        
        if not transcript or not transcript.strip():
            st.error("❌ Nie udało się transkrypcji audio.")
            return
        
        st.success(f"✅ Transkrypcja zakończona ({len(transcript)} znaków)")
        
        # Step 2: LLM - Assign roles, create summary, and extract info in ONE call
        with st.spinner("🤖 Analiza LLM (przypisanie ról, podsumowanie, ekstrakcja)..."):
            prompt = prompt_manager.process_interview_transcript(
                transcript=transcript,
                patient_scenario=st.session_state.patient_scenario if st.session_state.patient_scenario else None
            )
            
            response = llm_service.get_llm_response(
                prompt,
                model_name="google/gemini-2.5-flash-lite"
            )
        
        # Extract JSON from response (handle markdown code blocks)
        def extract_json_from_response(text: str) -> str:
            """Extract JSON from response, handling markdown code blocks."""
            # Remove markdown code block markers
            text = text.strip()
            
            # Remove ```json or ``` at the start
            if text.startswith("```json"):
                text = text[7:].strip()
            elif text.startswith("```"):
                text = text[3:].strip()
            
            # Remove ``` at the end
            if text.endswith("```"):
                text = text[:-3].strip()
            
            return text
        
        # Parse JSON response
        try:
            json_text = extract_json_from_response(response)
            
            result = json.loads(json_text)
            transcript_with_roles = result.get("transcript_with_roles", [])
            summary = result.get("summary", "")
            extracted_info = result.get("extracted_info", {})
            
            # Save transcript to database
            try:
                db.save_interview_transcript(
                    conversation_id=st.session_state.conversation_id,
                    chunk_number=1,
                    transcript_json={"transcript": transcript_with_roles}
                )
                st.success("✅ Transkrypcja z rolami zapisana do bazy")
            except Exception as e:
                st.error(f"❌ Błąd zapisu transkrypcji: {e}")
                import traceback
                st.text(traceback.format_exc())
            
            # Save summary and extracted info to database
            try:
                db.update_conversation_interview_summary(
                    conversation_id=st.session_state.conversation_id,
                    summary=summary,
                    extracted_info=extracted_info
                )
                st.success("✅ Podsumowanie i ekstrakcja zapisane do bazy")
            except Exception as e:
                st.error(f"❌ Błąd zapisu podsumowania: {e}")
                import traceback
                st.text(traceback.format_exc())
            
            # Mark conversation as processed
            try:
                from modules.db import get_cursor
                with get_cursor() as cur:
                    cur.execute(
                        "update conversations set diagnosis_evaluation = %s where id = %s",
                        ("OBSŁUŻONY - Nagrany wywiad", st.session_state.conversation_id)
                    )
            except Exception as e:
                st.warning(f"⚠️ Nie udało się oznaczyć konwersacji jako obsłużonej: {e}")
            
            # Update session state
            st.session_state.interview_summary = summary
            st.session_state.interview_extracted_info = extracted_info
            
            st.success("🎉 Wywiad przetworzony pomyślnie!")
            
        except json.JSONDecodeError as e:
            # Show error in UI
            st.error(f"❌ Błąd parsowania odpowiedzi LLM: {e}")
            st.text("Odpowiedź LLM (pierwsze 2000 znaków):")
            st.text(response[:2000])
            st.text("\nOstatnie 1000 znaków:")
            st.text(response[-1000:] if len(response) > 1000 else response)
    
    except Exception as exc:
        st.error(f"Błąd przetwarzania wywiadu: {exc}")


# Helper functions for interview processing
def process_all_chunks(all_chunks: List[str], last_fragment: Optional[str]):
    """Process all chunks and generate final summary."""
    if not st.session_state.conversation_id:
        return
    
    # Process all chunks
    chunks_to_process = all_chunks.copy()
    if last_fragment:
        chunks_to_process.append(last_fragment)
    
    # Process chunks using the new function signature
    for idx, chunk_file in enumerate(chunks_to_process, 1):
        # Read audio file and process
        if os.path.exists(chunk_file) and st.session_state.conversation_id:
            with open(chunk_file, 'rb') as f:
                audio_bytes = f.read()
            process_chunk_async(
                audio_bytes=audio_bytes,
                chunk_number=idx,
                conversation_id=st.session_state.conversation_id
            )
    
    # Get all transcripts
    transcripts = db.get_interview_transcripts(st.session_state.conversation_id)
    
    # Prepare transcripts for summary
    transcripts_with_roles = []
    for chunk_num, transcript_json in transcripts:
        transcripts_with_roles.append({
            "chunk_number": chunk_num,
            "transcript": transcript_json.get("transcript", [])
        })
    
    # Generate summary
    summary_prompt = prompt_manager.generate_interview_summary(
        transcripts_with_roles=transcripts_with_roles,
        patient_scenario=st.session_state.patient_scenario
    )
    
    summary = llm_service.get_llm_response(
        summary_prompt,
        model_name="google/gemini-2.5-flash-lite"
    )
    
    # Extract medical info
    extract_prompt = prompt_manager.extract_medical_info(
        transcripts_with_roles=transcripts_with_roles,
        doctor_proposals=None,  # TODO: Add doctor proposals input
        patient_scenario=st.session_state.patient_scenario
    )
    
    extract_response = llm_service.get_llm_response(
        extract_prompt,
        model_name="google/gemini-2.5-flash-lite"
    )
    
    # Parse extracted info
    try:
        extracted_info = json.loads(extract_response)
    except json.JSONDecodeError:
        extracted_info = {}
    
    # Save to database
    db.update_conversation_interview_summary(
        conversation_id=st.session_state.conversation_id,
        summary=summary,
        extracted_info=extracted_info
    )
    
    # Update session state
    st.session_state.interview_summary = summary
    st.session_state.interview_extracted_info = extracted_info


with tab_interview:
    # Interview recording section
    st.subheader("🎙️ Nagrywanie wywiadu")
    
    # Initialize recording state
    if "interview_audio_file" not in st.session_state:
        st.session_state.interview_audio_file = None
    if "interview_recording_started" not in st.session_state:
        st.session_state.interview_recording_started = False
    
    # Start recording button
    if not st.session_state.interview_recording_started:
        if st.button("▶️ Rozpocznij wywiad", type="secondary", use_container_width=False):
            # Create conversation for interview (with or without patient_id)
            try:
                patient_id = st.session_state.patient_id if st.session_state.patient_id else None
                # If no patient_id, create a temporary patient record for the interview
                if not patient_id:
                    # Create a temporary patient with minimal info
                    patient_id = db.create_patient(
                        scenario="Wywiad z rzeczywistym pacjentem",
                        summary="Wywiad nagrany"
                    )
                    st.session_state.patient_id = patient_id
                
                conv_id = db.create_conversation(
                    patient_id=patient_id,
                    title="Nagrany wywiad"
                )
                st.session_state.conversation_id = conv_id
                st.session_state.interview_recording_started = True
                st.session_state.interview_recording_start_time = time.time()
                st.session_state.interview_audio_file = None
                # Clear old interview data
                st.session_state.interview_summary = None
                st.session_state.interview_extracted_info = None
                st.rerun()
            except Exception as exc:
                st.error(f"Błąd tworzenia konwersacji: {exc}")
    
    # Recording interface
    if st.session_state.interview_recording_started:
        # Calculate duration
        if st.session_state.interview_recording_start_time:
            duration = time.time() - st.session_state.interview_recording_start_time
        else:
            duration = 0.0
        
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        duration_str = f"{minutes:02d}:{seconds:02d}"
        
        st.info(f"⏺️ Nagrywanie w toku... | Czas: {duration_str}")
        
        # Simple microphone recorder
        recording_active = False
        try:
            from streamlit_mic_recorder import mic_recorder
            
            col_mic1, col_mic2, col_mic3 = st.columns([1, 2, 1])
            with col_mic2:
                audio = mic_recorder(
                    start_prompt="🎤 Nagrywanie",
                    stop_prompt="⏹ Zatrzymaj nagranie",
                    just_once=False,
                    use_container_width=True,
                    format="wav",
                    key="interview_mic_recorder",
                )
            
            # Check if recording is active (no audio bytes yet means recording is in progress)
            if audio is None or not audio.get("bytes"):
                recording_active = True
            
            # Save audio when recording stops
            if audio and audio.get("bytes"):
                audio_bytes = audio["bytes"]
                audio_hash = hashlib.md5(audio_bytes).hexdigest()
                
                # Check if this is new audio (not already saved)
                if audio_hash != st.session_state.get("last_interview_audio_hash"):
                    st.session_state.last_interview_audio_hash = audio_hash
                    
                    # Save audio to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_bytes)
                        tmp_path = tmp_file.name
                    
                    st.session_state.interview_audio_file = tmp_path
        
        except ImportError:
            st.error("❌ Biblioteka streamlit-mic-recorder nie jest zainstalowana")
            st.info("📦 Wykonaj: `uv add streamlit-mic-recorder` i zrestartuj aplikację")
        except Exception as e:
            st.error(f"❌ Błąd nagrywania: {e}")
        
        # Stop button - disabled when recording is active
        if st.button(
            "⏹️ Zakończ wywiad", 
            type="secondary", 
            use_container_width=False,
            disabled=recording_active
        ):
            st.session_state.interview_recording_started = False
            
            # Process final audio if available
            if st.session_state.interview_audio_file and os.path.exists(st.session_state.interview_audio_file):
                with st.spinner("Przetwarzanie wywiadu (transkrypcja + analiza + podsumowanie)..."):
                    process_interview_audio(st.session_state.interview_audio_file)
                st.success("✅ Wywiad zakończony i przetworzony!")
                # Clean up
                try:
                    os.unlink(st.session_state.interview_audio_file)
                    st.session_state.interview_audio_file = None
                except Exception:
                    pass
                st.rerun()
            elif st.session_state.conversation_id:
                # Sprawdź czy są transkrypcje chunków w bazie (nowy mechanizm chunkowania)
                try:
                    transcripts = db.get_interview_transcripts(st.session_state.conversation_id)
                    if transcripts:
                        # Wygeneruj podsumowanie z wszystkich chunków
                        with st.spinner("Generowanie podsumowania z wszystkich chunków..."):
                            transcripts_with_roles = []
                            for chunk_num, transcript_json in sorted(transcripts, key=lambda x: x[0]):
                                transcripts_with_roles.append({
                                    "chunk_number": chunk_num,
                                    "transcript": transcript_json.get("transcript", [])
                                })
                            
                            # Wygeneruj podsumowanie
                            summary_prompt = prompt_manager.generate_interview_summary(
                                transcripts_with_roles=transcripts_with_roles,
                                patient_scenario=st.session_state.patient_scenario if st.session_state.patient_scenario else None
                            )
                            
                            summary = llm_service.get_llm_response(
                                summary_prompt,
                                model_name="google/gemini-2.5-flash-lite"
                            )
                            
                            # Wygeneruj ekstrakcję informacji
                            extract_prompt = prompt_manager.extract_medical_info(
                                transcripts_with_roles=transcripts_with_roles,
                                doctor_proposals=None,
                                patient_scenario=st.session_state.patient_scenario if st.session_state.patient_scenario else None
                            )
                            
                            extract_response = llm_service.get_llm_response(
                                extract_prompt,
                                model_name="google/gemini-2.5-flash-lite"
                            )
                            
                            # Parsuj extracted info
                            try:
                                extracted_info = json.loads(extract_response)
                            except json.JSONDecodeError:
                                extracted_info = {}
                            
                            # Zapisz podsumowanie do bazy
                            db.update_conversation_interview_summary(
                                conversation_id=st.session_state.conversation_id,
                                summary=summary,
                                extracted_info=extracted_info
                            )
                            
                            # Oznacz conversation jako "Nagrany wywiad" (jeśli jeszcze nie jest)
                            try:
                                from modules.db import get_cursor
                                with get_cursor() as cur:
                                    cur.execute(
                                        "update conversations set title = %s, diagnosis_evaluation = %s where id = %s",
                                        ("Nagrany wywiad", "OBSŁUŻONY - Nagrany wywiad", st.session_state.conversation_id)
                                    )
                            except Exception as e:
                                st.warning(f"⚠️ Nie udało się oznaczyć konwersacji: {e}")
                            
                            # Zaktualizuj session state
                            st.session_state.interview_summary = summary
                            st.session_state.interview_extracted_info = extracted_info
                            
                            st.success("✅ Wywiad zakończony i przetworzony! Podsumowanie zostało zapisane do bazy.")
                    else:
                        st.warning("⚠️ Brak transkrypcji w bazie. Wywiad może być jeszcze w trakcie przetwarzania.")
                except Exception as e:
                    st.error(f"❌ Błąd podczas przetwarzania wywiadu: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                st.rerun()
            else:
                st.warning("⚠️ Brak nagranego audio i conversation_id. Nagraj wywiad używając mikrofonu powyżej.")
                st.rerun()
    
    # Display transcript and summary after recording (only if not currently recording)
    if st.session_state.conversation_id and not st.session_state.interview_recording_started:
        # Show transcript if available
        try:
            transcripts = db.get_interview_transcripts(st.session_state.conversation_id)
            if transcripts:
                st.divider()
                st.subheader("📝 Transkrypcja wywiadu")
                for chunk_num, transcript_json in transcripts:
                    transcript_segments = transcript_json.get("transcript", [])
                    for segment in transcript_segments:
                        role = segment.get("role", "unknown")
                        text = segment.get("text", "")
                        timestamp = segment.get("timestamp", 0.0)
                        
                        role_label = "Lekarz" if role == "doctor" else "Pacjent"
                        role_icon = "👨‍⚕️" if role == "doctor" else "👤"
                        
                        st.markdown(f"**{role_icon} {role_label}** ({timestamp:.1f}s):")
                        st.markdown(text)
                        st.markdown("---")
        except Exception:
            pass  # Ignore errors for transcript
    
    # Display summary after recording (only if not currently recording)
    if st.session_state.interview_summary and not st.session_state.interview_recording_started:
        st.divider()
        st.subheader("📋 Podsumowanie wywiadu")
        st.markdown(st.session_state.interview_summary)
        
        if st.session_state.interview_extracted_info:
            st.subheader("💊 Ekstrakcja informacji")
            info = st.session_state.interview_extracted_info
            
            if info.get("leki"):
                st.write("**Leki:**")
                for lek in info["leki"]:
                    st.write(f"- {lek}")
            
            if info.get("zalecenia"):
                st.write("**Zalecenia:**")
                for zalecenie in info["zalecenia"]:
                    st.write(f"- {zalecenie}")
            
            if info.get("badania"):
                st.write("**Badania:**")
                for badanie in info["badania"]:
                    st.write(f"- {badanie}")
            
            if info.get("ocena_propozycji_lekarza"):
                st.write("**Ocena propozycji lekarza:**")
                st.info(info["ocena_propozycji_lekarza"])
            
            if info.get("sugestie_poprawek"):
                st.write("**Sugestie poprawek:**")
                st.warning(info["sugestie_poprawek"])
    
    st.divider()
    st.subheader("Ręczne tworzenie wywiadu")
    
    # Patient scenario input
    st.session_state.manual_patient_scenario = st.text_area(
        "Scenariusz pacjenta (opcjonalnie):",
        value=st.session_state.manual_patient_scenario,
        height=100,
        key="manual_scenario_input",
        help="Możesz wpisać krótki opis pacjenta lub zostawić puste",
    )
    
    st.divider()
    
    # Role selector and message input
    col_role, col_input, col_add = st.columns([1, 3, 1])
    with col_role:
        message_role = st.radio(
            "Rola:",
            ("Lekarz", "Pacjent"),
            key="manual_role",
            label_visibility="collapsed",
        )
    with col_input:
        message_text = st.text_input(
            "Wiadomość:",
            key="manual_message_input",
            label_visibility="collapsed",
            placeholder="Wpisz wiadomość...",
        )
    with col_add:
        st.write("")
        if st.button("➕ Dodaj", use_container_width=True):
            if message_text.strip():
                role_key = "user" if message_role == "Lekarz" else "assistant"
                st.session_state.manual_interview_history.append({
                    "role": role_key,
                    "content": message_text.strip(),
                })
                # Clear input by removing from session state
                if "manual_message_input" in st.session_state:
                    del st.session_state.manual_message_input
                st.rerun()
    
    st.divider()
    
    # Chat history display
    if st.session_state.manual_interview_history:
        st.subheader("Historia rozmowy")
        for i, msg in enumerate(st.session_state.manual_interview_history):
            role = msg.get("role", "assistant")
            speaker = "Lekarz" if role == "user" else "Pacjent"
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(f"**{speaker}:** {msg.get('content', '')}")
        
        # Display summary and recommendations if available
        if st.session_state.manual_interview_summary:
            st.divider()
            st.subheader("📋 Podsumowanie wywiadu")
            st.markdown(st.session_state.manual_interview_summary)
        
        if st.session_state.manual_interview_recommendations:
            st.subheader("💊 Zalecenia")
            st.markdown(st.session_state.manual_interview_recommendations)
        
        st.divider()
        
        # End interview button (only show if summary not generated yet)
        if not st.session_state.manual_interview_summary:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("Po zakończeniu wywiadu kliknij przycisk poniżej, aby wygenerować podsumowanie i zalecenia.")
            with col2:
                if st.button("🏁 Koniec wywiadu", use_container_width=True, type="primary"):
                    with st.spinner("Generowanie podsumowania i zaleceń..."):
                        # Generate summary
                        try:
                            summary_prompt = prompt_manager.generate_interview_summary_prompt(
                                st.session_state.manual_interview_history
                            )
                            interview_summary = llm_service.get_llm_response(
                                summary_prompt,
                                model_name="google/gemini-2.5-flash-lite",
                            )
                        except Exception as exc:
                            interview_summary = f"Błąd generowania podsumowania: {exc}"
                        
                        # Generate recommendations
                        try:
                            rec_prompt = prompt_manager.generate_recommendations_prompt(
                                st.session_state.manual_interview_history,
                                interview_summary,
                            )
                            recommendations = llm_service.get_llm_response(
                                rec_prompt,
                                model_name="google/gemini-2.5-flash-lite",
                            )
                        except Exception as exc:
                            recommendations = f"Błąd generowania zaleceń: {exc}"
                        
                        # Save to session state
                        st.session_state.manual_interview_summary = interview_summary
                        st.session_state.manual_interview_recommendations = recommendations
                        
                        # Save to database
                        try:
                            # Create patient with scenario or summary
                            patient_scenario = (
                                st.session_state.manual_patient_scenario
                                if st.session_state.manual_patient_scenario
                                else interview_summary
                            )
                            # Generate patient summary for listing
                            summary_for_db = ""
                            if st.session_state.manual_patient_scenario:
                                try:
                                    patient_summary_prompt = prompt_manager.generate_patient_summary_prompt(
                                        patient_scenario
                                    )
                                    summary_for_db = llm_service.get_llm_response(
                                        patient_summary_prompt,
                                        model_name="google/gemini-2.5-flash-lite",
                                    ).strip().strip('"').strip("'")
                                except Exception:
                                    pass
                            
                            patient_id = db.create_patient(
                                scenario=patient_scenario,
                                summary=summary_for_db or "Wywiad ręczny",
                            )
                            conversation_id = db.create_conversation(
                                patient_id=patient_id,
                                title="Wywiad ręczny",
                            )
                            # Save all messages
                            for msg in st.session_state.manual_interview_history:
                                db.add_message(
                                    conversation_id=conversation_id,
                                    role=msg.get("role", "assistant"),
                                    content=msg.get("content", ""),
                                )
                            # Save summary and recommendations as special messages
                            db.add_message(
                                conversation_id=conversation_id,
                                role="assistant",
                                content=f"PODSUMOWANIE WYWIADU:\n\n{interview_summary}",
                            )
                            db.add_message(
                                conversation_id=conversation_id,
                                role="assistant",
                                content=f"ZALECENIA:\n\n{recommendations}",
                            )
                            
                            st.success("Wywiad zapisany w bazie danych!")
                            st.rerun()
                            
                        except Exception as exc:
                            st.error(f"Błąd podczas zapisywania do bazy: {exc}")
    else:
        st.info("Rozpocznij wywiad dodając pierwszą wiadomość jako Lekarz lub Pacjent.")

with tab_browse:
    st.subheader("Lista wywiadów")
    
    # Initialize selected conversation in session state
    if "selected_conversation_id" not in st.session_state:
        st.session_state.selected_conversation_id = None
    
    try:
        rows = db.list_conversations_with_patient(limit=50)
    except Exception as exc:
        rows = []
        st.warning(f"Nie udało się pobrać listy wywiadów: {exc}")

    if not rows:
        st.info("Brak zapisanych wywiadów.")
    else:
        # Debug: show count and types
        nagrane_count = sum(1 for row in rows if row[2] == "Nagrany wywiad")
        st.caption(f"Znaleziono {len(rows)} konwersacji (w tym {nagrane_count} nagranych wywiadów)")
        
        # Track displayed conversation IDs to avoid duplicates in UI
        displayed_ids = set()
        
        # List of conversations
        for row in rows:
            # Unpack row - handle both old format (4 elements) and new format (5 elements)
            conv_id = row[0]
            created_at = row[1]
            title = row[2]
            summary = row[3]
            diagnosis_evaluation = row[4] if len(row) > 4 else None
            
            # Skip if we've already displayed this conversation ID
            if conv_id in displayed_ids:
                continue
            displayed_ids.add(conv_id)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                # Determine display text based on conversation type
                if title == "Nagrany wywiad":
                    display_text = "🎙️ Nagrany wywiad"
                    icon = "🎙️"
                elif diagnosis_evaluation and "Skipped" in str(diagnosis_evaluation):
                    display_text = "⏭️ Skipped"
                    icon = "⏭️"
                else:
                    # Use patient summary (contains name and surname) or "Simulation"
                    if summary and summary != "Brak podsumowania":
                        display_text = summary
                    else:
                        display_text = "📋 Simulation"
                    icon = "📋"
                
                button_text = f"{icon} {display_text}"
                if st.button(button_text, key=f"conv_{conv_id}", use_container_width=True):
                    st.session_state.selected_conversation_id = conv_id
                    st.rerun()
            with col2:
                st.caption(f"{created_at}")
        
        # Show conversation details if selected
        if st.session_state.selected_conversation_id:
            st.divider()
            conv_id = st.session_state.selected_conversation_id
            
            # Show interview transcript if available
            try:
                transcripts = db.get_interview_transcripts(conv_id)
                if transcripts:
                    st.subheader("📝 Transkrypcja wywiadu")
                    for chunk_num, transcript_json in transcripts:
                        transcript_segments = transcript_json.get("transcript", [])
                        for segment in transcript_segments:
                            role = segment.get("role", "unknown")
                            text = segment.get("text", "")
                            timestamp = segment.get("timestamp", 0.0)
                            
                            role_label = "Lekarz" if role == "doctor" else "Pacjent"
                            role_icon = "👨‍⚕️" if role == "doctor" else "👤"
                            
                            st.markdown(f"**{role_icon} {role_label}** ({timestamp:.1f}s):")
                            st.markdown(text)
                            st.markdown("---")
            except Exception as exc:
                st.warning(f"Nie udało się pobrać transkrypcji: {exc}")
            
            # Show interview summary if available
            try:
                interview_data = db.get_conversation_interview_data(conv_id)
                if interview_data:
                    summary, extracted_info = interview_data
                    if summary:
                        st.divider()
                        st.subheader("📋 Podsumowanie wywiadu")
                        st.markdown(summary)
                    
                    if extracted_info:
                        st.subheader("💊 Ekstrakcja informacji")
                        if extracted_info.get("leki"):
                            st.write("**Leki:**")
                            for lek in extracted_info["leki"]:
                                st.write(f"- {lek}")
                        if extracted_info.get("zalecenia"):
                            st.write("**Zalecenia:**")
                            for zalecenie in extracted_info["zalecenia"]:
                                st.write(f"- {zalecenie}")
                        if extracted_info.get("badania"):
                            st.write("**Badania:**")
                            for badanie in extracted_info["badania"]:
                                st.write(f"- {badanie}")
                        if extracted_info.get("ocena_propozycji_lekarza"):
                            st.write("**Ocena propozycji lekarza:**")
                            st.info(extracted_info["ocena_propozycji_lekarza"])
                        if extracted_info.get("sugestie_poprawek"):
                            st.write("**Sugestie poprawek:**")
                            st.warning(extracted_info["sugestie_poprawek"])
            except Exception:
                pass  # Ignore errors for interview data
            
            st.divider()
            
            # Get conversation details
            try:
                conv_details = db.get_conversation_details(conv_id)
                if not conv_details:
                    st.error("Nie znaleziono wywiadu.")
                else:
                    patient_id, conv_title, user_response, evaluation = conv_details
                    
                    # Get patient details
                    patient_details = db.get_patient_by_id(patient_id)
                    if not patient_details:
                        st.error("Nie znaleziono pacjenta.")
                    else:
                        scenario, summary, treatment_plan, patient_created = patient_details
                        
                        # Get conversation messages
                        messages = db.get_conversation_messages(conv_id)
                        
                        # Display conversation details
                        st.subheader("📋 Szczegóły wywiadu")
                        
                        # Patient scenario (only for non-recorded interviews)
                        if conv_title != "Nagrany wywiad":
                            with st.expander("👤 Scenariusz pacjenta", expanded=True):
                                st.markdown(scenario)
                            st.divider()
                        
                        # Conversation messages (only for non-recorded interviews)
                        if conv_title != "Nagrany wywiad":
                            st.subheader("💬 Konwersacja")
                            if messages:
                                for msg_id, role, content, msg_time in messages:
                                    speaker = "Lekarz" if role == "user" else "Pacjent"
                                    with st.chat_message("user" if role == "user" else "assistant"):
                                        st.markdown(f"**{speaker}:** {content}")
                            else:
                                st.info("Brak wiadomości w tym wywiadzie.")
                            
                            st.divider()
                            
                            # User treatment response (only for non-recorded interviews)
                            if user_response:
                                st.subheader("💊 Twoje zalecenia/odpowiedź")
                                st.markdown(user_response)
                            else:
                                st.info("Brak zapisanych zaleczeń.")
                        
                        # Evaluation if available (for both recorded and non-recorded interviews)
                        if evaluation:
                            st.divider()
                            st.subheader("📊 Ocena diagnozy")
                            # For recorded interviews, show suggestions from LLM
                            if conv_title == "Nagrany wywiad":
                                # Get and display LLM suggestions
                                try:
                                    suggestions = db.get_interview_suggestions(conv_id)
                                    if suggestions:
                                        for chunk_num, minute_num, suggestion_text in suggestions:
                                            st.info(suggestion_text)
                                    else:
                                        # If no suggestions, try to get evaluation from extracted_info
                                        try:
                                            interview_data = db.get_conversation_interview_data(conv_id)
                                            if interview_data:
                                                summary, extracted_info = interview_data
                                                if extracted_info and extracted_info.get("ocena_propozycji_lekarza"):
                                                    st.markdown(extracted_info["ocena_propozycji_lekarza"])
                                                elif evaluation != "OBSŁUŻONY - Nagrany wywiad":
                                                    st.markdown(evaluation)
                                                else:
                                                    st.info("✅ Wywiad został nagrany i przetworzony.")
                                            else:
                                                if evaluation != "OBSŁUŻONY - Nagrany wywiad":
                                                    st.markdown(evaluation)
                                                else:
                                                    st.info("✅ Wywiad został nagrany i przetworzony.")
                                        except Exception:
                                            if evaluation != "OBSŁUŻONY - Nagrany wywiad":
                                                st.markdown(evaluation)
                                            else:
                                                st.info("✅ Wywiad został nagrany i przetworzony.")
                                except Exception:
                                    # Fallback to showing evaluation if available
                                    if evaluation != "OBSŁUŻONY - Nagrany wywiad":
                                        st.markdown(evaluation)
                                    else:
                                        st.info("✅ Wywiad został nagrany i przetworzony.")
                            else:
                                st.markdown(evaluation)
                        
                        # Back button
                        if st.button("← Wróć do listy", use_container_width=True):
                            st.session_state.selected_conversation_id = None
                            st.rerun()
                            
            except Exception as exc:
                st.error(f"Błąd podczas ładowania szczegółów wywiadu: {exc}")

with tab_admin:
    st.subheader("Zarządzanie danymi")
    
    # Mass patient generation
    st.markdown("### 📦 Generowanie pacjentów masowo")
    col_gen1, col_gen2 = st.columns([2, 1])
    with col_gen1:
        num_patients = st.number_input(
            "Liczba pacjentów do wygenerowania:",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            help="Podaj liczbę pacjentów, które chcesz wygenerować do puli"
        )
    with col_gen2:
        st.write("")  # Spacer
        if st.button("🚀 Generuj masowo", use_container_width=True, type="secondary"):
            if num_patients > 0:
                progress_bar = st.progress(0)
                status_text = st.empty()
                success_count = 0
                error_count = 0
                
                for i in range(num_patients):
                    try:
                        generate_patient()
                        success_count += 1
                    except Exception:
                        error_count += 1
                    
                    # Update progress
                    progress = (i + 1) / num_patients
                    progress_bar.progress(progress)
                    status_text.text(f"Wygenerowano: {success_count}/{num_patients} (błędy: {error_count})")
                
                progress_bar.empty()
                status_text.empty()
                
                if success_count > 0:
                    st.success(f"✅ Wygenerowano {success_count} pacjentów!")
                if error_count > 0:
                    st.warning(f"⚠️ {error_count} pacjentów nie zostało wygenerowanych z powodu błędów.")
                
                # Show current pool status
                try:
                    current_pool = db.count_unprocessed_patients()
                    st.info(f"Obecna liczba nieobsłużonych pacjentów w puli: {current_pool}")
                except Exception:
                    pass
            else:
                st.warning("Podaj liczbę większą niż 0")
    
    st.divider()
    
    # Database wipe (dangerous)
    st.markdown("### ⚠️ Zarządzanie danymi (niebezpieczne)")
    st.warning("Operacje w tej sekcji są nieodwracalne. Upewnij się, że wiesz co robisz.")
    confirm = st.text_input("Aby wyczyścić bazę wpisz: WYMAŻ", value="")
    if st.button("🗑️ Wyzeruj bazę danych", type="secondary"):
        if confirm.strip().upper() == "WYMAŻ":
            try:
                db.wipe_all_data()
                # Reset in-memory session
                st.session_state.patient_scenario = None
                st.session_state.chat_history = []
                st.session_state.patient_id = None
                st.session_state.conversation_id = None
                st.success("Baza danych wyczyszczona.")
            except Exception as exc:
                st.error(f"Błąd podczas czyszczenia bazy: {exc}")
        else:
            st.info("Potwierdź operację wpisując dokładnie: WYMAŻ")



