import streamlit as st
import tempfile
import os
import hashlib

from modules import llm_service, prompt_manager
from modules import db
from modules import audio_processor


# --- Page config ---
st.set_page_config(page_title="Symulator Pacjenta POZ", layout="wide")
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

# --- Initialize DB schema (no-op if exists) ---
try:
    db.init_schema()
except Exception as exc:
    st.warning(f"Database not initialized: {exc}")

# --- Sidebar: patient scenario and chat history ---
with st.sidebar:
    st.header("Pacjent")
    if st.session_state.patient_scenario:
        with st.expander("Scenariusz pacjenta", expanded=False):
            st.markdown(st.session_state.patient_scenario)
    else:
        st.info("Brak scenariusza pacjenta.")
    
    # Show HTTPS info if available
    st.divider()
    with st.expander("ℹ️ Informacje o HTTPS", expanded=False):
        st.caption("Aplikacja działa przez HTTPS")
        st.caption("Aby sprawdzić adres Cloudflare Tunnel:")
        st.code("""
# Na serwerze AWS:
./check_tunnel_url.sh

# Lub sprawdź logi:
tail -f /tmp/cloudflared.log

# Lub sprawdź systemd:
sudo journalctl -u cloudflared -f
        """)

# --- Tabs ---
tab_sim, tab_interview, tab_browse, tab_admin = st.tabs(["Symulacja", "Wywiad", "Przeglądanie", "Admin"])

with tab_sim:
    st.subheader("Generowanie pacjenta")
    col1, col2 = st.columns([3, 1])
    with col1:
        keywords_input = st.text_input(
            "Słowa kluczowe (opcjonalnie):",
            placeholder="np. kobieta, ból głowy, nadciśnienie",
        )
    with col2:
        if st.button("Wygeneruj", use_container_width=True):
            with st.spinner("Tworzenie scenariusza..."):
                prompt_messages = prompt_manager.generate_patient_scenario_prompt(keywords=keywords_input)
                scenario = llm_service.get_llm_response(
                    prompt_messages,
                    model_name="google/gemini-2.5-flash-lite",
                )
                st.session_state.patient_scenario = scenario
                st.session_state.chat_history = []  # Start with empty chat
                # Generate summary via LLM
                summary = ""
                try:
                    summary_prompt = prompt_manager.generate_patient_summary_prompt(scenario)
                    summary = llm_service.get_llm_response(
                        summary_prompt,
                        model_name="google/gemini-2.5-flash-lite",
                    ).strip()
                    # Clean summary if LLM added quotes or extra text
                    summary = summary.strip('"').strip("'").strip()
                except Exception as exc:
                    st.warning(f"Nie udało się wygenerować podsumowania: {exc}")
                
                # Generate treatment plan via LLM (before interview, based on scenario only)
                treatment_plan = ""
                try:
                    with st.spinner("Generowanie planu postępowania..."):
                        plan_prompt = prompt_manager.generate_treatment_plan_prompt(
                            patient_scenario=scenario,
                            chat_history=[],  # Empty chat history at this point
                        )
                        treatment_plan = llm_service.get_llm_response(
                            plan_prompt,
                            model_name="google/gemini-2.5-flash-lite",
                        )
                        # Store in session state for quick access
                        st.session_state.correct_treatment_plan = treatment_plan
                except Exception as exc:
                    st.warning(f"Nie udało się wygenerować planu postępowania: {exc}")
                
                # Persist patient and conversation
                try:
                    st.session_state.patient_id = db.create_patient(
                        scenario=scenario, 
                        summary=summary,
                        treatment_plan=treatment_plan
                    )
                    st.session_state.conversation_id = db.create_conversation(
                        patient_id=st.session_state.patient_id, title="Initial interview"
                    )
                except Exception as exc:
                    st.warning(f"Nie udało się zapisać pacjenta/konwersacji: {exc}")
                st.success("Nowy pacjent gotowy. Rozpocznij wywiad.")

    st.divider()
    if st.session_state.patient_scenario:
        st.subheader("Tryb wywiadu")
        col_mode, col_reset = st.columns([3, 1])
        with col_mode:
            mode = st.radio(
                "Kto zadaje pytanie?",
                ("Lekarz", "Pacjent", "Dopytaj AI"),
                index=0,  # Default to "Lekarz"
                horizontal=True,
            )
        with col_reset:
            st.write("")  # Spacer
            if st.button("🔄 Resetuj wywiad", use_container_width=True, type="secondary"):
                st.session_state.chat_history = []
                st.session_state.interview_end_mode = None
                st.session_state.correct_treatment_plan = None
                st.session_state.user_treatment_response = None
                st.session_state.diagnosis_evaluation = None
                # Create a new conversation for the same patient
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
        
        st.divider()
        st.subheader("Rozmowa")
        
        # Chat history in tab
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                role = message.get("role", "assistant")
                speaker = "Lekarz" if role == "user" else "Pacjent"
                with st.chat_message("user" if role == "user" else "assistant"):
                    st.markdown(f"**{speaker}:** {message.get('content', '')}")
        
        # End interview button - placed after chat history
        if st.session_state.interview_end_mode is None and st.session_state.chat_history:
            st.divider()
            col_end1, col_end2, col_end3 = st.columns([2, 1, 2])
            with col_end2:
                if st.button("✅ Zakończ wywiad", use_container_width=True, type="secondary"):
                    # Use pre-generated treatment plan from database if available
                    if st.session_state.correct_treatment_plan:
                        # Plan already exists (from patient creation or loaded from DB)
                        st.session_state.interview_end_mode = "waiting_for_response"
                    elif st.session_state.patient_id:
                        # Try to load from database
                        treatment_plan = db.get_patient_treatment_plan(st.session_state.patient_id)
                        if treatment_plan:
                            st.session_state.correct_treatment_plan = treatment_plan
                            st.session_state.interview_end_mode = "waiting_for_response"
                        else:
                            # Generate new plan based on interview (fallback)
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
                        # No patient ID, generate plan from scratch
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
        
        # Voice input section (only if not in end interview mode)
        if st.session_state.interview_end_mode is None:
            st.caption("Możesz użyć mikrofonu lub wpisać pytanie")
            
            try:
                from streamlit_mic_recorder import mic_recorder
                
                col_mic1, col_mic2, col_mic3 = st.columns([1, 1, 4])
                with col_mic2:
                    audio = mic_recorder(
                        start_prompt="🎤 Nagraj pytanie",
                        stop_prompt="⏹ Zatrzymaj",
                        just_once=False,
                        use_container_width=False,
                        format="wav",
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
                - Sprawdź logi aplikacji na serwerze: `tail -f logs/app.err.log`
                """)

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
                        start_prompt="🎤 Nagraj odpowiedź",
                        stop_prompt="⏹ Zatrzymaj",
                        just_once=False,
                        use_container_width=True,
                        format="wav",
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
    else:
        st.subheader("Tryb wywiadu")
        st.info("Najpierw wygeneruj pacjenta, aby móc rozpocząć wywiad.")

with tab_interview:
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
        # List of conversations
        for conv_id, created_at, title, summary in rows:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📋 {summary}", key=f"conv_{conv_id}", use_container_width=True):
                    st.session_state.selected_conversation_id = conv_id
                    st.rerun()
            with col2:
                st.caption(f"{created_at}")
        
        # Show conversation details if selected
        if st.session_state.selected_conversation_id:
            st.divider()
            conv_id = st.session_state.selected_conversation_id
            
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
                        
                        # Patient scenario
                        with st.expander("👤 Scenariusz pacjenta", expanded=True):
                            st.markdown(scenario)
                        
                        st.divider()
                        
                        # Conversation messages
                        st.subheader("💬 Konwersacja")
                        if messages:
                            for msg_id, role, content, msg_time in messages:
                                speaker = "Lekarz" if role == "user" else "Pacjent"
                                with st.chat_message("user" if role == "user" else "assistant"):
                                    st.markdown(f"**{speaker}:** {content}")
                        else:
                            st.info("Brak wiadomości w tym wywiadzie.")
                        
                        st.divider()
                        
                        # User treatment response
                        if user_response:
                            st.subheader("💊 Twoje zalecenia/odpowiedź")
                            st.markdown(user_response)
                        else:
                            st.info("Brak zapisanych zaleczeń.")
                        
                        # Evaluation if available
                        if evaluation:
                            st.divider()
                            st.subheader("📊 Ocena diagnozy")
                            st.markdown(evaluation)
                        
                        # Back button
                        if st.button("← Wróć do listy", use_container_width=True):
                            st.session_state.selected_conversation_id = None
                            st.rerun()
                            
            except Exception as exc:
                st.error(f"Błąd podczas ładowania szczegółów wywiadu: {exc}")

with tab_admin:
    st.subheader("Zarządzanie danymi (niebezpieczne)")
    st.warning("Operacje w tej sekcji są nieodwracalne. Upewnij się, że wiesz co robisz.")
    confirm = st.text_input("Aby wyczyścić bazę wpisz: WYMAŻ", value="")
    if st.button("🗑️ Wyzeruj bazę danych", type="primary"):
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



