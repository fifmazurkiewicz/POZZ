import streamlit as st
import tempfile
import os

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
if "pending_transcript" not in st.session_state:
    st.session_state.pending_transcript = None

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
                # Persist patient and conversation
                try:
                    st.session_state.patient_id = db.create_patient(scenario=scenario, summary=summary)
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
        
        # Voice input section
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
                # Save audio to temp file and transcribe
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    with st.spinner("Transkrypcja audio..."):
                        transcript = audio_processor.transcribe_audio_file(tmp_path)
                        if transcript and transcript.strip() and not transcript.lower().startswith("transcription failed"):
                            st.session_state.pending_transcript = transcript.strip()
                        else:
                            st.error("Nie udało się rozpoznać mowy. Spróbuj ponownie.")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        except ImportError:
            st.info("📦 Instalacja: `uv sync` (wymaga streamlit-mic-recorder)")
        # If we have a pending transcript, let user decide
        if st.session_state.pending_transcript:
            st.info("Rozpoznany tekst z mikrofonu:")
            st.markdown(f"> {st.session_state.pending_transcript}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Wstaw do czatu (bez odpowiedzi)"):
                    prompt = st.session_state.pending_transcript
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    if st.session_state.conversation_id:
                        try:
                            db.add_message(st.session_state.conversation_id, role="user", content=prompt)
                        except Exception as exc:
                            st.warning(f"Nie udało się zapisać wiadomości: {exc}")
                    st.session_state.pending_transcript = None
                    st.rerun()
            with c2:
                if st.button("Wyślij i uzyskaj odpowiedź"):
                    prompt = st.session_state.pending_transcript
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
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
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        if st.session_state.conversation_id:
                            try:
                                db.add_message(st.session_state.conversation_id, role="user", content=prompt)
                                db.add_message(st.session_state.conversation_id, role="assistant", content=response)
                            except Exception as exc:
                                st.warning(f"Nie udało się zapisać wiadomości: {exc}")
                    st.session_state.pending_transcript = None
                    st.rerun()

        # Input box
        input_placeholder = (
            "Zadaj pytanie pacjentowi..." if st.session_state.current_mode == "doctor_asks" else
            "Zadaj pytanie lekarzowi..." if st.session_state.current_mode == "patient_asks" else
            "Zadaj pytanie AI..."
        )
        if prompt := st.chat_input(input_placeholder):
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
    try:
        rows = db.list_conversations_with_patient(limit=50)
    except Exception as exc:
        rows = []
        st.warning(f"Nie udało się pobrać listy wywiadów: {exc}")

    if not rows:
        st.info("Brak zapisanych wywiadów.")
    else:
        for conv_id, created_at, title, summary in rows:
            with st.container():
                st.markdown(f"- **{summary}**")
                st.caption(f"{created_at} | {title or 'Wywiad'} | {conv_id}")

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



