import streamlit as st

from modules import llm_service, prompt_manager
from modules import db


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
tab_sim, tab_browse, tab_admin = st.tabs(["Symulacja", "Przeglądanie", "Admin"])

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
    else:
        st.subheader("Tryb wywiadu")
        st.info("Najpierw wygeneruj pacjenta, aby móc rozpocząć wywiad.")

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

    # Removed in-tab scenario expander; moved to sidebar below

# --- Main area ---
chat_container = st.container()

if st.session_state.patient_scenario:
    # Render chat history
    with chat_container:
        for message in st.session_state.chat_history:
            role = message.get("role", "assistant")
            with st.chat_message("assistant" if role == "assistant" else "user"):
                st.markdown(message.get("content", ""))

    # Input box
    input_placeholder = (
        "Zadaj pytanie pacjentowi..." if st.session_state.current_mode == "doctor_asks" else
        "Zadaj pytanie lekarzowi..." if st.session_state.current_mode == "patient_asks" else
        "Zadaj pytanie AI..."
    )
    if prompt := st.chat_input(input_placeholder):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

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
    st.info("Aby rozpocząć, wygeneruj nowego pacjenta w zakładce 'Symulacja'.")


