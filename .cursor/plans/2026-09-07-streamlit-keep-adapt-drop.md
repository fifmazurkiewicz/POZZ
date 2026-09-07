# Streamlit keep / adapt / drop (reference map)

Not an import checklist. Production code is greenfield FastAPI + Next.js.

| Area | Keep (reimplement) | Adapt | Drop |
|---|---|---|---|
| Patient scenario prompt | Polish RPG-style generator + **Karta pacjenta** block | Move to Langfuse; `TEXT_MODEL` env | Streamlit `st.spinner` |
| First-time 20/80 | Same weights | Persist `is_first_time` on `patients` | Session-only flags |
| Card parser | Label map (imię, wiek, …) | Dedicated module + tests | Fragile 2000-char window as the only parser long-term |
| Simulation modes | doctor / patient / meta | REST `mode` | Streamlit radio + rerun |
| End interview eval | Gold plan + user response + evaluation prompt | Per-user conversation; hide gold until done | Showing plan from session_state |
| Groq Whisper | Default STT | `SpeechToTextProvider` | Temp files on Render disk beyond request scope |
| Recorded interview LLM JSON | roles + summary + extract keys | One-shot MVP | 10s chunk loop in Streamlit; `process_chunk_async` fail-open |
| Manual interview | Typed lines → summary + recommendations | Same | Mixing with recorded conversation_id hacks |
| Browse list | Session detail views | **Per-user RLS** (prototype is global) | Global `list_conversations_with_patient` |
| Admin bulk generate | 1–100 catalog rows | Admin-only + spend cap | Progress bar in Streamlit; wipe UI |
| AWS Secrets Manager | — | — | **Drop** for production; Render/Vercel env |
| cloudflared / start_up.sh | — | Keep files as legacy ops | **Drop** as canonical deploy |
| `diarization_test/` | Ideas for Phase 2 | — | **Drop** from MVP |
| `streamlit-webrtc` / mic-recorder | UX lesson: HTTPS + gesture | Browser MediaRecorder | Python Streamlit components |
| `wipe_all_data` | — | — | **Drop** from product UI |
