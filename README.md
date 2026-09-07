# POZZ

PWA do treningu wywiadu lekarskiego w POZ: symulowany pacjent (tekst + głos lekarza) oraz nagranie prawdziwego wywiadu z transkrypcją i oceną. **Greenfield** — obecny Streamlit (`app.py`) jest tylko referencją (bez deployu na Vercel/Render).

## Dokumentacja

Zacznij od [`docs/README.md`](docs/README.md).

- **Architektura (biznes + tech):** [`docs/architecture-for-cursor.md`](docs/architecture-for-cursor.md)
- **UX/UI:** [`docs/ux/`](docs/ux/)

## Stack

Vercel (frontend) · Render (backend) · Supabase (DB + Auth) — szczegóły w architekturze i `AGENTS.md`.

## Legacy prototype

Lokalny Streamlit (nie produkcja):

```bash
cp .env.example .env   # DATABASE_URL, OPENROUTER_API_KEY, optional GROQ_API_KEY
uv sync
uv run streamlit run app.py --server.port 8501
```
