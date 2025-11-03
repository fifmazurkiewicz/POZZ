# POZ Patient Simulator (med-sim-app)

Training application for physicians and medical students that simulates a live patient interview in Primary Care (POZ). The app uses Streamlit for the UI and OpenRouter for LLM responses.

All source code and comments are in English. Use `uv` for dependency and environment management. Locally, secrets load from `.env` when `ENVIRONMENT=local`. In non-local environments (e.g., production on AWS Lightsail), secrets are retrieved from AWS Secrets Manager.

## Tech Stack

- Python 3.11+
- Streamlit
- OpenRouter (Claude, GPT, Llama families)
- uv (package and environment management)
- AWS Lightsail + AWS Secrets Manager

## Project Structure

```
med-sim-app/
├── pyproject.toml
├── .gitignore
├── README.md
├── app.py
└── modules/
    ├── __init__.py
    ├── config.py
    ├── llm_service.py
    ├── prompt_manager.py
    └── audio_processor.py
```

## Environment & Secrets

- Local development: set `ENVIRONMENT=local` and add `OPENROUTER_API_KEY` in your `.env` file.
- Non-local (e.g., prod): set `ENVIRONMENT=prod` (or anything else than `local`). The app will attempt to fetch the OpenRouter key from AWS Secrets Manager using `OPENROUTER_SECRET_NAME` (default: `med-sim/openrouter`). The secret can be either:
  - a raw string (the API key), or
  - a JSON object with field `OPENROUTER_API_KEY`.

Example `.env` (do not commit):

```
ENVIRONMENT=local
OPENROUTER_API_KEY=sk-or-...
AWS_REGION=eu-central-1
OPENROUTER_SECRET_NAME=med-sim/openrouter
DATABASE_URL=postgresql://user:password@localhost:5432/medsim
```

## Install and Run (with uv)

1) Install uv (if needed): see `https://docs.astral.sh/uv/`.

2) Create the environment and install dependencies:

```
uv sync
```

3) Run Streamlit app:

```
uv run streamlit run app.py
```

The app will be available on `http://localhost:8501` by default.

## OpenRouter Configuration

The app uses OpenAI-compatible endpoints via OpenRouter. Set `OPENROUTER_API_KEY` and optionally customize model names in `modules/llm_service.py` and within `app.py`.

## Audio Transcription (Whisper)

- `modules/audio_processor.py` contains a minimal wrapper for sending audio files to OpenRouter Whisper-compatible transcription endpoint.
- For live recording inside Streamlit, you can integrate `streamlit-webrtc` (already listed in dependencies) and connect the recorded file to `transcribe_audio_file`.

## Deployment on AWS Lightsail

1) Provision a Linux instance (e.g., Ubuntu 22.04). Open firewall for TCP 8501 (Streamlit).
2) Install system dependencies: Python 3.11+, git, and `uv`.
3) Clone your repository.
4) Set environment variables for production (do not use `.env` in prod):
   - `ENVIRONMENT=prod`
   - `AWS_REGION=eu-central-1` (or your region)
   - `OPENROUTER_SECRET_NAME=med-sim/openrouter` (or your chosen secret name)
   - `POSTGRES_SECRET_NAME=med-sim/postgres` (secret with database URL)
5) Create a role or instance profile/credentials with permission to read the secret in AWS Secrets Manager. The app uses `boto3` under the hood.
6) Install and run:

```
uv sync
uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

7) Keep the process running:
   - Use a process manager (e.g., `systemd`, `pm2` for python via wrapped command) or a terminal multiplexer (`screen`/`tmux`).

## Future Enhancements

- Performance feedback after interviews (grading, missed threads, diagnosis quality)
- RAG integration with external knowledge sources (e.g., PubMed)
- Dynamic “patient card” updating with ordered labs/imaging
- Live interview with streaming transcription and summarization
- Conversation persistence in Postgres (done): tables `patients`, `conversations`, `messages`

## License

MIT
