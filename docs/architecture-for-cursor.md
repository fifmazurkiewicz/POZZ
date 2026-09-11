# Architecture Specification — for implementation (Cursor)

This is the **business + technical** project description for implementation. UX/UI behavior is fully defined in [`ux/ux-ui-spec.md`](./ux/ux-ui-spec.md) — this document assumes that as given and focuses on product scope, stack, domain, and how to build it. Deeper market/business write-ups (when added) live under [`business/`](./business/).

**Product posture:** public free web app for doctors training primary-care (POZ) interviews (own use + gated access). **Greenfield codebase** — the Streamlit prototype at repo root is [reference only](#2-streamlit-as-reference-no-deploy). Product UI language is **always Polish** (not configurable).

## 1. MVP Scope (build this first, nothing else)

- **Simulation:** load or generate a patient scenario, run a text/voice interview as the doctor, optionally flip who asks (doctor / patient / ask-AI meta), then **End interview** → doctor writes a treatment plan → LLM scores it against a hidden gold plan.
- **Interview (recorded):** record a real doctor–patient conversation (mic), transcribe (STT), assign roles, summarize, extract medications / advice / tests.
- **Sessions:** browse the current user’s past simulations and recorded interviews (preview + detail). Resume is out of MVP (new conversation per Start).
- **Patient catalog:** shared pool of generated scenarios; admin can bulk-generate; doctor can request “next” or generate from keywords.
- Bottom nav: **Simulation / Interview / Menu**.
- Single responsive PWA (web only).
- Google OAuth, admin approval gate, admin panel with **monthly** per-user spend cap — foundational at MVP.
- Observability: **Langfuse** (tracing, cost, prompt management). Pre-deploy prompt regression: **promptfoo**.

**Out of MVP:** diarization GPU pipeline (`diarization_test/`), storing audio blobs, public unauthenticated use, database wipe from the product UI (admin SQL / explicit later spec only), chunked live suggestions on recorded Interview.

Voice **is** in MVP: patient TTS + Gemini Live switcher on Simulation (spec: [`superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md`](./superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md)). Build order: [`superpowers/specs/2026-09-08-refactor-build-order-design.md`](./superpowers/specs/2026-09-08-refactor-build-order-design.md).

## 2. Streamlit as reference (no deploy)

POZZ production is **greenfield**: own `backend/` + `frontend/` built from this document and UX specs. The existing Streamlit app (`app.py`, `modules/`, `start_up.sh`) is **reference only** — study prompts, schema, and flows; **do not** deploy Streamlit to Vercel, Render, or Cloudflare quick tunnel as the product. ADR: [`technical/decisions/2026-09-07-greenfield-no-streamlit-deploy.md`](./technical/decisions/2026-09-07-greenfield-no-streamlit-deploy.md). Keep/adapt/drop map: `.cursor/plans/2026-09-07-streamlit-keep-adapt-drop.md`.

### 2.1 Patterns to reimplement (informed by the prototype)

| Pattern | POZZ implementation |
|---|---|
| Patient scenario generator (Polish, card + hidden traps) | Langfuse prompt + OpenRouter; persist on `patients` |
| First-time vs returning patient (~20/80) | Card shows name+age only when `has_history_here = false` |
| Hidden gold treatment plan | Generated at patient create; revealed after evaluation |
| Simulation modes doctor / patient / meta | Same three roles; REST chat turns |
| End interview → user plan vs gold evaluation | REST + persist on `conversations` |
| Mic → Groq Whisper → LLM | Chained STT; no durable audio |
| Recorded interview: roles + summary + extract | Jobs in Postgres; JSON extract `leki` / `zalecenia` / `badania` |
| Browse past interviews | Per-user session list (prototype was global — **change**) |
| Admin bulk generate | Admin-only; catalog not per-user |

### 2.2 POZZ-native (spec-driven)

| Element | Why greenfield |
|---|---|
| Per-user conversations + RLS | Prototype has no auth |
| Approval gate + spend_cap | Public free app governance |
| PWA Classical UI (Simulation / Interview / Menu) | Streamlit is not the product chrome |
| Langfuse + promptfoo | Ops and prompt quality |
| Supabase auth (Google OAuth) + RLS | Deployment standard |
| ApiPulse / Render cold start | Required on Free plan |

### 2.3 License

POZZ codebase is **not** a Langy fork. Langy is the **process + stack template** (docs, Cursor rules, hosting). Product license stays unset until the first public release.

## 3. Stack

| Layer | Choice | Notes |
|---|---|---|
| Backend | **FastAPI** (greenfield) | Python 3.12; Render Docker |
| Backend hosting | **Render** Free (Docker) | Cold start OK; UI “Waking up…”; `api-pozz.fmazurkiewicz.dev` |
| Frontend | **Next.js** PWA (greenfield) | Classical DS; `pozz.fmazurkiewicz.dev` |
| Frontend hosting | **Vercel** | |
| Database + Auth | **Supabase** (Postgres, Google OAuth, RLS) | catalog + conversations + ledger + job queue |
| Text LLM | OpenRouter adapter | Default `google/gemini-2.5-flash-lite` |
| STT | Groq Whisper, fallback OpenAI/OpenRouter | Chained path when lamp OFF |
| TTS | ElevenLabs (product) or browser | Patient replies spoken on chained path |
| Live | Gemini Live via ephemeral token | Lamp ON; `VOICE_MODE=speech_to_speech` enables |
| AI observability | **Langfuse Cloud** | Runtime SoT for prompts |
| Prompt regression | **promptfoo** | Fixtures in repo; gate before deploy |
| Async jobs | **Postgres** job table / polling | **No Redis in MVP** |
| Admin bootstrap | `ALLOWED_ADMIN_EMAILS` | Sole admin: `fifmazurkiewicz@gmail.com` |

## 4. Provider Abstraction Layer

Requirement: switch providers without rewriting product logic. Config through env.

### 4.1 `TextCompletionProvider`

`complete(messages, response_model) -> T`. Default: OpenRouter (`TEXT_PROVIDER`, `TEXT_MODEL`).

### 4.2 `SpeechToTextProvider`

`transcribe(audio_bytes, language="pl") -> str`. Default Groq Whisper. Used for simulation mic turns and recorded-interview audio.

### 4.3 Voice conversation (MVP)

Two paths, one Simulation lamp (Langy Chat metaphor; not a Langy fork):

- **Lamp ON (`VOICE_MODE=speech_to_speech`):** Gemini Live. Browser ↔ Live after `GET /api/voice/live-token`. Render = auth, scenario agenda, spend, transcript persist — not the media proxy.
- **Lamp OFF (chained):** doctor mic → `POST /api/stt` (or typed text) → LLM turn → patient **TTS** (`TTS_PROVIDER=elevenlabs|browser`). Never opens Live.
- Env `VOICE_MODE=chained` forces the lamp off (`live_available=false` on `GET /api/voice/config`).
- Preference: `localStorage` `pozz-sim-live-gemini` (default ON). Text input always present. Listening optional. Mute Listening while the patient speaks.
- Spec: [`superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md`](./superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md).

### 4.4 Langfuse + promptfoo

- **Langfuse Cloud** = runtime source of truth for prompts (patient generate, simulation roles, gold plan, evaluation, interview roles, summary, extract).
- Repo keeps **promptfoo fixtures** (and optional exported snapshots) for CI; failures block deploy once backend exists.
- Every GenAI / ASR usage that we meter also feeds Langfuse traces + `usage_ledger`.

### 4.5 Start recommendation

Ship **text simulation first** (Package 2), then chained STT+TTS (Package 3), then Live (Package 4). The PWA shell includes the lamp from Package 0 so chrome does not get retrofitted.

## 5. Domain model summary

**Bounded contexts:**

- **Patient Catalog** — generated scenarios, card fields, hidden gold plan, first-time flag
- **Simulation Conversation** — modes, turns, end-interview evaluation
- **Recorded Interview** — audio→STT chunks or whole file, role assignment, summary, medical extract
- **Sessions** — per-user history of both kinds
- **Spend Governance** — monthly cap, ledger, admin
- **Access** — Google OAuth, approval gate

**Key invariant:** a `patients` row is a reusable scenario. A `conversations` row is one user’s attempt (simulation or recorded) and always has `user_id`.

**User journey SoT:** [`superpowers/specs/2026-09-07-domain-user-journey-design.md`](./superpowers/specs/2026-09-07-domain-user-journey-design.md).

**Key events (summary):**

`SymulacjaRozpoczęta` → turns → **Zakończ wywiad** → `PlanLeczeniaWprowadzony` → `OcenaWystawiona`.

Parallel: `WywiadNagrany` → STT job → `TranskryptZRolami` → `PodsumowanieIEkstrakcja`.

## 6. Data model (authoritative)

```sql
users (
  id                    uuid primary key,  -- = Supabase auth.uid()
  email                 text,
  display_name          text,
  is_admin              boolean default false,
  is_approved           boolean default false,
  spend_cap_usd         numeric default 10, -- monthly USD; default $10; admin-editable
  onboarding_completed_at timestamp
);

usage_ledger (
  id                    uuid primary key,
  user_id               uuid references users,
  action_type           text,              -- 'asr' | 'gen_ai' | later 'tts'
  cost_usd              numeric,
  provider              text nullable,
  langfuse_trace_id     text nullable,
  created_at            timestamp
);

patients (
  id                    uuid primary key,
  created_at            timestamptz not null default now(),
  created_by            uuid references users nullable, -- admin / generator
  scenario              text not null,     -- full hidden scenario (Polish markdown)
  summary               text,              -- short listing line (name-ish)
  treatment_plan        text,              -- hidden gold plan
  keywords              text nullable,
  is_first_time         boolean default false
  -- card fields are parsed from scenario; optional generated columns later
);

conversations (
  id                    uuid primary key,
  user_id               uuid references users not null,
  patient_id            uuid references patients not null,
  created_at            timestamptz not null default now(),
  ended_at              timestamptz nullable,
  kind                  text not null,     -- 'simulation' | 'recorded' | 'manual'
  title                 text,
  mode                  text,              -- simulation: 'doctor_asks' | 'patient_asks' | 'meta_ask'
  user_treatment_response text,
  diagnosis_evaluation  text,
  interview_summary     text,
  extracted_info        jsonb,             -- {leki, zalecenia, badania, ocena_propozycji_lekarza, sugestie_poprawek}
  audio_ref             text nullable      -- MVP: always null
);

messages (
  id                    bigserial primary key,
  conversation_id       uuid references conversations on delete cascade,
  role                  varchar(16) not null, -- 'user' | 'assistant' | 'system'
  content               text not null,
  created_at            timestamptz not null default now()
);

interview_transcripts (
  id                    uuid primary key,
  conversation_id       uuid references conversations on delete cascade,
  chunk_number          int not null,
  transcript_json       jsonb not null,    -- {transcript: [{role, text, timestamp}]}
  created_at            timestamptz not null default now()
);

interview_suggestions (
  id                    uuid primary key,
  conversation_id       uuid references conversations on delete cascade,
  chunk_number          int not null,
  minute_number         int not null,
  suggestions           text not null,
  created_at            timestamptz not null default now()
);

jobs (
  id                    uuid primary key,
  user_id               uuid references users nullable,
  kind                  text not null,     -- 'generate_patient' | 'transcribe_interview' | 'evaluate_plan'
  payload               jsonb,
  status                text not null,     -- 'queued' | 'running' | 'done' | 'failed'
  error                 text,
  created_at            timestamptz default now(),
  updated_at            timestamptz default now()
);

-- Optional: per-user skip / done against catalog
patient_user_state (
  user_id               uuid references users,
  patient_id            uuid references patients,
  status                text not null,     -- 'skipped' | 'completed'
  updated_at            timestamptz default now(),
  primary key (user_id, patient_id)
);
```

RLS: `auth.uid() = user_id` on user-owned tables. `patients` readable by approved users; insert/update/delete admin or service role. `jobs` admin/service. Render backend uses SQLAlchemy with the DB URL (bypasses RLS).

## 7. Key application logic

### 7.1 Next patient / generate

- **Next:** oldest catalog patient this user has not `completed` or `skipped` (see `patient_user_state`). If none, enqueue `generate_patient` (no keywords).
- **Keywords:** always generate a new `patients` row (shared catalog) then open a conversation for this user.
- On load: parse **Karta pacjenta** from `scenario` (same fields as the prototype). Do **not** show the gold `treatment_plan` until after evaluation.
- First-time: name + age + “Historia w punkcie: Nie”; other card fields gathered in interview.

### 7.2 Simulation turn

- Client sends text (typed or STT). Backend builds prompt from `role_to_play` + scenario + history (Langfuse).
- Persist user + assistant `messages`.
- Modes: doctor asks (LLM = patient), patient asks (LLM = doctor), meta (LLM = clinical tutor; speaker label “AI”).
- Reset interview: new `conversations` row for the same patient; old row remains in Sessions.

### 7.3 End interview + evaluation

- Requires at least one user turn.
- If `patients.treatment_plan` empty, generate once and persist on the patient.
- Prompt the doctor for medications / advice / tests (text or STT).
- Evaluation prompt compares user response vs gold + scenario + history. Persist `user_treatment_response` + `diagnosis_evaluation`. Mark `patient_user_state.status = completed`.
- Reveal gold plan in an expander after evaluation. Retry allowed (re-enter waiting state; does not create a new conversation unless the user resets).

### 7.4 Recorded interview

- Start creates `conversations.kind = recorded` (may attach a placeholder patient “Wywiad z rzeczywistym pacjentem”).
- Browser records WAV; upload to `POST /api/interviews/{id}/audio` (multipart). Backend STT + one LLM call (`process_interview_transcript`) → transcripts + summary + extract. No audio persisted (`audio_ref` null); process then discard bytes.
- Chunked live suggestions (prototype 10s chunks) are **Phase 2**. MVP = one shot after Stop.
- Manual interview constructor (typed doctor/patient lines → summary + recommendations) is **in MVP** as Interview → Manual (matches prototype tab).

### 7.5 Sessions list

- Current user only. Labels: patient `summary` for simulations; “Nagrany wywiad” / “Wywiad ręczny” for others; skipped marker from `patient_user_state`.
- Detail: scenario (not for recorded), messages, user plan, evaluation, transcript, extract.

### 7.6 Monthly spend cap

- Cap is **per calendar month**, TZ **Europe/Warsaw** (`SPEND_CAP_TZ`).
- Default for new users: **`spend_cap_usd = 10`**.
- `users.is_approved` (default false). New public signups wait for admin Accept before feature APIs; Accept does not overwrite `spend_cap_usd`. Spec: [`superpowers/specs/2026-09-07-user-approval-gate-design.md`](./superpowers/specs/2026-09-07-user-approval-gate-design.md).
- Sum `usage_ledger.cost_usd` for current month vs cap. Counted: **TTS + ASR + GenAI**.
- Admin can edit cap anytime.
- **At cap:** block generate patient, simulation turns, STT, TTS, Live, evaluation, recorded processing. User **may still** browse Sessions and read existing evaluations. Nothing deleted. Resets next calendar month.

### 7.7 Approval gate

- `GET /api/auth/me` ungated (returns `is_approved`).
- Feature routes use `require_approved`.
- Waiting UI: Polish copy; poll `/api/auth/me` every 15s.
- Admin: list users + `PATCH is_approved`. Hide self-revoke. Allowlist emails auto-approved on insert. Local `dev-token` auto-approved. No email notifications in v1.

### 7.8 Admin

- Bulk generate N patients (1–100) into the catalog (GenAI → cap of the **admin** user).
- Approval queue + spend cap editor.
- **No** product-UI “wipe database”. Prototype wipe stays legacy-only.

## 8. Deployment

Operator runbook (click-by-click): [`technical/production-deploy.md`](./technical/production-deploy.md).

- Frontend: Vercel → **`pozz.fmazurkiewicz.dev`**.
- Backend: Render Docker Free → **`api-pozz.fmazurkiewicz.dev`**. Expect spin-down; UI shows waking state; never rely on in-memory jobs.
- Supabase: Postgres + Google OAuth + RLS; pooler URL on Render.
- Env: providers, Langfuse, `SPEND_CAP_TZ`, `ALLOWED_ADMIN_EMAILS`, `VOICE_MODE`, `CORS_ORIGINS`.
- No Redis service required for MVP.
- Cloudflare: `pozz` CNAME to Vercel; `api-pozz` CNAME to Render DNS-only — separate records.
- Google OAuth callback `/auth/callback` with **server-side PKCE** (not client-side on `/`).

## 9. Build order (MVP)

SoT: [`superpowers/specs/2026-09-08-refactor-build-order-design.md`](./superpowers/specs/2026-09-08-refactor-build-order-design.md).

0. Scaffold: FastAPI health + voice config, Next.js PWA tabs, ApiPulse, Live/TTS lamp chrome.
1. Schema + auth + AuthGate.
2. Simulation **text** (next patient, card, modes, turns).
3. Chained STT + patient TTS; mute Listening during TTS.
4. Gemini Live + lamp wiring (token, disconnect on lamp OFF).
5. End interview evaluation.
6. Recorded + manual Interview tab.
7. Sessions + Admin (approval, cap, bulk generate).
8. promptfoo suites in CI.

## 10. Known risks

- iOS Safari mic — test early; HTTPS required (Vercel/Render terminate TLS).
- Medical content in LLM output — prompts must stay in-character; evaluation is training feedback, not clinical advice. UI must say the app is a **simulator**, not a medical device.
- PHI in recorded interviews — RLS + approval gate; no audio retention in MVP; no training on user audio with third parties beyond STT/LLM providers named in env.
- Render cold start — ApiPulse mandatory.
- Prototype global catalog “processed” semantics must not leak across users.
