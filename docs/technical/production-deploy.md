# Production deploy (step by step)

Operational runbook for **Vercel + Render + Supabase + Cloudflare**. Stack decisions live in [architecture-for-cursor.md](../architecture-for-cursor.md) §8 and [ADR 2026-09-07](./decisions/2026-09-07-stack-vercel-render-supabase.md). Env **names** only: [configuration.md](./configuration.md). Local loop: [local-setup.md](./local-setup.md).

Do **not** deploy Streamlit (`app.py`). Do **not** use AWS Secrets Manager or `cloudflared` as the public URL.

| Layer | Platform | Public URL |
|---|---|---|
| Frontend | Vercel (Next.js, root `frontend/`) | `https://pozz.fmazurkiewicz.dev` |
| Backend | Render Docker (root `backend/`) | `https://api-pozz.fmazurkiewicz.dev` |
| DB + Auth | Supabase (Postgres + Google OAuth + RLS) | `https://<project-ref>.supabase.co` |
| DNS | Cloudflare (registrar + DNS) | `pozz` and `api-pozz` as **separate** CNAMEs |

GitHub: `fifmazurkiewicz/POZZ`, production branch `main`. CI (`.github/workflows/ci.yml`) runs Gitleaks, pytest, frontend lint/test/build — it does **not** deploy.

---

## 0. Before you start

Accounts: GitHub, Supabase, Render, Vercel, Cloudflare (zone `fmazurkiewicz.dev`), plus provider keys you will paste only into dashboards (never into git or chat):

- OpenRouter (`OPENROUTER_API_KEY`) — text LLM
- Groq (`GROQ_API_KEY`) — Whisper STT
- Google (`GOOGLE_API_KEY`) — Gemini Live when `VOICE_MODE=speech_to_speech`
- ElevenLabs (`ELEVENLABS_API_KEY`) — patient TTS
- Langfuse Cloud (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`)

Also have a **Google Cloud** project ready for the OAuth client (step 3).

Confirm locally first (optional but recommended):

```bash
cd backend && python -m pytest
cd frontend && npm run lint && npm test && npm run build
```

---

## 1. Create the Supabase project

1. Open [Supabase Dashboard](https://supabase.com/dashboard) → **New project**.
2. Region: pick one close to Render (Europe if possible). Wait until the project is healthy.
3. **Project Settings → API**: copy **Project URL** (`SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_URL`) and **anon public** key. Copy **service_role** only into Render later — never into Vercel or the browser.
4. **Project Settings → Database**: you need two URLs:
   - **Session pooler** (Supavisor, typically port **5432**) → this is Render `DATABASE_URL`.
   - Do **not** use `db.<ref>.supabase.co` (direct, IPv6-only — Render often cannot reach it).
5. Append `?sslmode=require` if the pooler URI does not already include SSL.

URI shape (placeholder only):

```text
postgresql://postgres.<PROJECT_REF>:<PASSWORD>@aws-0-<REGION>.pooler.supabase.com:5432/postgres?sslmode=require
```

If SQLAlchemy later fails with prepared-statement / pgbouncer errors, switch to the **transaction** pooler (port **6543**) and add `?sslmode=require`. Prefer session mode first for this long-lived Docker app.

---

## 2. Apply the schema (RLS included)

The contract is `supabase/migrations/001_initial.sql` (tables + RLS). The Render API uses SQLAlchemy with `DATABASE_URL` and **bypasses RLS**; RLS still protects any direct Supabase client.

**Option A — Dashboard (simplest)**

1. Supabase → **SQL Editor** → New query.
2. Paste the full contents of `supabase/migrations/001_initial.sql`.
3. Run. Confirm tables exist: `users`, `usage_ledger`, `patients`, `conversations`, `messages`, `interview_transcripts`, `interview_suggestions`, `jobs`, `patient_user_state`.

**Option B — CLI** (if you use the Supabase CLI locally, linked to this project)

```bash
supabase link --project-ref <PROJECT_REF>
supabase db push
```

Do not invent extra tables here. Later migrations go in `supabase/migrations/` and are applied the same way.

---

## 3. Google OAuth (Supabase + Google Cloud)

The app uses **server-side PKCE** at `/auth/callback` (not a client-side exchange on `/`).

### 3a. Google Cloud OAuth client

1. [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → **Credentials** → Create **OAuth client ID** (Web application).
2. Authorized redirect URI (Supabase, not the POZZ domain):

   ```text
   https://<PROJECT_REF>.supabase.co/auth/v1/callback
   ```

3. Copy Client ID and Client Secret.

### 3b. Supabase Auth

1. Authentication → **Sign In / Providers** → enable **Google**. Paste Client ID and Secret.
2. Authentication → **URL Configuration**:
   - **Site URL:** `https://pozz.fmazurkiewicz.dev`
   - **Redirect URLs** (add all):
     - `https://pozz.fmazurkiewicz.dev/auth/callback`
     - `http://localhost:3000/auth/callback` (local only)
3. Leave email/password off unless you later decide otherwise. Product login is Google.

New public signups stay on **AuthGate** until an admin Accepts (`users.is_approved`). The allowlist email in `ALLOWED_ADMIN_EMAILS` is auto-approved on insert.

---

## 4. Deploy the API on Render

1. [Render Dashboard](https://dashboard.render.com/) → **New → Web Service** → connect `fifmazurkiewicz/POZZ`.
2. Settings that must match this repo (easy to get wrong):

   | Setting | Value |
   |---|---|
   | Branch | `main` |
   | **Root Directory** | `backend` |
   | Runtime | **Docker** (not a Python native runtime) |
   | Dockerfile path | `./Dockerfile` (relative to `backend/`) |
   | Instance | Free is OK (30–60 s cold start after ~15 min idle) |
   | Health check path | `/api/health` |
   | Internal port | **8000** (Dockerfile `CMD` binds `0.0.0.0:8000`) |

   Root Directory is `backend`. Do **not** name the root folder `Docker`.

3. **Environment** (Render dashboard). Set production values here — never `DEV_AUTH_ENABLED=true`.

   **Required for a live API + auth**

   | Name | Notes |
   |---|---|
   | `DATABASE_URL` | Supavisor **session** pooler (step 1). |
   | `SUPABASE_URL` | `https://<ref>.supabase.co` — enables JWKS. |
   | `CORS_ORIGINS` | `https://pozz.fmazurkiewicz.dev` (comma-separate extras if needed). |
   | `SPEND_CAP_TZ` | `Europe/Warsaw` |
   | `ALLOWED_ADMIN_EMAILS` | `fifmazurkiewicz@gmail.com` |
   | `TEXT_PROVIDER` | `openrouter` |
   | `TEXT_MODEL` | `google/gemini-2.5-flash-lite` (set explicitly; deprecated slugs 404) |
   | `OPENROUTER_API_KEY` | dashboard only |
   | `VOICE_MODE` | `speech_to_speech` (or `chained` to force lamp off) |

   **Voice / tracing (needed once those packages are used in prod)**

   | Name | Notes |
   |---|---|
   | `GOOGLE_API_KEY` | Gemini Live |
   | `STT_PROVIDER` | `groq` |
   | `GROQ_API_KEY` | Whisper |
   | `TTS_PROVIDER` | `elevenlabs` |
   | `TTS_VOICE_ID` | Polish patient voice |
   | `ELEVENLABS_API_KEY` | |
   | `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | |
   | `LANGFUSE_HOST` | `https://cloud.langfuse.com` |

   **Never set on Render**

   - `DEV_AUTH_ENABLED`
   - `ENVIRONMENT`, `AWS_REGION`, `OPENROUTER_SECRET_NAME`, `POSTGRES_SECRET_NAME`
   - `NEXT_PUBLIC_*` (those belong on Vercel)

4. Deploy. First build pulls `python:3.12-slim` and installs `backend/requirements.txt`.
5. Render will give `https://<service>.onrender.com`. Keep this hostname — Cloudflare CNAME target in step 5.

### 4a. Smoke the API (before DNS)

Wait through cold start if needed, then:

```bash
curl https://<service>.onrender.com/api/health
# {"status":"ok","service":"pozz"}

curl https://<service>.onrender.com/api/health/ready
# {"status":"ok","checks":{"database":"ok"}}

curl https://<service>.onrender.com/api/voice/config
# live_available true when VOICE_MODE=speech_to_speech
```

If liveness is 200 but ready is 503: `DATABASE_URL` is wrong (direct host, bad password, missing SSL, or IPv6). Fix the pooler URI; do not disable the health check.

---

## 5. Cloudflare DNS — API

In Cloudflare → DNS → Records for `fmazurkiewicz.dev`:

| Type | Name | Target | Proxy |
|---|---|---|---|
| CNAME | `api-pozz` | `<service>.onrender.com` | **DNS only** (grey cloud) |

Then in Render → Custom Domain → add `api-pozz.fmazurkiewicz.dev`. Wait for the certificate.

`api-pozz` **must** be DNS-only. Orange-cloud proxy breaks Render TLS issuance.

Smoke:

```bash
curl https://api-pozz.fmazurkiewicz.dev/api/health
```

---

## 6. Deploy the PWA on Vercel

1. [Vercel](https://vercel.com/) → **Add New → Project** → import `fifmazurkiewicz/POZZ`.
2. Settings:

   | Setting | Value |
   |---|---|
   | Framework | Next.js |
   | **Root Directory** | `frontend` |
   | Node | 22 (`frontend/package.json` engines) |
   | Build | `npm run build` (default) |
   | Install | `npm install` / `npm ci` |

3. **Environment variables** (Production; also Preview if you want OAuth on preview URLs):

   | Name | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | `https://api-pozz.fmazurkiewicz.dev` |
   | `NEXT_PUBLIC_SUPABASE_URL` | same Project URL as Render `SUPABASE_URL` |
   | `NEXT_PUBLIC_SUPABASE_ANON_KEY` | anon public key (not service_role) |

   Empty Supabase vars enable local `dev-token` only. In production the client **refuses** `dev-token`. Do not leave these empty on Vercel.

4. Deploy. Note the `*.vercel.app` URL for the next step.

`NEXT_PUBLIC_*` is inlined at **build** time. Changing a public env var requires a **redeploy**.

---

## 7. Cloudflare DNS — frontend + Vercel domain

1. Vercel → Project → **Domains** → add `pozz.fmazurkiewicz.dev`. Vercel shows the CNAME target (typically `cname.vercel-dns.com`).
2. Cloudflare → DNS:

   | Type | Name | Target | Proxy |
   |---|---|---|---|
   | CNAME | `pozz` | `cname.vercel-dns.com` (or the value Vercel shows) | Orange cloud is OK **if** SSL mode is **Full (strict)** |

   Keep `pozz` and `api-pozz` as **two records**. Do not fold them into one.

3. Wait until Vercel domain status is **Valid**.

---

## 8. Cross-wire after both domains are live

1. Render `CORS_ORIGINS` includes `https://pozz.fmazurkiewicz.dev` (no trailing slash). Redeploy or restart the API if you changed it after first deploy.
2. Supabase Site URL + Redirect URLs match step 3b.
3. Confirm Vercel `NEXT_PUBLIC_API_URL` is the custom API host, not `*.onrender.com` (works, but the product URL is `api-pozz`).

---

## 9. Production smoke (end to end)

Expect a **Waking up / Uruchamianie API…** banner for 30–60 s if Render was asleep. That is accepted on Free.

| # | Check | Expected |
|---|---|---|
| 1 | `GET https://api-pozz.fmazurkiewicz.dev/api/health` | `{"status":"ok","service":"pozz"}` |
| 2 | `GET https://api-pozz.fmazurkiewicz.dev/api/health/ready` | 200, database `ok` |
| 3 | Open `https://pozz.fmazurkiewicz.dev` | PWA loads; banner clears when health is 200 |
| 4 | Google Sign in | Redirect → `/auth/callback` → session cookie (not a hash on `/`) |
| 5 | New non-admin Google user | AuthGate waiting screen until Accept |
| 6 | Admin (`ALLOWED_ADMIN_EMAILS`) | Approved; Admin surfaces visible |
| 7 | Simulation lamp | Preference persists (`localStorage` `pozz-sim-live-gemini`); Live only if `VOICE_MODE=speech_to_speech` |
| 8 | Mic on iOS | Only on HTTPS (custom domain). First Listening / record tap is the gesture that unlocks the mic. |

CI green on `main` is the gate before you treat a deploy as done. Frontend `build` alone is not enough.

---

## 10. What this deploy is (and is not)

This runbook ships the **greenfield** app (`backend/` + `frontend/`). Feature completeness follows [build order](../superpowers/specs/2026-09-08-refactor-build-order-design.md): schema/auth are in-repo; later packages add text simulation, chained voice, Live wiring, evaluation, Interview, admin extras.

You can (and should) deploy the scaffold early so ApiPulse, custom domains, and OAuth are proven before those packages.

**Out of scope for this path**

- Redis / a second worker service
- Durable audio blobs
- Deploying `diarization_test/`
- Database wipe from the product UI
- Auto-deploy from GitHub Actions (Vercel + Render git integration is the deploy)

---

## 11. Recurring operations

| Change | Where | Then |
|---|---|---|
| Backend code | push `main` | Render auto-deploys; watch `/api/health` |
| Frontend code | push `main` | Vercel production deploy |
| Secret / private env | Render or Vercel dashboard | Render restarts; Vercel **redeploy** if `NEXT_PUBLIC_*` |
| Schema | new file in `supabase/migrations/` | apply in SQL Editor or `supabase db push` |
| Spend month | already `SPEND_CAP_TZ=Europe/Warsaw` | no extra cron; cap is calendar month in that TZ |
| Rollback | Vercel Instant Rollback / Render previous deploy | keep Supabase schema compatible |

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Render deploy “no Dockerfile” | Root Directory is not `backend`, or Runtime is not Docker |
| Health check failing | Port is not **8000**, or path is `/` instead of `/api/health` |
| Ready 503 | `DATABASE_URL` is direct `db.` host or missing SSL |
| Browser CORS errors | `CORS_ORIGINS` missing `https://pozz.fmazurkiewicz.dev` |
| OAuth returns to `/login?error=oauth` | Redirect URL not listed in Supabase, or empty `NEXT_PUBLIC_SUPABASE_*` on Vercel |
| Infinite OAuth loop | Client-side exchange on `/` — product must use `/auth/callback` |
| `dev-token` in production | Never set `DEV_AUTH_ENABLED` on Render; fill Supabase public vars on Vercel |
| API banner never clears | Wrong `NEXT_PUBLIC_API_URL`, or `api-pozz` proxied (orange cloud) |
| 404 from OpenRouter | `TEXT_MODEL` unset or a deprecated slug |
| Cold start “broken app” | Wait; ApiPulse retries every 5 s while unhealthy |

---

## Decision (2026-09-11)

This file is the **operator SoT** for first production cutover. Hosting choices are unchanged (Vercel / Render Docker / Supabase / Cloudflare CNAMEs). `CORS_ORIGINS` is a required Render variable and is documented in [configuration.md](./configuration.md).
