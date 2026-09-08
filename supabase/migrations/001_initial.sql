-- POZZ initial schema (architecture §6). Idempotent-friendly for local re-runs.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY,
  email text,
  display_name text,
  is_admin boolean NOT NULL DEFAULT false,
  is_approved boolean NOT NULL DEFAULT false,
  spend_cap_usd numeric NOT NULL DEFAULT 10,
  onboarding_completed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS usage_ledger (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users (id),
  action_type text NOT NULL,
  cost_usd numeric NOT NULL,
  provider text,
  langfuse_trace_id text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS patients (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  created_by uuid REFERENCES users (id),
  scenario text NOT NULL,
  summary text,
  treatment_plan text,
  keywords text,
  is_first_time boolean NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS conversations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users (id),
  patient_id uuid NOT NULL REFERENCES patients (id),
  created_at timestamptz NOT NULL DEFAULT now(),
  ended_at timestamptz,
  kind text NOT NULL,
  title text,
  mode text,
  user_treatment_response text,
  diagnosis_evaluation text,
  interview_summary text,
  extracted_info jsonb,
  audio_ref text
);

CREATE TABLE IF NOT EXISTS messages (
  id bigserial PRIMARY KEY,
  conversation_id uuid NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
  role varchar(16) NOT NULL,
  content text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS interview_transcripts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id uuid NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
  chunk_number int NOT NULL,
  transcript_json jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS interview_suggestions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id uuid NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
  chunk_number int NOT NULL,
  minute_number int NOT NULL,
  suggestions text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES users (id),
  kind text NOT NULL,
  payload jsonb,
  status text NOT NULL DEFAULT 'queued',
  error text,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS patient_user_state (
  user_id uuid NOT NULL REFERENCES users (id),
  patient_id uuid NOT NULL REFERENCES patients (id),
  status text NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, patient_id)
);

CREATE OR REPLACE FUNCTION public.is_app_admin()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM users WHERE id = auth.uid() AND is_admin = TRUE
  );
$$;

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS users_select ON users;
CREATE POLICY users_select ON users FOR SELECT
  USING (id = auth.uid() OR is_app_admin());
DROP POLICY IF EXISTS users_update ON users;
CREATE POLICY users_update ON users FOR UPDATE
  USING (id = auth.uid() OR is_app_admin());

ALTER TABLE usage_ledger ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS ledger_select ON usage_ledger;
CREATE POLICY ledger_select ON usage_ledger FOR SELECT
  USING (user_id = auth.uid() OR is_app_admin());
DROP POLICY IF EXISTS ledger_insert ON usage_ledger;
CREATE POLICY ledger_insert ON usage_ledger FOR INSERT
  WITH CHECK (user_id = auth.uid() OR is_app_admin());

ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS conversations_all ON conversations;
CREATE POLICY conversations_all ON conversations FOR ALL
  USING (user_id = auth.uid() OR is_app_admin())
  WITH CHECK (user_id = auth.uid() OR is_app_admin());

ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS messages_all ON messages;
CREATE POLICY messages_all ON messages FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM conversations c
      WHERE c.id = messages.conversation_id
        AND (c.user_id = auth.uid() OR is_app_admin())
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM conversations c
      WHERE c.id = messages.conversation_id
        AND (c.user_id = auth.uid() OR is_app_admin())
    )
  );

ALTER TABLE interview_transcripts ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS interview_transcripts_all ON interview_transcripts;
CREATE POLICY interview_transcripts_all ON interview_transcripts FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM conversations c
      WHERE c.id = interview_transcripts.conversation_id
        AND (c.user_id = auth.uid() OR is_app_admin())
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM conversations c
      WHERE c.id = interview_transcripts.conversation_id
        AND (c.user_id = auth.uid() OR is_app_admin())
    )
  );

ALTER TABLE interview_suggestions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS interview_suggestions_all ON interview_suggestions;
CREATE POLICY interview_suggestions_all ON interview_suggestions FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM conversations c
      WHERE c.id = interview_suggestions.conversation_id
        AND (c.user_id = auth.uid() OR is_app_admin())
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM conversations c
      WHERE c.id = interview_suggestions.conversation_id
        AND (c.user_id = auth.uid() OR is_app_admin())
    )
  );

ALTER TABLE patient_user_state ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS patient_user_state_all ON patient_user_state;
CREATE POLICY patient_user_state_all ON patient_user_state FOR ALL
  USING (user_id = auth.uid() OR is_app_admin())
  WITH CHECK (user_id = auth.uid() OR is_app_admin());

ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS patients_select ON patients;
CREATE POLICY patients_select ON patients FOR SELECT
  USING (
    is_app_admin()
    OR EXISTS (SELECT 1 FROM users u WHERE u.id = auth.uid() AND u.is_approved = TRUE)
  );
DROP POLICY IF EXISTS patients_write ON patients;
CREATE POLICY patients_write ON patients FOR INSERT
  WITH CHECK (is_app_admin());
DROP POLICY IF EXISTS patients_update ON patients;
CREATE POLICY patients_update ON patients FOR UPDATE
  USING (is_app_admin());
DROP POLICY IF EXISTS patients_delete ON patients;
CREATE POLICY patients_delete ON patients FOR DELETE
  USING (is_app_admin());

ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS jobs_admin ON jobs;
CREATE POLICY jobs_admin ON jobs FOR ALL
  USING (is_app_admin())
  WITH CHECK (is_app_admin());
