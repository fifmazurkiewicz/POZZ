-- User-authored manual cases must never enter the shared patient catalogue.
ALTER TABLE patients ADD COLUMN IF NOT EXISTS is_private boolean NOT NULL DEFAULT false;

ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS patients_select ON patients;
CREATE POLICY patients_select ON patients FOR SELECT
  USING ((NOT is_private) OR created_by = auth.uid() OR is_app_admin());
