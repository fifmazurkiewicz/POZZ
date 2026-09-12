-- POZZ privacy/security foundation.
-- Users may edit only non-privileged profile fields through the Supabase client.
-- Approval, admin role and spend cap remain backend/admin-only operations.
REVOKE UPDATE ON TABLE public.users FROM authenticated;
GRANT UPDATE (display_name) ON TABLE public.users TO authenticated;

DROP POLICY IF EXISTS users_update ON public.users;
CREATE POLICY users_update_own_profile ON public.users FOR UPDATE
  USING (id = auth.uid())
  WITH CHECK (id = auth.uid());

-- User-authored and user-seeded cases are private unless explicitly reviewed
-- and published by an administrator through a future catalog workflow.
CREATE INDEX IF NOT EXISTS patients_private_owner_idx
  ON public.patients (created_by)
  WHERE is_private = true;
