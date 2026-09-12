-- Migration 003 may already be applied. Defensively remove any inherited or
-- historical anonymous UPDATE privilege without rewriting migration history.
REVOKE UPDATE ON TABLE public.users FROM anon;
