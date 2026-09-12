import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";

export function createClient(): SupabaseClient | null {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) {
    return null;
  }
  return createBrowserClient(url, key, {
    auth: {
      flowType: "pkce",
      detectSessionInUrl: false,
      persistSession: true,
      autoRefreshToken: true,
    },
  });
}

export function isDevAuthMode(): boolean {
  if (process.env.NODE_ENV === "production") return false;
  return !process.env.NEXT_PUBLIC_SUPABASE_URL || !process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
}

export const DEV_USER_ID = "00000000-0000-4000-8000-000000000001";
export const DEV_TOKEN = "dev-token";
