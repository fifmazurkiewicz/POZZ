/** True while Supabase is still exchanging the OAuth PKCE code in the URL. */
export function hasPendingOAuthRedirect(): boolean {
  if (typeof window === "undefined") return false;
  const url = new URL(window.location.href);
  return url.searchParams.has("code") || url.hash.includes("access_token=");
}

/** Remove OAuth callback params so a failed exchange does not loop on reload. */
export function clearOAuthRedirectParams(): void {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  if (!url.searchParams.has("code") && !url.hash.includes("access_token=")) return;
  url.searchParams.delete("code");
  url.searchParams.delete("state");
  url.hash = "";
  window.history.replaceState({}, "", `${url.pathname}${url.search}`);
}
