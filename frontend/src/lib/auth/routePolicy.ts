import type { AuthStatus } from "@/components/AuthProvider";

export const LOGIN_ROUTE = "/login";
export const SIMULATION_ROUTE = "/simulation";
export const APPROVAL_POLL_MS = 15_000;

const ENTRY_ROUTES = new Set<string>([LOGIN_ROUTE, "/"]);

/**
 * Where this visitor belongs. Unknown states return null so routing never guesses.
 */
export function resolveRedirect(status: AuthStatus, pathname: string): string | null {
  switch (status) {
    case "initializing":
    case "profile_unknown":
    case "pending_approval":
      return null;
    case "anonymous":
      return pathname === LOGIN_ROUTE ? null : LOGIN_ROUTE;
    case "ready":
      return ENTRY_ROUTES.has(pathname) ? SIMULATION_ROUTE : null;
  }
}
