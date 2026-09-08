const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type ApiOptions = {
  method?: string;
  body?: unknown;
  token?: string | null;
};

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type PendingApprovalListener = () => void;
let pendingApprovalListener: PendingApprovalListener | null = null;

/** AuthProvider registers this so a mid-session revoke shows the waiting screen. */
export function setPendingApprovalListener(listener: PendingApprovalListener | null) {
  pendingApprovalListener = listener;
}

export async function apiFetch<T>(path: string, options: ApiOptions = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (options.token) {
    headers.Authorization = `Bearer ${options.token}`;
  }
  const res = await fetch(`${API_URL}${path}`, {
    method: options.method ?? "GET",
    headers,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!res.ok) {
    const detail = await res.text();
    const parsed = parseApiError(detail, res.statusText);
    if (parsed.code === "account_pending_approval") {
      pendingApprovalListener?.();
    }
    throw new ApiError(parsed.message, res.status, parsed.code);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

function parseApiError(body: string, fallback: string): { message: string; code?: string } {
  if (!body) return { message: fallback };
  try {
    const json = JSON.parse(body) as {
      code?: string;
      message?: string;
      detail?: string | { code?: string; message?: string } | Array<{ msg?: string }>;
    };
    const detail = json.detail;
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      return {
        message: detail.message || fallback,
        code: detail.code,
      };
    }
    if (typeof json.code === "string") {
      return { message: json.message || fallback, code: json.code };
    }
    if (typeof detail === "string") return { message: detail };
    if (Array.isArray(detail)) {
      return { message: detail.map((item) => item.msg).filter(Boolean).join(", ") || fallback };
    }
  } catch {
    /* plain-text error body */
  }
  return { message: body };
}

export { fetchApiLiveness, fetchApiPulse } from "@/lib/api/pulse";
