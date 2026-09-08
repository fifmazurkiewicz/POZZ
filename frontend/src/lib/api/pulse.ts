const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type PulseStatus = "unknown" | "checking" | "healthy" | "unhealthy";

export const PULSE_LIVENESS_PATH = "/api/health";
export const PULSE_TIMEOUT_MS = 8_000;
export const PULSE_INTERVAL_FAST_MS = 5_000;
export const PULSE_INTERVAL_SLOW_MS = 30_000;

async function fetchWithTimeout(path: string): Promise<Response | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), PULSE_TIMEOUT_MS);
  try {
    return await fetch(`${API_URL}${path}`, { cache: "no-store", signal: controller.signal });
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

export async function fetchApiLiveness(): Promise<boolean> {
  const res = await fetchWithTimeout(PULSE_LIVENESS_PATH);
  return res?.ok ?? false;
}

export async function fetchApiPulse(): Promise<boolean> {
  return fetchApiLiveness();
}
