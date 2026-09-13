import { ApiError, apiUrl } from "@/lib/api";
import type { SimulationSession } from "@/lib/simulation/api";

export async function uploadRecordedInterview(
  token: string,
  audio: File,
  title?: string,
  signal?: AbortSignal
): Promise<SimulationSession> {
  const form = new FormData();
  form.append("audio", audio, audio.name);
  if (title?.trim()) form.append("title", title.trim());
  const response = await fetch(apiUrl("/api/interviews/recordings"), {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
    signal,
  });
  if (!response.ok) {
    const text = await response.text();
    let message = text || response.statusText;
    let code: string | undefined;
    try {
      const body = JSON.parse(text) as {
        detail?: string | { code?: string; message?: string };
      };
      if (typeof body.detail === "string") message = body.detail;
      else if (body.detail) {
        message = body.detail.message || message;
        code = body.detail.code;
      }
    } catch {
      // Keep the plain response body.
    }
    throw new ApiError(message, response.status, code);
  }
  return response.json() as Promise<SimulationSession>;
}
