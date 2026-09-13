import { ApiError, apiUrl } from "@/lib/api";
import type { SimulationSession } from "@/lib/simulation/api";

/** Format the current time as a sortable filename stem (Europe/Warsaw). */
function defaultAudioName(): string {
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `nagranie-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.webm`;
}

export async function uploadRecordedInterview(
  token: string,
  audio: Blob,
  title?: string,
  signal?: AbortSignal
): Promise<SimulationSession> {
  const form = new FormData();
  const filename = "name" in audio && typeof audio.name === "string" && audio.name
    ? audio.name
    : defaultAudioName();
  form.append("audio", audio, filename);
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
