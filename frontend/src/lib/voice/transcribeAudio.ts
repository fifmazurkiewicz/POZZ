import { ApiError, apiUrl } from "../api";

type ApiErrorBody = {
  code?: string;
  message?: string;
  detail?: string | { code?: string; message?: string };
};

export async function transcribeAudio(
  token: string,
  blob: Blob,
  signal?: AbortSignal
): Promise<{ text: string }> {
  const form = new FormData();
  form.append("audio", blob, "doctor-turn.webm");

  const response = await fetch(apiUrl("/api/voice/transcribe"), {
    method: "POST",
    signal,
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });

  if (!response.ok) {
    const detail = await response.text();
    const parsed = parseApiError(detail, response.statusText);
    throw new ApiError(parsed.message, response.status, parsed.code);
  }

  const body = (await response.json()) as { text?: string };
  return { text: body.text ?? "" };
}

function parseApiError(body: string, fallback: string): { message: string; code?: string } {
  if (!body) return { message: fallback };
  try {
    const json = JSON.parse(body) as ApiErrorBody;
    const detail = json.detail;
    if (detail && typeof detail === "object") {
      return { message: detail.message || fallback, code: detail.code };
    }
    if (typeof json.code === "string") {
      return { message: json.message || fallback, code: json.code };
    }
    if (typeof detail === "string") return { message: detail };
  } catch {
    /* plain-text error body */
  }
  return { message: body };
}