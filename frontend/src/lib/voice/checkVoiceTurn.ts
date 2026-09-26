import { apiUrl } from "@/lib/api";

type Decision = { decision: "complete" | "continue"; text: string };

export async function checkVoiceTurn(audio: Blob, token: string, signal?: AbortSignal): Promise<Decision> {
  const form = new FormData();
  form.append("audio", audio, "doctor-turn.webm");
  const response = await fetch(apiUrl("/api/voice/turn-check"), {
    method: "POST", headers: { Authorization: `Bearer ${token}` }, body: form, signal,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body?.detail?.message || body?.message || "Nie udało się rozpoznać końca wypowiedzi. Powiedz ją ponownie.");
  if (!body?.text || (body.decision !== "complete" && body.decision !== "continue")) {
    throw new Error("Nie udało się rozpoznać końca wypowiedzi. Powiedz ją ponownie.");
  }
  return body as Decision;
}
