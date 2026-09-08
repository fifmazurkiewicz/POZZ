import { apiFetch } from "@/lib/api";
import type { PatientCard } from "@/lib/simulation/cardDisplay";
import type { SimMessage, SimMode } from "@/lib/simulation/types";

export type { SimMessage, SimMode };

export type SimulationSession = {
  conversation_id: string;
  patient_id: string;
  kind: string;
  mode: SimMode;
  title: string | null;
  card: PatientCard;
  messages?: SimMessage[];
  assistant?: SimMessage;
};

export async function fetchNextPatient(token: string, keywords?: string): Promise<SimulationSession> {
  return apiFetch<SimulationSession>("/api/patients/next", {
    method: "POST",
    token,
    body: keywords ? { keywords } : {},
  });
}

export async function fetchConversation(token: string, conversationId: string): Promise<SimulationSession> {
  return apiFetch<SimulationSession>(`/api/conversations/${conversationId}`, { token });
}

export async function postTurn(
  token: string,
  conversationId: string,
  content: string,
  mode?: SimMode
): Promise<SimulationSession> {
  return apiFetch<SimulationSession>(`/api/conversations/${conversationId}/turns`, {
    method: "POST",
    token,
    body: { content, mode },
  });
}

export { composerPlaceholder, speakerLabel } from "@/lib/simulation/labels";
