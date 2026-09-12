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
  ended_at?: string | null;
  user_treatment_response?: string | null;
  diagnosis_evaluation?: string | null;
};

export async function fetchNextPatient(token: string, keywords?: string, signal?: AbortSignal): Promise<SimulationSession> {
  return apiFetch<SimulationSession>("/api/patients/next", {
    method: "POST",
    token,
    signal,
    body: keywords ? { keywords } : {},
  });
}

export async function fetchConversation(token: string, conversationId: string, signal?: AbortSignal): Promise<SimulationSession> {
  return apiFetch<SimulationSession>(`/api/conversations/${conversationId}`, { token, signal });
}

export async function postTurn(
  token: string,
  conversationId: string,
  content: string,
  mode?: SimMode,
  signal?: AbortSignal
): Promise<SimulationSession> {
  return apiFetch<SimulationSession>(`/api/conversations/${conversationId}/turns`, {
    method: "POST",
    token,
    signal,
    body: { content, mode },
  });
}

export function requestExamination(token: string, id: string, examination: string, signal?: AbortSignal) {
  return apiFetch<SimulationSession>(`/api/conversations/${id}/examinations`, {
    method: "POST", token, signal, body: { examination },
  });
}

export function finishInterview(token: string, id: string, treatmentPlan: string, signal?: AbortSignal) {
  return apiFetch<SimulationSession>(`/api/conversations/${id}/finish`, {
    method: "POST", token, signal, body: { treatment_plan: treatmentPlan },
  });
}

export { composerPlaceholder, speakerLabel } from "@/lib/simulation/labels";
