import type { SimMessage, SimMode } from "./types";

export function speakerLabel(mode: SimMode, role: SimMessage["role"]): string {
  if (role === "system") return "System";
  if (mode === "patient_asks") {
    return role === "user" ? "Pacjent" : "Lekarz";
  }
  if (mode === "meta_ask") {
    return role === "user" ? "Lekarz" : "AI";
  }
  return role === "user" ? "Lekarz" : "Pacjent";
}

export function composerPlaceholder(mode: SimMode): string {
  if (mode === "patient_asks") return "Zadaj pytanie lekarzowi…";
  if (mode === "meta_ask") return "Zadaj pytanie AI…";
  return "Zadaj pytanie pacjentowi…";
}
