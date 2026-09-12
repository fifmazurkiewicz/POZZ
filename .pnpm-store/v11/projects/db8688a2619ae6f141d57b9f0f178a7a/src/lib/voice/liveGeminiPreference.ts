export const LIVE_GEMINI_STORAGE_KEY = "pozz-sim-live-gemini";

/** Default ON — matches production speech_to_speech. SSR-safe. */
export function readLiveGeminiPreference(): boolean {
  if (typeof window === "undefined") return true;
  const raw = window.localStorage.getItem(LIVE_GEMINI_STORAGE_KEY);
  if (raw === null) return true;
  return raw !== "false";
}

export function writeLiveGeminiPreference(on: boolean): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LIVE_GEMINI_STORAGE_KEY, String(on));
}
