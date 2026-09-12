export const VOICE_ID_STORAGE_KEY = "pozz-tts-voice-id";

const VOICE_ID_PATTERN = /^[A-Za-z0-9_-]{1,128}$/;

export function isValidVoiceId(value: string): boolean {
  const trimmed = value.trim();
  return trimmed === "" || VOICE_ID_PATTERN.test(trimmed);
}

export function readVoiceId(): string {
  try {
    if (typeof window === "undefined") return "";
    const value = window.localStorage.getItem(VOICE_ID_STORAGE_KEY)?.trim() ?? "";
    return isValidVoiceId(value) ? value : "";
  } catch {
    return "";
  }
}

export function saveVoiceId(value: string): void {
  const trimmed = value.trim();
  if (!isValidVoiceId(trimmed)) {
    throw new Error("Invalid voice ID.");
  }
  if (typeof window === "undefined") return;
  if (trimmed) window.localStorage.setItem(VOICE_ID_STORAGE_KEY, trimmed);
  else window.localStorage.removeItem(VOICE_ID_STORAGE_KEY);
}
