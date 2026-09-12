import { afterEach, describe, expect, it, vi } from "vitest";

import { isValidVoiceId, readVoiceId, saveVoiceId, VOICE_ID_STORAGE_KEY } from "./voicePreference";

afterEach(() => vi.unstubAllGlobals());

describe("voicePreference", () => {
  it("validates and trims voice IDs", () => {
    expect(isValidVoiceId("")).toBe(true);
    expect(isValidVoiceId(" voice_2-A ")).toBe(true);
    expect(isValidVoiceId("voice id")).toBe(false);
    expect(isValidVoiceId("x".repeat(129))).toBe(false);
  });

  it("stores a trimmed value and removes the empty default", () => {
    const setItem = vi.fn();
    const removeItem = vi.fn();
    vi.stubGlobal("window", { localStorage: { getItem: vi.fn(), setItem, removeItem } });
    saveVoiceId(" voice_1 ");
    expect(setItem).toHaveBeenCalledWith(VOICE_ID_STORAGE_KEY, "voice_1");
    saveVoiceId("  ");
    expect(removeItem).toHaveBeenCalledWith(VOICE_ID_STORAGE_KEY);
  });

  it("reads safely but surfaces save failures", () => {
    vi.stubGlobal("window", {
      localStorage: {
        getItem: () => { throw new Error("blocked"); },
        setItem: () => { throw new Error("full"); },
        removeItem: vi.fn(),
      },
    });
    expect(readVoiceId()).toBe("");
    expect(() => saveVoiceId("voice_1")).toThrow("full");
  });
});
