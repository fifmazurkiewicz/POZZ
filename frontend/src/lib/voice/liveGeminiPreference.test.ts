import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  LIVE_GEMINI_STORAGE_KEY,
  readLiveGeminiPreference,
  writeLiveGeminiPreference,
} from "./liveGeminiPreference";

describe("liveGeminiPreference", () => {
  const store = new Map<string, string>();

  beforeEach(() => {
    store.clear();
    vi.stubGlobal("window", {
      localStorage: {
        getItem: (k: string) => store.get(k) ?? null,
        setItem: (k: string, v: string) => {
          store.set(k, v);
        },
        removeItem: (k: string) => {
          store.delete(k);
        },
      },
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("defaults to true when unset", () => {
    expect(readLiveGeminiPreference()).toBe(true);
  });

  it("reads false after write false", () => {
    writeLiveGeminiPreference(false);
    expect(readLiveGeminiPreference()).toBe(false);
    expect(window.localStorage.getItem(LIVE_GEMINI_STORAGE_KEY)).toBe("false");
  });

  it("reads true after write true", () => {
    writeLiveGeminiPreference(false);
    writeLiveGeminiPreference(true);
    expect(readLiveGeminiPreference()).toBe(true);
  });
});
