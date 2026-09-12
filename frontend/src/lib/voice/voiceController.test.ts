import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api", () => ({ apiUrl: (path: string) => `http://localhost:8000${path}` }));

import { VoiceController } from "./voiceController";

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("VoiceController", () => {
  const speaking = vi.fn();
  const listening = vi.fn();
  const error = vi.fn();

  beforeEach(() => {
    speaking.mockReset(); listening.mockReset(); error.mockReset();
    vi.stubGlobal("window", { localStorage: { getItem: () => null }, speechSynthesis: { cancel: vi.fn(), speak: vi.fn() } });
    vi.stubGlobal("URL", { createObjectURL: vi.fn(() => "blob:test"), revokeObjectURL: vi.fn() });
  });

  afterEach(() => vi.unstubAllGlobals());

  function controller() {
    return new VoiceController({ onSpeakingChange: speaking, onListeningChange: listening, onError: error });
  }

  it("aborts a loading request and does not fall back after cancellation", async () => {
    let rejectFetch!: (error: Error) => void;
    vi.stubGlobal("fetch", vi.fn((_url, init: RequestInit) => new Promise((_resolve, reject) => {
      rejectFetch = reject;
      init.signal?.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
    })));
    const voice = controller();
    const pending = voice.play("Cześć", "token");
    expect(speaking).toHaveBeenLastCalledWith(true);
    voice.stop();
    rejectFetch(Object.assign(new Error("aborted"), { name: "AbortError" }));
    await pending;
    expect(window.speechSynthesis.speak).not.toHaveBeenCalled();
    expect(speaking).toHaveBeenLastCalledWith(false);
  });

  it("falls back only for explicit tts_unavailable", async () => {
    let finish!: () => void;
    vi.stubGlobal("SpeechSynthesisUtterance", class { lang = ""; onend: (() => void) | null = null; onerror = null; constructor(_text: string) { finish = () => this.onend?.(); } });
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: false, status: 503, statusText: "Unavailable",
      json: async () => ({ detail: { code: "tts_unavailable", message: "missing" } }),
    })));
    const voice = controller();
    const pending = voice.play("Cześć", "token");
    await tick();
    expect(window.speechSynthesis.speak).toHaveBeenCalledOnce();
    finish();
    await pending;
    expect(error).not.toHaveBeenCalled();
  });

  it("does not fallback for an authorization error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: false, status: 403, statusText: "Forbidden",
      json: async () => ({ detail: { code: "account_pending_approval", message: "Brak dostępu." } }),
    })));
    await controller().play("Cześć", "token");
    expect(window.speechSynthesis.speak).not.toHaveBeenCalled();
    expect(error).toHaveBeenCalledWith("Brak dostępu.");
  });

  it("stops tracks from a microphone permission result that arrives after stop", async () => {
    let grant!: (stream: MediaStream) => void;
    const track = { stop: vi.fn() };
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: () => new Promise((resolve) => { grant = resolve; }) } });
    vi.stubGlobal("MediaRecorder", class {});
    const voice = controller();
    const pending = voice.startRecording(vi.fn());
    voice.stop();
    grant({ getTracks: () => [track] } as unknown as MediaStream);
    await pending;
    expect(track.stop).toHaveBeenCalledOnce();
    expect(listening).not.toHaveBeenCalledWith(true);
  });

  it("finishes a recording with its complete blob and stops tracks", async () => {
    const track = { stop: vi.fn() };
    const stream = { getTracks: () => [track] } as unknown as MediaStream;
    class Recorder {
      state: RecordingState = "inactive";
      mimeType = "audio/webm";
      ondataavailable: ((event: BlobEvent) => void) | null = null;
      onstop: (() => void) | null = null;
      constructor(_stream: MediaStream) {}
      start() { this.state = "recording"; }
      stop() {
        this.ondataavailable?.({ data: new Blob(["hello"]) } as BlobEvent);
        this.state = "inactive";
        this.onstop?.();
      }
    }
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: vi.fn(async () => stream) } });
    vi.stubGlobal("MediaRecorder", Recorder);
    const onBlob = vi.fn();
    const voice = controller();
    await voice.startRecording(onBlob);
    voice.finishRecording();
    expect(onBlob).toHaveBeenCalledOnce();
    expect(onBlob.mock.calls[0][0].size).toBe(5);
    expect(track.stop).toHaveBeenCalledOnce();
    expect(listening.mock.calls).toEqual([[true], [false]]);
  });
});
