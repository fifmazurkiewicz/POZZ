import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, apiUrl: (path: string) => `http://localhost:8000${path}` };
});

import { useInterviewVoiceInput } from "./useInterviewVoiceInput";

type RecordingHandlers = {
  ondataavailable: ((event: { data: Blob }) => void) | null;
  onstop: (() => void) | null;
};

let recorderState: "inactive" | "recording" = "inactive";
let pendingStop: (() => void) | null = null;
let pendingData: (() => void) | null = null;
let lastRecorderHandlers: RecordingHandlers | null = null;
let grantStream: ((stream: MediaStream) => void) | null = null;
let pendingFetch: ((response: Response) => void) | null = null;

function makeRecorderClass() {
  return class FakeRecorder {
    static isTypeSupported = () => true;
    ondataavailable: RecordingHandlers["ondataavailable"] = null;
    onstop: RecordingHandlers["onstop"] = null;
    state: "inactive" | "recording" = "inactive";
    mimeType = "audio/webm";
    constructor(stream: MediaStream) {
      void stream;
      lastRecorderHandlers = {
        ondataavailable: (event) => this.ondataavailable?.(event),
        onstop: () => this.onstop?.(),
      };
      this.ondataavailable = lastRecorderHandlers.ondataavailable;
      this.onstop = lastRecorderHandlers.onstop;
      this.state = recorderState;
    }
    start() {
      recorderState = "recording";
      this.state = "recording";
      pendingData = () => this.ondataavailable?.({ data: new Blob(["audio"], { type: "audio/webm" }) });
    }
    stop() {
      recorderState = "inactive";
      this.state = "inactive";
      pendingStop = () => this.onstop?.();
    }
  };
}

beforeEach(() => {
  recorderState = "inactive";
  pendingStop = null;
  pendingData = null;
  lastRecorderHandlers = null;
  grantStream = null;
  pendingFetch = null;
  vi.stubGlobal("MediaRecorder", makeRecorderClass());
  vi.stubGlobal("navigator", {
    mediaDevices: {
      getUserMedia: () => new Promise<MediaStream>((resolve) => {
        grantStream = resolve;
      }),
    },
  });
  vi.stubGlobal("window", { localStorage: { getItem: () => null }, speechSynthesis: { cancel: vi.fn(), speak: vi.fn() } });
  vi.stubGlobal("URL", { createObjectURL: vi.fn(() => "blob:test"), revokeObjectURL: vi.fn() });
  vi.stubGlobal("fetch", vi.fn(async () => {
    return await new Promise<Response>((resolve) => {
      pendingFetch = (response) => resolve(response as unknown as Response);
    });
  }));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function flushAsync() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("useInterviewVoiceInput", () => {
  it("starts in idle state and exposes a toggle function", () => {
    const { result } = renderHook(() => useInterviewVoiceInput({ token: "tok", appendTranscript: vi.fn() }));
    expect(result.current.listening).toBe(false);
    expect(result.current.busy).toBe(false);
    expect(result.current.error).toBe(null);
    expect(typeof result.current.toggle).toBe("function");
  });

  it("asks for microphone permission on first toggle", async () => {
    const { result } = renderHook(() => useInterviewVoiceInput({ token: "tok", appendTranscript: vi.fn() }));
    act(() => { result.current.toggle(); });
    await flushAsync();
    expect(grantStream).not.toBeNull();
    expect(result.current.listening).toBe(false); // still false until grant + start
    await act(async () => {
      grantStream!({ getTracks: () => [] } as unknown as MediaStream);
      await Promise.resolve();
    });
    expect(result.current.listening).toBe(true);
  });

  it("transcribes the recording and appends text on second toggle", async () => {
    const append = vi.fn();
    const { result } = renderHook(() => useInterviewVoiceInput({ token: "tok", appendTranscript: append }));

    // First toggle: start recording
    act(() => { result.current.toggle(); });
    await act(async () => {
      grantStream!({ getTracks: () => [] } as unknown as MediaStream);
      await Promise.resolve();
    });
    expect(result.current.listening).toBe(true);

    // Second toggle: finalize + transcribe
    act(() => { result.current.toggle(); });
    await flushAsync();
    // Simulate recorder emitting data + stopping
    await act(async () => {
      pendingData?.();
      pendingStop?.();
      await Promise.resolve();
      pendingFetch?.({
        ok: true,
        status: 200,
        statusText: "OK",
        text: async () => JSON.stringify({ text: "Ból w klatce od rana" }),
        json: async () => ({ text: "Ból w klatce od rana" }),
      } as unknown as Response);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(append).toHaveBeenCalledWith("Ból w klatce od rana");
    expect(result.current.listening).toBe(false);
    expect(result.current.busy).toBe(false);
  });

  it("surfaces ApiError message verbatim when transcription fails", async () => {
    const append = vi.fn();
    const { result } = renderHook(() => useInterviewVoiceInput({ token: "tok", appendTranscript: append }));

    act(() => { result.current.toggle(); });
    await act(async () => {
      grantStream!({ getTracks: () => [] } as unknown as MediaStream);
      await Promise.resolve();
    });

    act(() => { result.current.toggle(); });
    await flushAsync();
    // Resolve the pending fetch with an error response (the beforeEach
    // fetch mock waits on `pendingFetch`).
    await act(async () => {
      pendingData?.();
      pendingStop?.();
      // Wait for transcribeAudio to call fetch (microtask chain)
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      expect(pendingFetch).not.toBeNull();
      const respond = pendingFetch!;
      respond({
        ok: false,
        status: 403,
        statusText: "Forbidden",
        text: async () => JSON.stringify({ detail: { code: "account_pending_approval", message: "Konto oczekuje na akceptację." } }),
        json: async () => ({}),
      } as unknown as Response);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(append).not.toHaveBeenCalled();
    expect(result.current.error).toBe("Konto oczekuje na akceptację.");
    expect(result.current.busy).toBe(false);
  });

  it("aborts a previous in-flight transcription when a new recording starts", async () => {
    const append = vi.fn();
    const { result } = renderHook(() => useInterviewVoiceInput({ token: "tok", appendTranscript: append }));

    // First recording: start → finalize → fetch hangs (never resolved)
    act(() => { result.current.toggle(); });
    await act(async () => {
      grantStream!({ getTracks: () => [] } as unknown as MediaStream);
      await Promise.resolve();
    });
    act(() => { result.current.toggle(); });
    await flushAsync();
    await act(async () => {
      pendingData?.();
      pendingStop?.();
      await Promise.resolve();
      // fetch now hangs — pendingFetch never called
    });

    // Second recording: starting should cancel the previous in-flight run.
    act(() => { result.current.toggle(); });
    await act(async () => {
      grantStream!({ getTracks: () => [] } as unknown as MediaStream);
      await Promise.resolve();
    });
    act(() => { result.current.toggle(); });
    await flushAsync();
    await act(async () => {
      pendingData?.();
      pendingStop?.();
      await Promise.resolve();
      pendingFetch?.({
        ok: true,
        status: 200,
        statusText: "OK",
        text: async () => JSON.stringify({ text: "Nowa wypowiedź" }),
        json: async () => ({ text: "Nowa wypowiedź" }),
      } as unknown as Response);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(append).toHaveBeenLastCalledWith("Nowa wypowiedź");
    expect(append).not.toHaveBeenCalledWith("Ból w klatce od rana");
  });
});