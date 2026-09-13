import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, apiUrl: (path: string) => `http://localhost:8000${path}` };
});

import { useInterviewVoiceInput } from "./useInterviewVoiceInput";

type FakeRecorderInstance = {
  state: "inactive" | "recording";
  fireData: () => void;
  fireStop: () => void;
};

const recorders: FakeRecorderInstance[] = [];
let getUserMediaCalls = 0;
const getUserMediaResolvers: Array<(stream: MediaStream) => void> = [];
let pendingFetch: ((response: Response) => void) | null = null;

function makeRecorderClass() {
  return class FakeRecorder {
    static isTypeSupported = () => true;
    ondataavailable: ((event: { data: Blob }) => void) | null = null;
    onstop: (() => void) | null = null;
    state: "inactive" | "recording" = "inactive";
    mimeType = "audio/webm";
    private instance: FakeRecorderInstance;

    constructor(stream: MediaStream) {
      void stream;
      const instance: FakeRecorderInstance = {
        state: "inactive",
        fireData: () => {},
        fireStop: () => {},
      };
      instance.fireData = () => this.ondataavailable?.({ data: new Blob(["audio"], { type: "audio/webm" }) });
      instance.fireStop = () => this.onstop?.();
      this.instance = instance;
      recorders.push(instance);
    }
    start() {
      this.state = "recording";
      this.instance.state = "recording";
    }
    stop() {
      this.state = "inactive";
      this.instance.state = "inactive";
    }
  };
}

beforeEach(() => {
  recorders.length = 0;
  getUserMediaCalls = 0;
  getUserMediaResolvers.length = 0;
  pendingFetch = null;
  vi.stubGlobal("MediaRecorder", makeRecorderClass());
  vi.stubGlobal("navigator", {
    mediaDevices: {
      getUserMedia: () => {
        getUserMediaCalls += 1;
        return new Promise<MediaStream>((resolve) => {
          getUserMediaResolvers.push(resolve);
        });
      },
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

function lastRecorder(): FakeRecorderInstance {
  const rec = recorders[recorders.length - 1];
  if (!rec) throw new Error("No recorder instance created");
  return rec;
}

async function grantLastMic(): Promise<void> {
  await act(async () => {
    const resolver = getUserMediaResolvers[getUserMediaResolvers.length - 1];
    resolver!({ getTracks: () => [] } as unknown as MediaStream);
    await Promise.resolve();
  });
}

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
    expect(getUserMediaCalls).toBe(1);
    expect(result.current.listening).toBe(false); // still false until grant + start
    await grantLastMic();
    expect(result.current.listening).toBe(true);
  });

  it("transcribes the recording and appends text on second toggle", async () => {
    const append = vi.fn();
    const { result } = renderHook(() => useInterviewVoiceInput({ token: "tok", appendTranscript: append }));

    // First toggle: start recording
    act(() => { result.current.toggle(); });
    await grantLastMic();
    expect(result.current.listening).toBe(true);

    // Second toggle: finalize + transcribe
    act(() => { result.current.toggle(); });
    await flushAsync();
    // Simulate recorder emitting data + stopping
    await act(async () => {
      const rec = lastRecorder();
      rec.fireData();
      rec.fireStop();
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
    await grantLastMic();

    act(() => { result.current.toggle(); });
    await flushAsync();
    // Resolve the pending fetch with an error response (the beforeEach
    // fetch mock waits on `pendingFetch`).
    await act(async () => {
      const rec = lastRecorder();
      rec.fireData();
      rec.fireStop();
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
    await grantLastMic();
    act(() => { result.current.toggle(); });
    await flushAsync();
    await act(async () => {
      const rec = lastRecorder();
      rec.fireData();
      rec.fireStop();
      await Promise.resolve();
      // fetch now hangs — pendingFetch never called
    });

    // Second recording: starting should cancel the previous in-flight run.
    act(() => { result.current.toggle(); });
    await grantLastMic();
    act(() => { result.current.toggle(); });
    await flushAsync();
    await act(async () => {
      const rec = lastRecorder();
      rec.fireData();
      rec.fireStop();
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

  it("keeps two independent hook instances separate (no cross-contamination)", async () => {
    const titleAppend = vi.fn();
    const scenarioAppend = vi.fn();
    const { result: title } = renderHook(() =>
      useInterviewVoiceInput({ token: "tok", appendTranscript: titleAppend })
    );
    const { result: scenario } = renderHook(() =>
      useInterviewVoiceInput({ token: "tok", appendTranscript: scenarioAppend })
    );

    // Start recording on the TITLE field.
    act(() => { title.current.toggle(); });
    await grantLastMic();
    expect(title.current.listening).toBe(true);
    expect(scenario.current.listening).toBe(false);

    // Start recording on the SCENARIO field while title is still listening.
    act(() => { scenario.current.toggle(); });
    await grantLastMic();
    expect(title.current.listening).toBe(true);
    expect(scenario.current.listening).toBe(true);

    // Finalize title; its STT should land on titleAppend only.
    act(() => { title.current.toggle(); });
    await flushAsync();
    await act(async () => {
      const titleRecorder = recorders[recorders.length - 2];
      titleRecorder!.fireData();
      titleRecorder!.fireStop();
      await Promise.resolve();
      pendingFetch?.({
        ok: true,
        status: 200,
        statusText: "OK",
        text: async () => JSON.stringify({ text: "Ból w klatce" }),
        json: async () => ({ text: "Ból w klatce" }),
      } as unknown as Response);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(titleAppend).toHaveBeenCalledWith("Ból w klatce");
    expect(scenarioAppend).not.toHaveBeenCalled();

    // Finalize scenario; its STT should land on scenarioAppend only.
    act(() => { scenario.current.toggle(); });
    await flushAsync();
    await act(async () => {
      const scenarioRecorder = recorders[recorders.length - 1];
      scenarioRecorder!.fireData();
      scenarioRecorder!.fireStop();
      await Promise.resolve();
      pendingFetch?.({
        ok: true,
        status: 200,
        statusText: "OK",
        text: async () => JSON.stringify({ text: "Pacjent lat 45, od rana" }),
        json: async () => ({ text: "Pacjent lat 45, od rana" }),
      } as unknown as Response);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(scenarioAppend).toHaveBeenCalledWith("Pacjent lat 45, od rana");
    expect(titleAppend).not.toHaveBeenCalledWith("Pacjent lat 45, od rana");
  });
});