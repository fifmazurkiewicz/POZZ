import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, apiUrl: (path: string) => `http://localhost:8000${path}` };
});

vi.mock("@/lib/interview/api", async () => {
  return {
    uploadRecordedInterview: vi.fn(async () => ({
      conversation_id: "rec-conv",
      patient_id: "rec-patient",
      kind: "recorded_interview",
      mode: "doctor_asks",
      title: "Wizyta kontrolna",
      card: { first_time: false, lines: [] },
    })),
  };
});

import { RecordedRecorder } from "./RecordedRecorder";
import { uploadRecordedInterview } from "@/lib/interview/api";

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

type RecorderStub = {
  state: RecordingState;
  mimeType: string;
  ondataavailable: ((event: BlobEvent) => void) | null;
  onstop: (() => void) | null;
  start: () => void;
  stop: () => void;
};

function installMediaRecorder() {
  const recorders: RecorderStub[] = [];
  class Recorder {
    state: RecordingState = "inactive";
    mimeType = "audio/webm";
    ondataavailable: ((event: BlobEvent) => void) | null = null;
    onstop: (() => void) | null = null;
    constructor(stream: MediaStream) {
      void stream;
      recorders.push(this as unknown as RecorderStub);
    }
    start() {
      this.state = "recording";
    }
    stop() {
      this.ondataavailable?.({ data: new Blob(["audio-data"]) } as BlobEvent);
      this.state = "inactive";
      this.onstop?.();
    }
  }
  const track = { stop: vi.fn() };
  const stream = { getTracks: () => [track] } as unknown as MediaStream;
  vi.stubGlobal("navigator", {
    mediaDevices: { getUserMedia: vi.fn(async () => stream) },
  });
  vi.stubGlobal("MediaRecorder", Recorder);
  return { recorders, track };
}

describe("RecordedRecorder", () => {
  beforeEach(() => {
    vi.mocked(uploadRecordedInterview).mockClear();
    vi.stubGlobal("URL", { createObjectURL: vi.fn(() => "blob:test"), revokeObjectURL: vi.fn() });
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("starts recording when the user taps the button", async () => {
    const { recorders } = installMediaRecorder();
    const onCreated = vi.fn();
    render(<RecordedRecorder token="token" title="Wizyta" onCreated={onCreated} onError={vi.fn()} />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /rozpocznij nagrywanie/i }));
    });
    await tick();

    expect(recorders).toHaveLength(1);
    expect(screen.getByRole("button", { name: /zatrzymaj i transkrybuj/i })).toBeTruthy();
    expect(screen.getByText(/nagrywanie/i)).toBeTruthy();
  });

  it("stops recording, submits the blob, and surfaces the new session", async () => {
    const { recorders } = installMediaRecorder();
    const onCreated = vi.fn();
    const onError = vi.fn();
    render(<RecordedRecorder token="token" title="Wizyta kontrolna" onCreated={onCreated} onError={onError} />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /rozpocznij nagrywanie/i }));
    });
    await tick();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /zatrzymaj i transkrybuj/i }));
    });
    await waitFor(() => expect(uploadRecordedInterview).toHaveBeenCalledTimes(1));
    expect(uploadRecordedInterview).toHaveBeenCalledWith(
      "token",
      expect.any(Blob),
      "Wizyta kontrolna",
    );
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith(expect.objectContaining({ conversation_id: "rec-conv" })));
    expect(onError).not.toHaveBeenCalled();
    expect(recorders[0]?.state).toBe("inactive");
  });

  it("surfaces a Polish error message when the upload fails", async () => {
    installMediaRecorder();
    vi.mocked(uploadRecordedInterview).mockRejectedValueOnce(
      Object.assign(new Error("Brak dostępu do mikrofonu."), { name: "ApiError", status: 403 }),
    );
    const onError = vi.fn();
    const onCreated = vi.fn();
    render(<RecordedRecorder token="token" onCreated={onCreated} onError={onError} />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /rozpocznij nagrywanie/i }));
    });
    await tick();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /zatrzymaj i transkrybuj/i }));
    });

    await waitFor(() => expect(onError).toHaveBeenCalledWith("Brak dostępu do mikrofonu."));
    expect(onCreated).not.toHaveBeenCalled();
  });
});