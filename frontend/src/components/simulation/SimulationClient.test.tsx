import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

const { startRecording } = vi.hoisted(() => ({ startRecording: vi.fn() }));

// jsdom does not implement <dialog> showModal/close — polyfill so mounted
// InterviewActions (which uses <dialog> for examination/finish modals) does not throw.
if (typeof HTMLDialogElement !== "undefined") {
  if (!HTMLDialogElement.prototype.showModal) {
    HTMLDialogElement.prototype.showModal = function showModal() {
      this.setAttribute("open", "");
    };
  }
  if (!HTMLDialogElement.prototype.close) {
    HTMLDialogElement.prototype.close = function close() {
      this.removeAttribute("open");
    };
  }
}

vi.mock("@/components/AuthProvider", () => ({
  useAuth: () => ({ token: "token", getAccessToken: async () => "token" }),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/voice/useVoiceController", () => ({
  useVoiceController: () => ({
    startRecording,
    finishRecording: vi.fn(),
    play: vi.fn(),
    stop: vi.fn(),
    speaking: false,
    listening: false,
    error: null,
  }),
}));

vi.mock("@/lib/voice/transcribeAudio", () => ({ transcribeAudio: vi.fn() }));

vi.mock("@/lib/simulation/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/simulation/api")>("@/lib/simulation/api");
  return {
    ...actual,
    fetchNextPatient: vi.fn(async () => makeSession()),
    fetchConversation: vi.fn(),
    postTurn: vi.fn(),
  };
});

import { SimulationClient } from "./SimulationClient";
import { fetchNextPatient, postTurn } from "@/lib/simulation/api";
import { transcribeAudio } from "@/lib/voice/transcribeAudio";

function makeSession() {
  return {
    conversation_id: "conv-1",
    patient_id: "patient-1",
    kind: "simulation",
    mode: "doctor_asks",
    title: "Nowy pacjent",
    card: { first_time: false, lines: [] },
    messages: [],
    ended_at: null,
  };
}

async function openKeywordsDialog() {
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: "Wygeneruj pacjenta" }));
  });
}

describe("SimulationClient — Wygeneruj pacjenta", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    startRecording.mockReset();
  });

  it("opens a dialog when Wygeneruj pacjenta is clicked and the input is not in the header", async () => {
    render(<SimulationClient />);
    const dialog = document.querySelector("dialog");
    expect(dialog?.hasAttribute("open")).toBe(false);
    expect(screen.getByRole("button", { name: "Wygeneruj pacjenta" })).toBeTruthy();
    await openKeywordsDialog();
    expect(dialog?.hasAttribute("open")).toBe(true);
    expect(screen.getByLabelText(/Słowa kluczowe pacjenta/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Generuj" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Anuluj" })).toBeTruthy();
  });

  it("passes trimmed keywords to fetchNextPatient when provided", async () => {
    vi.mocked(fetchNextPatient).mockClear();
    render(<SimulationClient />);
    await openKeywordsDialog();
    const textarea = screen.getByLabelText(/Słowa kluczowe pacjenta/) as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "  zaburzenia neurologiczne  " } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Generuj" }));
    });
    expect(fetchNextPatient).toHaveBeenCalledWith("token", "zaburzenia neurologiczne", expect.anything());
  });

  it("calls fetchNextPatient without keywords when the dialog is submitted empty", async () => {
    vi.mocked(fetchNextPatient).mockClear();
    render(<SimulationClient />);
    await openKeywordsDialog();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Generuj" }));
    });
    expect(fetchNextPatient).toHaveBeenCalledWith("token", undefined, expect.anything());
  });

  it("Następny pacjent still calls fetchNextPatient without keywords", async () => {
    vi.mocked(fetchNextPatient).mockClear();
    render(<SimulationClient />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Następny pacjent" }));
    });
    expect(fetchNextPatient).toHaveBeenCalledWith("token", undefined, expect.anything());
  });

  it("Anuluj closes the dialog without calling fetchNextPatient", async () => {
    vi.mocked(fetchNextPatient).mockClear();
    render(<SimulationClient />);
    const dialog = document.querySelector("dialog");
    await openKeywordsDialog();
    fireEvent.change(screen.getByLabelText(/Słowa kluczowe pacjenta/), { target: { value: "x" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Anuluj" }));
    });
    expect(fetchNextPatient).not.toHaveBeenCalled();
    expect(dialog?.hasAttribute("open")).toBe(false);
  });

  it("disables the dialog input and submit button while busy", async () => {
    vi.mocked(fetchNextPatient).mockImplementationOnce(
      async () => {
        await new Promise((r) => setTimeout(r, 30));
        return makeSession() as unknown as Awaited<ReturnType<typeof fetchNextPatient>>;
      },
    );

    render(<SimulationClient />);
    await openKeywordsDialog();
    const textarea = screen.getByLabelText(/Słowa kluczowe pacjenta/) as HTMLTextAreaElement;
    const submit = screen.getByRole("button", { name: /Generuj|Generowanie/ }) as HTMLButtonElement;

    await act(async () => {
      fireEvent.click(submit);
    });

    expect(textarea.disabled).toBe(true);
    expect(submit.disabled).toBe(true);

    await act(async () => {
      await new Promise((r) => setTimeout(r, 50));
    });
  });

  it("adds microphone transcription to the draft without sending a turn", async () => {
    vi.mocked(transcribeAudio).mockResolvedValueOnce({ text: "Czy ból promieniuje?" });
    vi.mocked(postTurn).mockClear();
    render(<SimulationClient />);
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Następny pacjent" })); });

    fireEvent.click(screen.getByRole("button", { name: "Mikrofon" }));
    const onBlob = startRecording.mock.calls[0]?.[0] as (blob: Blob) => void;
    await act(async () => { onBlob(new Blob(["audio"])); });

    expect((screen.getByRole("textbox", { name: "Wiadomość" }) as HTMLInputElement).value).toBe("Czy ból promieniuje?");
    expect(postTurn).not.toHaveBeenCalled();
  });
});
