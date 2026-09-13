import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

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
    startRecording: vi.fn(),
    finishRecording: vi.fn(),
    play: vi.fn(),
    stop: vi.fn(),
    speaking: false,
    listening: false,
    error: null,
  }),
}));

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
import { fetchNextPatient } from "@/lib/simulation/api";

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

describe("SimulationClient — Wygeneruj pacjenta", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("passes keywords to fetchNextPatient when provided", async () => {
    render(<SimulationClient />);
    const input = screen.getByLabelText("Słowa kluczowe pacjenta") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "zaburzenia neurologiczne" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Wygeneruj pacjenta" }));
    });
    expect(fetchNextPatient).toHaveBeenCalledWith("token", "zaburzenia neurologiczne", expect.anything());
  });

  it("calls fetchNextPatient without keywords when input is empty", async () => {
    render(<SimulationClient />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Wygeneruj pacjenta" }));
    });
    expect(fetchNextPatient).toHaveBeenCalledWith("token", undefined, expect.anything());
  });

  it("Następny pacjent still calls fetchNextPatient without keywords", async () => {
    render(<SimulationClient />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Następny pacjent" }));
    });
    expect(fetchNextPatient).toHaveBeenCalledWith("token", undefined, expect.anything());
  });

  it("Escape clears the keyword input", () => {
    vi.mocked(fetchNextPatient).mockClear();
    render(<SimulationClient />);
    const input = screen.getByLabelText("Słowa kluczowe pacjenta") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "x" } });
    expect(input.value).toBe("x");
    fireEvent.keyDown(input, { key: "Escape" });
    expect(input.value).toBe("");
    expect(fetchNextPatient).not.toHaveBeenCalled();
  });

  it("disables the input and both buttons while busy", async () => {
    vi.mocked(fetchNextPatient).mockImplementationOnce(
      async () => {
        await new Promise((r) => setTimeout(r, 30));
        return makeSession() as unknown as Awaited<ReturnType<typeof fetchNextPatient>>;
      },
    );

    render(<SimulationClient />);
    const input = screen.getByLabelText("Słowa kluczowe pacjenta") as HTMLInputElement;
    const generateButton = screen.getByRole("button", { name: "Wygeneruj pacjenta" }) as HTMLButtonElement;
    const nextButton = screen.getByRole("button", { name: "Następny pacjent" }) as HTMLButtonElement;

    fireEvent.click(generateButton);
    await act(async () => {});

    expect(input.disabled).toBe(true);
    expect(generateButton.disabled).toBe(true);
    expect(nextButton.disabled).toBe(true);

    // Let the slow action settle so cleanup() does not warn.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 50));
    });
  });
});
