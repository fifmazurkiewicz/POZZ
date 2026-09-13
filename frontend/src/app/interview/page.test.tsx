import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("@/components/AuthProvider", () => ({
  useAuth: () => ({ token: "token", getAccessToken: async () => "token" }),
}));

vi.mock("@/lib/simulation/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/simulation/api")>("@/lib/simulation/api");
  return {
    ...actual,
    fetchConversation: vi.fn(),
    generateCasePlan: vi.fn(),
    postTurn: vi.fn(),
  };
});

vi.mock("@/lib/voice/useInterviewVoiceInput", () => ({
  useInterviewVoiceInput: () => ({
    listening: false, busy: false, error: null, toggle: vi.fn(), stop: vi.fn(),
  }),
}));

vi.mock("@/components/interview/RecordedRecorder", () => ({
  RecordedRecorder: () => null,
}));

import InterviewPage from "./page";
import { generateCasePlan, type SimulationSession } from "@/lib/simulation/api";

type SessionOverrides = Partial<{
  interview_summary: string | null;
  ended_at: string | null;
  title: string;
}>;

function makeSession(overrides: SessionOverrides = {}): SimulationSession {
  return {
    conversation_id: "conv-1",
    patient_id: "patient-1",
    kind: "manual_interview",
    mode: "doctor_asks",
    title: overrides.title ?? "nagranie-20260913-115752",
    card: {
      name: "",
      age: "",
      has_history_here: null,
      chronic_diseases: "",
      operations: "",
      allergies: "",
      family_history: "",
    },
    messages: [],
    ended_at: overrides.ended_at ?? null,
    interview_summary: overrides.interview_summary ?? null,
  };
}

const stubClipboard = () => {
  const write = vi.fn(async () => undefined);
  vi.stubGlobal("navigator", { ...(globalThis.navigator ?? {}), clipboard: { writeText: write } });
  return write;
};

const stubFetch = (session: SimulationSession) => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => session,
    }) as Response),
  );
};

async function startManualInterview() {
  fireEvent.click(screen.getByRole("button", { name: "Ręcznie" }));
  fireEvent.change(screen.getByLabelText("Tytuł przypadku"), { target: { value: "Tytuł" } });
  fireEvent.change(screen.getByLabelText("Opis pacjenta i sytuacji"), { target: { value: "x".repeat(40) } });
  fireEvent.click(screen.getByRole("button", { name: "Rozpocznij wywiad" }));
}

describe("InterviewPage — Opis i plan card", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.mocked(generateCasePlan).mockReset();
  });

  it("renders generated card with toggle and shows body by default", async () => {
    const summary = "## WYWIAD\n- Skargi";
    vi.mocked(generateCasePlan).mockResolvedValue(makeSession({ interview_summary: summary }));
    stubClipboard();
    stubFetch(makeSession({ interview_summary: summary }));
    render(<InterviewPage />);
    await startManualInterview();
    await waitFor(() => expect(screen.getByRole("button", { name: "Odśwież opis" })).toBeTruthy());
    expect(screen.getByRole("button", { name: "Ukryj opis" })).toBeTruthy();
    expect(screen.getByText((content) => content.includes("## WYWIAD"))).toBeTruthy();
  });

  it("hides the body when Ukryj opis is clicked and shows it again on Pokaż opis", async () => {
    const summary = "## WYWIAD\n- Skargi";
    vi.mocked(generateCasePlan).mockResolvedValue(makeSession({ interview_summary: summary }));
    stubClipboard();
    stubFetch(makeSession({ interview_summary: summary }));
    render(<InterviewPage />);
    await startManualInterview();
    await waitFor(() => expect(screen.getByText((content) => content.includes("## WYWIAD"))).toBeTruthy());

    fireEvent.click(screen.getByRole("button", { name: "Ukryj opis" }));
    expect(screen.queryByText((content) => content.includes("## WYWIAD"))).toBeNull();
    expect(screen.getByRole("button", { name: "Pokaż opis" })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Pokaż opis" }));
    expect(screen.getByText((content) => content.includes("## WYWIAD"))).toBeTruthy();
  });

  it("does not render the toggle in the empty-state card", async () => {
    stubClipboard();
    stubFetch(makeSession());
    render(<InterviewPage />);
    await startManualInterview();
    await waitFor(() => expect(screen.getByRole("button", { name: "Generuj opis i plan" })).toBeTruthy());
    expect(screen.queryByRole("button", { name: "Ukryj opis" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Pokaż opis" })).toBeNull();
  });
});