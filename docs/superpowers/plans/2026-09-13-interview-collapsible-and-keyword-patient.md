# Collapsible "Opis i plan" + keyword-driven patient — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hide/show toggle to the Wywiad "Opis i plan" card and a keyword input + "Wygeneruj pacjenta" button in the Symulacja header (both already wired to existing endpoints).

**Architecture:** Two independent UI-only deltas in the same plan because they share the brainstorming/spec phase. No backend changes, no schema changes, no new client functions. `fetchNextPatient` already accepts `keywords` (`frontend/src/lib/simulation/api.ts:24-32`).

**Tech Stack:** Next.js PWA (App Router), React 19, vitest + jsdom + `@testing-library/react`, existing CSS variable theme.

## Global Constraints

- Polish UI strings; English code/comments/commits. Preserve existing affordances.
- Existing `classical-btn` / `classical-btn-primary` classes for buttons. No new CSS files.
- Tests use vitest + `@testing-library/react` (jsdom). Mocks via `vi.mock("@/lib/api", ...)`, `vi.mock("next/navigation", ...)`, `vi.mock("@/components/AuthProvider", ...)`.
- No backend changes. No localStorage.

## File Map

| File | Role |
|---|---|
| `frontend/src/app/interview/page.tsx` | Add `planOpen` state + Ukryj/Pokaż toggle button in the generated "Opis i plan" card |
| `frontend/src/app/interview/page.test.tsx` | New — RTL tests for the toggle + empty-state card |
| `frontend/src/components/simulation/SimulationClient.tsx` | Add `keywords` state + input + "Wygeneruj pacjenta" button in the header |
| `frontend/src/components/simulation/SimulationClient.test.tsx` | New — RTL tests for keyword input + buttons |

No changes to `api.ts`, `backend/`, or any existing tests.

---

## Task 1: Wywiad "Opis i plan" — collapsible card

**Files:**
- Create: `frontend/src/app/interview/page.test.tsx`
- Modify: `frontend/src/app/interview/page.tsx` (generated-state branch around lines 299-313)

**Interfaces:**
- Consumes: existing `generatePlan`, `copyPlan`, `session.interview_summary`, `useAuth` (mocked), `useInterviewVoiceInput` (mocked), `RecordedRecorder` (mocked).
- Produces: a new `planOpen: boolean` state (default `true`) on the page component; a third button "Ukryj opis" / "Pokaż opis" rendered next to "Odśwież opis" / "Kopiuj".

### Step 1.1: Write the failing test

Create `frontend/src/app/interview/page.test.tsx`:

```tsx
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("@/components/AuthProvider", () => ({
  useAuth: () => ({ token: "token", getAccessToken: async () => "token" }),
}));

vi.mock("@/lib/simulation/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/simulation/api")>("@/lib/simulation/api");
  return {
    ...actual,
    fetchConversation: vi.fn(),
    generateCasePlan: vi.fn(async () => makeSession()),
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

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }), useSearchParams: () => new URLSearchParams() }));

import InterviewPage from "./page";

function makeSession(overrides: Partial<{
  interview_summary: string | null;
  ended_at: string | null;
  title: string;
}> = {}) {
  return {
    conversation_id: "conv-1",
    patient_id: "patient-1",
    kind: "manual_interview",
    mode: "doctor_asks",
    title: overrides.title ?? "nagranie-20260913-115752",
    card: { first_time: false, lines: [] },
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

describe("InterviewPage — Opis i plan card", () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("renders generated card with toggle and shows body by default", async () => {
    const summary = "## WYWIAD\n- Skargi";
    const { generateCasePlan } = await import("@/lib/simulation/api");
    vi.mocked(generateCasePlan).mockResolvedValue(makeSession({ interview_summary: summary }));
    stubClipboard();
    render(<InterviewPage />);
    // Switch to "Ręcznie" tab to enter the start form (we use the manual route to reach the session).
    fireEvent.click(screen.getByRole("button", { name: "Ręcznie" }));
    // Fill scenario + title to enable submit
    fireEvent.change(screen.getByLabelText("Tytuł przypadku"), { target: { value: "Tytuł" } });
    fireEvent.change(screen.getByLabelText("Opis pacjenta i sytuacji"), { target: { value: "x".repeat(40) } });
    // Stub apiFetch via fetchConversation is mocked above — but we need to stub apiFetch (used by /api/patients/manual). Use a fetch stub.
    vi.stubGlobal("fetch", vi.fn(async (url: string) => ({
      ok: true, status: 200,
      json: async () => makeSession({ interview_summary: summary }),
    }) as Response));
    fireEvent.click(screen.getByRole("button", { name: "Rozpocznij wywiad" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Odśwież opis" })).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Ukryj opis" })).toBeInTheDocument();
    expect(screen.getByText("## WYWIAD")).toBeInTheDocument();
  });

  it("hides the body when Ukryj opis is clicked and shows it again on Pokaż opis", async () => {
    const summary = "## WYWIAD\n- Skargi";
    const { generateCasePlan } = await import("@/lib/simulation/api");
    vi.mocked(generateCasePlan).mockResolvedValue(makeSession({ interview_summary: summary }));
    stubClipboard();
    render(<InterviewPage />);
    fireEvent.click(screen.getByRole("button", { name: "Ręcznie" }));
    fireEvent.change(screen.getByLabelText("Tytuł przypadku"), { target: { value: "Tytuł" } });
    fireEvent.change(screen.getByLabelText("Opis pacjenta i sytuacji"), { target: { value: "x".repeat(40) } });
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true, status: 200, json: async () => makeSession({ interview_summary: summary }),
    }) as Response));
    fireEvent.click(screen.getByRole("button", { name: "Rozpocznij wywiad" }));
    await waitFor(() => expect(screen.getByText("## WYWIAD")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Ukryj opis" }));
    expect(screen.queryByText("## WYWIAD")).toBeNull();
    expect(screen.getByRole("button", { name: "Pokaż opis" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Pokaż opis" }));
    expect(screen.getByText("## WYWIAD")).toBeInTheDocument();
  });

  it("does not render the toggle in the empty-state card", async () => {
    stubClipboard();
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true, status: 200, json: async () => makeSession(),
    }) as Response));
    render(<InterviewPage />);
    fireEvent.click(screen.getByRole("button", { name: "Ręcznie" }));
    fireEvent.change(screen.getByLabelText("Tytuł przypadku"), { target: { value: "Tytuł" } });
    fireEvent.change(screen.getByLabelText("Opis pacjenta i sytuacji"), { target: { value: "x".repeat(40) } });
    fireEvent.click(screen.getByRole("button", { name: "Rozpocznij wywiad" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Generuj opis i plan" })).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Ukryj opis" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Pokaż opis" })).toBeNull();
  });
});
```

### Step 1.2: Run test to verify it fails

Run: `cd frontend && npx vitest run src/app/interview/page.test.tsx`
Expected: FAIL — "Ukryj opis" / "Pokaż opis" buttons are not in the DOM yet (component has no `planOpen` state and no toggle button).

### Step 1.3: Implement the toggle in the page

In `frontend/src/app/interview/page.tsx`:

- Add `const [planOpen, setPlanOpen] = useState(true);` next to `const [planBusy, setPlanBusy] = useState(false);` (around line 23).
- In the generated-state branch (around lines 299-313), change:

```tsx
<section className="classical-card mt-4 space-y-2 p-4" aria-label="Opis i plan">
  <div className="flex items-center justify-between gap-2">
    <h2 className="text-xl">Opis i plan</h2>
    <div className="flex gap-2">
      <button className="classical-btn text-sm" type="button" disabled={planBusy} onClick={() => void generatePlan()}>
        {planBusy ? "Generowanie…" : "Odśwież opis"}
      </button>
      <button className="classical-btn text-sm" type="button" onClick={copyPlan}>
        Kopiuj
      </button>
    </div>
  </div>
  <pre className="whitespace-pre-wrap rounded border border-[var(--color-divider)] bg-[var(--color-bg)] p-3 text-sm">
    {session.interview_summary}
  </pre>
</section>
```

to:

```tsx
<section className="classical-card mt-4 space-y-2 p-4" aria-label="Opis i plan">
  <div className="flex items-center justify-between gap-2">
    <h2 className="text-xl">Opis i plan</h2>
    <div className="flex gap-2">
      <button
        className="classical-btn text-sm"
        type="button"
        aria-expanded={planOpen}
        aria-controls="interview-plan-body"
        onClick={() => setPlanOpen((open) => !open)}
      >
        {planOpen ? "Ukryj opis" : "Pokaż opis"}
      </button>
      <button className="classical-btn text-sm" type="button" disabled={planBusy} onClick={() => void generatePlan()}>
        {planBusy ? "Generowanie…" : "Odśwież opis"}
      </button>
      <button className="classical-btn text-sm" type="button" onClick={copyPlan}>
        Kopiuj
      </button>
    </div>
  </div>
  {planOpen ? (
    <pre id="interview-plan-body" className="whitespace-pre-wrap rounded border border-[var(--color-divider)] bg-[var(--color-bg)] p-3 text-sm">
      {session.interview_summary}
    </pre>
  ) : null}
</section>
```

Empty-state branch is untouched.

### Step 1.4: Run test to verify it passes

Run: `cd frontend && npx vitest run src/app/interview/page.test.tsx`
Expected: PASS (all three tests).

### Step 1.5: Commit

```bash
git add docs/superpowers/specs/2026-09-13-interview-collapsible-and-keyword-patient-design.md docs/superpowers/plans/2026-09-13-interview-collapsible-and-keyword-patient.md frontend/src/app/interview/page.tsx frontend/src/app/interview/page.test.tsx
git commit -m "feat(wywiad): collapsible Opis i plan card"
```

---

## Task 2: Symulacja — "Wygeneruj pacjenta" with keyword input

**Files:**
- Create: `frontend/src/components/simulation/SimulationClient.test.tsx`
- Modify: `frontend/src/components/simulation/SimulationClient.tsx` (header section, around lines 81-84)

**Interfaces:**
- Consumes: existing `fetchNextPatient(access, keywords?, signal)`, `run` from `useAbortableAction`, `busy` flag.
- Produces: a `keywords: string` state, a text input, and a primary "Wygeneruj pacjenta" button that calls `fetchNextPatient` with `keywords.trim() || undefined`.

### Step 2.1: Write the failing test

Create `frontend/src/components/simulation/SimulationClient.test.tsx`:

```tsx
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

vi.mock("@/components/AuthProvider", () => ({
  useAuth: () => ({ token: "token", getAccessToken: async () => "token" }),
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }), useSearchParams: () => new URLSearchParams() }));

vi.mock("@/lib/voice/useVoiceController", () => ({
  useVoiceController: () => ({
    startRecording: vi.fn(), finishRecording: vi.fn(), play: vi.fn(), stop: vi.fn(),
    speaking: false, listening: false, error: null,
  }),
}));

vi.mock("@/lib/simulation/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/simulation/api")>("@/lib/simulation/api");
  return {
    ...actual,
    fetchNextPatient: vi.fn(async (_token, keywords) => makeSession()),
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
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("passes keywords to fetchNextPatient when provided", async () => {
    render(<SimulationClient />);
    const input = screen.getByLabelText("Słowa kluczowe pacjenta") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "zaburzenia neurologiczne" } });
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Wygeneruj pacjenta" })); });
    expect(fetchNextPatient).toHaveBeenCalledWith("token", "zaburzenia neurologiczne", expect.anything());
  });

  it("calls fetchNextPatient without keywords when input is empty", async () => {
    render(<SimulationClient />);
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Wygeneruj pacjenta" })); });
    expect(fetchNextPatient).toHaveBeenCalledWith("token", undefined, expect.anything());
  });

  it("Następny pacjent still calls fetchNextPatient without keywords", async () => {
    render(<SimulationClient />);
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Następny pacjent" })); });
    expect(fetchNextPatient).toHaveBeenCalledWith("token", undefined, expect.anything());
  });

  it("Escape clears the keyword input", () => {
    render(<SimulationClient />);
    const input = screen.getByLabelText("Słowa kluczowe pacjenta") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "x" } });
    expect(input.value).toBe("x");
    fireEvent.keyDown(input, { key: "Escape" });
    expect(input.value).toBe("");
  });
});
```

### Step 2.2: Run test to verify it fails

Run: `cd frontend && npx vitest run src/components/simulation/SimulationClient.test.tsx`
Expected: FAIL — input with label "Słowa kluczowe pacjenta" and "Wygeneruj pacjenta" button are not in the DOM yet.

### Step 2.3: Implement the keyword input + "Wygeneruj pacjenta" button

In `frontend/src/components/simulation/SimulationClient.tsx`:

- Add `const [keywords, setKeywords] = useState("");` next to `const [cardOpen, setCardOpen] = useState(true);` (around line 23).
- Add a `generateFromKeywords` callback next to `nextPatient`:

```tsx
function generateFromKeywords() {
  stop();
  void run(async (signal) => {
    const access = await bearer();
    signal.throwIfAborted();
    if (!access) throw new Error("Missing token");
    return fetchNextPatient(access, keywords.trim() || undefined, signal);
  }, (next) => { setSession(next); setMode(next.mode); setDraft(""); setCardOpen(true); setKeywords(""); });
}
```

- Replace the header's right-side single button (around lines 81-84):

```tsx
<header className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--color-divider)] px-3 py-1">
  <div><h1 className="text-lg">Symulacja</h1><p className="text-xs text-[var(--color-soft)]">Pacjent i ocena są generowane przez AI</p></div>
  <button type="button" className="classical-btn text-sm" disabled={busy} onClick={nextPatient}>Następny pacjent</button>
</header>
```

with:

```tsx
<header className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--color-divider)] px-3 py-1">
  <div><h1 className="text-lg">Symulacja</h1><p className="text-xs text-[var(--color-soft)]">Pacjent i ocena są generowane przez AI</p></div>
  <div className="flex shrink-0 items-center gap-2">
    <input
      type="text"
      value={keywords}
      maxLength={500}
      disabled={busy}
      aria-label="Słowa kluczowe pacjenta"
      placeholder="np. zaburzenia neurologiczne, ból w klatce"
      onChange={(event) => setKeywords(event.target.value)}
      onKeyDown={(event) => { if (event.key === "Escape") setKeywords(""); }}
      className="min-h-11 w-44 rounded border border-[var(--color-divider)] bg-[var(--color-bg)] px-2 text-sm"
    />
    <button type="button" className="classical-btn text-sm" disabled={busy} onClick={nextPatient}>Następny pacjent</button>
    <button type="button" className="classical-btn classical-btn-primary text-sm" disabled={busy} onClick={generateFromKeywords}>
      {busy ? "Generowanie…" : "Wygeneruj pacjenta"}
    </button>
  </div>
</header>
```

### Step 2.4: Run test to verify it passes

Run: `cd frontend && npx vitest run src/components/simulation/SimulationClient.test.tsx`
Expected: PASS (all four tests).

### Step 2.5: Commit

```bash
git add frontend/src/components/simulation/SimulationClient.tsx frontend/src/components/simulation/SimulationClient.test.tsx
git commit -m "feat(symulacja): Wygeneruj pacjenta with keyword input"
```

---

## Task 3: Verify — full suite, lint, build

**Files:** none modified in this task.

### Step 3.1: Run the full vitest suite

Run: `cd frontend && npm test -- --run`
Expected: PASS — all existing + new tests green.

### Step 3.2: Run ESLint

Run: `cd frontend && npm run lint`
Expected: 0 errors. Warnings acceptable; fix any new warnings introduced in the changed files.

### Step 3.3: Run Next.js production build

Run: `cd frontend && npm run build`
Expected: build OK (no type errors, no missing imports). Catches any consumer of the changed components (e.g. typing of `keywords`).

### Step 3.4: Browser smoke

Per `frontend/AGENTS.md`, UI verification in browser:

1. `cd frontend && npm run dev` (already running in another terminal is fine).
2. Open `/interview` → start a manual interview with scenario ≥ 20 chars → click **"Generuj opis i plan"** → wait for summary → click **"Ukryj opis"** → verify body disappears, button label flips → click **"Pokaż opis"** → body returns. **"Odśwież opis"** and **"Kopiuj"** still work.
3. Open `/simulation` → confirm header now shows the keyword input + **"Następny pacjent"** + **"Wygeneruj pacjenta"**.
4. Type `ból w klatce piersiowej` → click **"Wygeneruj pacjenta"** → new private patient appears, conversation opened, scenario reflects keywords.
5. Clear input → click **"Wygeneruj pacjenta"** → behaves like **"Następny pacjent"** (shared catalog row if unused, or generated on demand).
6. Focus input, press Escape after typing → input clears.

### Step 3.5: Sync decisions into the plan frontmatter

Append the two new decisions to the "Decisions" section at the top of `docs/superpowers/plans/2026-09-13-interview-collapsible-and-keyword-patient.md` (mirror `2026-09-13-interview-case-plan.md`'s style):

- **2026-09-13 — Wywiad "Opis i plan" collapse is in-memory only** — no localStorage / URL persistence. Precedent: `cardOpen` on the Symulacja "Karta pacjenta".
- **2026-09-13 — Symulacja header keeps both buttons** — "Następny pacjent" (shared catalog) and "Wygeneruj pacjenta" (keyword-driven) coexist; only the latter passes non-empty `keywords` to `fetchNextPatient`. Both call `POST /api/patients/next`.

Mirror the decisions in `docs/architecture-for-cursor.md` if it has a Wywiad/Symulacja section, or note "no spec drift" otherwise.

### Step 3.6: Commit docs sync

```bash
git add docs/superpowers/plans/2026-09-13-interview-collapsible-and-keyword-patient.md AGENTS.md
git commit -m "docs: decisions for collapsible opis plan and keyword patient"
```

### Step 3.7: Final summary

Print a one-line summary of the two changes and point to the commits. Done.
