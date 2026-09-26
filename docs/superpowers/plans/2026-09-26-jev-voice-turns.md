# JEV Voice Turns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an automatic spoken-turn mode to Simulation, submitting only when JEV confirms the utterance is complete.

**Architecture:** A FastAPI checkpoint transcribes in-memory audio and obtains a strict JEV `complete` or `continue` result. A browser controller owns capture timing, while Simulation sends an ordinary turn only after `complete`; the transcript gains managed scroll-follow behavior.

**Tech Stack:** FastAPI, httpx, Groq Whisper, OpenRouter Decisions/JEV, Next.js, MediaRecorder, Vitest.

**Spec:** `docs/superpowers/specs/2026-09-26-jev-voice-turns-design.md`

## Global Constraints

- Product UI and errors are Polish; provider errors are never exposed raw.
- Audio and checkpoint text stay in memory only; neither is persisted.
- Use `typesafe/jev-1.13` with existing `OPENROUTER_API_KEY`; add no dependency or env var.
- Check at 2.5 seconds, then every 1 second; fail closed at 15 seconds.
- JEV decides only turn completion, never a clinical outcome.
- Keep the existing text workflow intact.

## Review Focus

- Provider failure must discard capture and post no partial turn (Tasks 1, 2, and 4).
- Malformed or uncertain Decisions output must fail closed (Task 1).
- Stop, mode change, end interview, and next patient must release audio tracks and timers (Tasks 3 and 4).
- A reader scrolled upward must not be pulled down by a new message (Task 5).
- Keyboard users must receive status, error, and a labelled jump-to-latest action (Task 5).

---

### Task 1: Strict JEV turn decision

**Files:**
- Modify: `backend/app/llm/jev.py`
- Create: `backend/tests/llm/test_jev.py`

**Interfaces:**
- Produces `voice_turn_decision(transcript: str) -> Literal["complete", "continue"]` and `VoiceTurnDecisionError`.

- [ ] **Step 1: Write failing tests**

```python
def test_complete(monkeypatch):
    monkeypatch.setattr(jev.httpx, "post", complete_response)
    assert jev.voice_turn_decision("To wszystko.") == "complete"

def test_uncertain_raises(monkeypatch):
    monkeypatch.setattr(jev.httpx, "post", uncertain_response)
    with pytest.raises(jev.VoiceTurnDecisionError):
        jev.voice_turn_decision("Chciałem jeszcze")
```

- [ ] **Step 2: Verify failure** — Run `cd backend && python -m pytest tests/llm/test_jev.py -q`; expect missing helper failure.
- [ ] **Step 3: Implement minimum helper**

```python
class VoiceTurnDecisionError(Exception): pass
def voice_turn_decision(transcript: str) -> Literal["complete", "continue"]:
    value = _decisions({"transcript": transcript}, voice_turn_question())
    if value is True: return "complete"
    if value is False: return "continue"
    raise VoiceTurnDecisionError()
```

Use the existing bounded retry pattern; preserve `training_signals` behavior.
- [ ] **Step 4: Verify passing test** — Run `cd backend && python -m pytest tests/llm/test_jev.py -q`; expect PASS.
- [ ] **Step 5: Commit** — `git add backend/app/llm/jev.py backend/tests/llm/test_jev.py && git commit -m "feat: add JEV voice turn decision"`.

### Task 2: Authenticated turn-check endpoint

**Files:**
- Modify: `backend/app/api/routes/voice.py`
- Create: `backend/tests/api/test_voice_turn_check.py`

**Interfaces:**
- Consumes approved user, `audio`, `transcribe_upload`, `voice_turn_decision`.
- Produces `POST /api/voice/turn-check`: `{ "decision": "complete", "text": str }` or `{ "decision": "continue" }`.

- [ ] **Step 1: Write failing endpoint tests**

```python
def test_complete(client, monkeypatch):
    monkeypatch.setattr(voice, "transcribe_upload", async_text("To już wszystko."))
    monkeypatch.setattr(voice, "voice_turn_decision", lambda _: "complete")
    assert post_audio(client).json() == {"decision": "complete", "text": "To już wszystko."}

def test_jev_failure_is_polish_retry_error(client, monkeypatch):
    monkeypatch.setattr(voice, "voice_turn_decision", raises(VoiceTurnDecisionError()))
    response = post_audio(client)
    assert response.json()["detail"]["code"] == "voice_turn_check_failed"
```

Also test unauthenticated access, `continue`, and absence of persistence calls.
- [ ] **Step 2: Verify failure** — Run `cd backend && python -m pytest tests/api/test_voice_turn_check.py -q`; expect missing route failure.
- [ ] **Step 3: Implement endpoint**

```python
@router.post("/turn-check")
async def check_voice_turn(audio: UploadFile, user=Depends(get_approved_user), db=Depends(get_db)):
    assert_under_cap(db, user)
    text = await transcribe_audio_in_memory(audio)
    try: decision = voice_turn_decision(text)
    except VoiceTurnDecisionError as exc: raise retry_error() from exc
    return {"decision": decision, **({"text": text} if decision == "complete" else {})}
```

Use an `async with httpx.AsyncClient` block and structured Polish error `{code: voice_turn_check_failed}`.
- [ ] **Step 4: Verify passing test** — Run `cd backend && python -m pytest tests/api/test_voice_turn_check.py -q`; expect PASS.
- [ ] **Step 5: Commit** — `git add backend/app/api/routes/voice.py backend/tests/api/test_voice_turn_check.py && git commit -m "feat: add voice turn checkpoint API"`.

### Task 3: Browser checkpoint controller

**Files:**
- Create: `frontend/src/lib/voice/checkVoiceTurn.ts`
- Create: `frontend/src/lib/voice/checkVoiceTurn.test.ts`
- Modify: `frontend/src/lib/voice/voiceController.ts`
- Modify: `frontend/src/lib/voice/useVoiceController.ts`
- Modify: `frontend/src/lib/voice/voiceController.test.ts`

**Interfaces:**
- Produces `startConversationTurn(token, onComplete)` and `stopConversationTurn()` from `useVoiceController`.

- [ ] **Step 1: Write failing API-client and fake-timer tests**

```typescript
it("posts audio and returns a complete transcript", async () => {
  fetchMock.mockResolvedValue(json({ decision: "complete", text: "Gotowe." }));
  await expect(checkVoiceTurn(blob, "token")).resolves.toEqual({ decision: "complete", text: "Gotowe." });
});
it("checks after 2500ms, repeats after continue, and fails at 15000ms", async () => {
  check.mockResolvedValue({ decision: "continue" });
  await voice.startConversationTurn("token", complete);
  await vi.advanceTimersByTimeAsync(15_000);
  expect(complete).not.toHaveBeenCalled();
  expect(error).toHaveBeenCalledWith(expect.stringMatching(/Powiedz.*ponownie/));
});
it("clears timers and stops tracks when stopped", async () => {
  await voice.startConversationTurn("token", complete);
  voice.stopConversationTurn();
  expect(track.stop).toHaveBeenCalledOnce();
});
```

- [ ] **Step 2: Verify failure** — Run `cd frontend && npm test -- checkVoiceTurn voiceController`; expect missing client/method failure.
- [ ] **Step 3: Implement the smallest in-memory capture loop**

```typescript
const FIRST_CHECK_MS = 2_500;
const RETRY_CHECK_MS = 1_000;
const MAX_CAPTURE_MS = 15_000;
```

Emit accumulated recorder data for each checkpoint, release it after each terminal branch, and call `onComplete` once only on `complete`. Provider error, malformed response, cancellation, and the maximum call existing Polish error handling and never invoke `onComplete`.
- [ ] **Step 4: Verify passing test** — Run `cd frontend && npm test -- checkVoiceTurn voiceController`; expect PASS.
- [ ] **Step 5: Commit** — `git add frontend/src/lib/voice && git commit -m "feat: add automatic voice turn capture"`.

### Task 4: Simulation conversation mode

**Files:**
- Modify: `frontend/src/components/simulation/ConversationComposer.tsx`
- Modify: `frontend/src/components/simulation/SimulationClient.tsx`
- Modify: `frontend/src/components/simulation/SimulationClient.test.tsx`

**Interfaces:**
- Consumes `startConversationTurn(token, onComplete)` and `stopConversationTurn()`.
- Produces native `Wiadomości | Rozmowa` buttons with `aria-pressed`; calls existing `submitTurn(text)` only from `onComplete`.

- [ ] **Step 1: Write failing component tests**

```tsx
it("starts automatic capture in Rozmowa mode", () => {
  fireEvent.click(screen.getByRole("button", { name: "Rozmowa" }));
  fireEvent.click(screen.getByRole("button", { name: "Rozpocznij rozmowę" }));
  expect(startConversationTurn).toHaveBeenCalled();
});
it("does not post a turn after voice error", () => expect(postTurn).not.toHaveBeenCalled());
```

Test returning to Messages, patient changes, and interview end all stop capture.
- [ ] **Step 2: Verify failure** — Run `cd frontend && npm test -- SimulationClient`; expect missing mode controls.
- [ ] **Step 3: Implement mode state and lifecycle cleanup**

```tsx
const [inputMode, setInputMode] = useState<"messages" | "conversation">("messages");
function leaveConversationMode() { voice.stopConversationTurn(); setInputMode("messages"); }
```

Retain the existing text input/manual mic in Messages mode. Use a polite live region for listening/checkpoint status and `role=alert` for errors.
- [ ] **Step 4: Verify passing test** — Run `cd frontend && npm test -- SimulationClient`; expect PASS.
- [ ] **Step 5: Commit** — `git add frontend/src/components/simulation && git commit -m "feat: add simulation conversation mode"`.

### Task 5: Scrollable transcript

**Files:**
- Create: `frontend/src/components/simulation/ConversationTranscript.tsx`
- Create: `frontend/src/components/simulation/ConversationTranscript.test.tsx`
- Modify: `frontend/src/components/simulation/SimulationClient.tsx`
- Modify: `frontend/src/app/globals.css`

**Interfaces:**
- Produces `ConversationTranscript({ messages, mode, evaluation })`, a labelled scroll region, and a labelled jump button.

- [ ] **Step 1: Write failing tests**

```tsx
it("follows a new message while the reader is at bottom", () => {
  setGeometry(region, { scrollHeight: 200, clientHeight: 100, scrollTop: 100 });
  rerender(<ConversationTranscript messages={[first, second]} mode="doctor_asks" />);
  expect(region.scrollTo).toHaveBeenCalledWith({ top: 200 });
});
it("shows Przejdź do najnowszej wiadomości when reader scrolled up", () => {
  fireEvent.scroll(region, { target: { scrollTop: 10 } });
  rerender(<ConversationTranscript messages={[first, second]} mode="doctor_asks" />);
  expect(screen.getByRole("button", { name: "Przejdź do najnowszej wiadomości" })).toBeVisible();
});
it("jump button scrolls to the newest content", () => {
  fireEvent.click(screen.getByRole("button", { name: "Przejdź do najnowszej wiadomości" }));
  expect(region.scrollTo).toHaveBeenCalledWith({ top: region.scrollHeight });
});
```

- [ ] **Step 2: Verify failure** — Run `cd frontend && npm test -- ConversationTranscript`; expect missing component.
- [ ] **Step 3: Implement scroll management**

```tsx
const atBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 24;
useEffect(() => { if (atBottomRef.current) node?.scrollTo({ top: node.scrollHeight }); }, [messages.length]);
```

Apply `scrollbar-gutter: stable`; preserve touch scrolling and use `overscroll-contain`.
- [ ] **Step 4: Verify passing test** — Run `cd frontend && npm test -- ConversationTranscript`; expect PASS.
- [ ] **Step 5: Commit** — `git add frontend/src/components/simulation/ConversationTranscript* frontend/src/components/simulation/SimulationClient.tsx frontend/src/app/globals.css && git commit -m "fix: make simulation transcript follow and scroll"`.

### Task 6: Documentation and verification

**Files:**
- Modify: `docs/ux/ux-ui-spec.md`
- Modify: `docs/technical/configuration.md`
- Modify: `docs/superpowers/specs/2026-09-25-jev-decisions-design.md`

- [ ] **Step 1: Document the non-clinical turn-boundary exception, 2.5/1/15 timing, and no-new-env configuration.**
- [ ] **Step 2: Run backend suite** — `cd backend && python -m pytest`; expect PASS.
- [ ] **Step 3: Run frontend suite** — `cd frontend && npm run lint && npm test && npm run build`; expect PASS.
- [ ] **Step 4: Commit only feature files** — `git add backend frontend docs && git commit -m "feat: add JEV simulation voice turns"`; confirm `.cursor/*` is unstaged.
- [ ] **Step 5: Push main** — `git push origin main`; expect configured deployment to start.
