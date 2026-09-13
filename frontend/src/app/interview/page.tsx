"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useAuth } from "@/components/AuthProvider";
import { ApiError, apiFetch } from "@/lib/api";
import { RecordedRecorder } from "@/components/interview/RecordedRecorder";
import { fetchConversation, generateCasePlan, postTurn, speakerLabel, type SimMessage, type SimulationSession } from "@/lib/simulation/api";
import { useInterviewVoiceInput } from "@/lib/voice/useInterviewVoiceInput";

type InterviewMode = "recorded" | "manual";

export default function InterviewPage() {
  const { token, getAccessToken } = useAuth();
  const [title, setTitle] = useState("");
  const [scenario, setScenario] = useState("");
  const [session, setSession] = useState<SimulationSession | null>(null);
  const [messages, setMessages] = useState<SimMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [planBusy, setPlanBusy] = useState(false);
  const [planOpen, setPlanOpen] = useState(true);
  const [mode, setMode] = useState<InterviewMode>("recorded");
  const restoredConversation = useRef<string | null>(null);

  const voice = useInterviewVoiceInput({
    token: token ?? "",
    appendTranscript: (text) =>
      setDraft((current) => (current ? `${current} ${text}` : text)),
  });

  const titleVoice = useInterviewVoiceInput({
    token: token ?? "",
    appendTranscript: (text) =>
      setTitle((current) => (current ? `${current} ${text}` : text)),
  });

  const scenarioVoice = useInterviewVoiceInput({
    token: token ?? "",
    appendTranscript: (text) =>
      setScenario((current) => (current ? `${current} ${text}` : text)),
  });

  // If the interview ends while the doctor was recording, abort the
  // recording + any pending transcription so the UI does not stay stuck.
  useEffect(() => {
    if (session?.ended_at) voice.stop();
  }, [session, voice]);

  useEffect(() => {
    const conversationId = new URLSearchParams(window.location.search).get("conversation");
    if (!conversationId || restoredConversation.current === conversationId) return;
    restoredConversation.current = conversationId;
    void (async () => {
      const access = token ?? (await getAccessToken());
      if (!access) return;
      setBusy(true);
      try {
        const restored = await fetchConversation(access, conversationId);
        setSession(restored);
        setMessages(restored.messages ?? []);
        setMode(restored.kind === "recorded_interview" ? "recorded" : "manual");
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Nie udało się otworzyć wywiadu.");
      } finally {
        setBusy(false);
      }
    })();
  }, [getAccessToken, token]);

  async function accessToken() {
    return token ?? (await getAccessToken());
  }

  async function start(event: FormEvent) {
    event.preventDefault();
    const access = await accessToken();
    if (!access) return;
    setBusy(true);
    setError(null);
    try {
      const created = await apiFetch<SimulationSession>("/api/patients/manual", {
        method: "POST",
        token: access,
        body: { title, scenario },
      });
      setSession(created);
      setMessages([]);
      setDraft("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Nie udało się utworzyć przypadku.");
    } finally {
      setBusy(false);
    }
  }

  async function send(event: FormEvent) {
    event.preventDefault();
    if (!session || !draft.trim()) return;
    const access = await accessToken();
    if (!access) return;
    setBusy(true);
    setError(null);
    try {
      const updated = await postTurn(access, session.conversation_id, draft.trim(), "doctor_asks");
      setSession(updated);
      setMessages(updated.messages ?? []);
      setDraft("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Nie udało się wysłać wiadomości.");
    } finally {
      setBusy(false);
    }
  }

  async function generatePlan() {
    if (!session || planBusy) return;
    const access = await accessToken();
    if (!access) return;
    setPlanBusy(true);
    setError(null);
    try {
      const updated = await generateCasePlan(access, session.conversation_id);
      setSession(updated);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Nie udało się wygenerować opisu.");
    } finally {
      setPlanBusy(false);
    }
  }

  function copyPlan() {
    if (!session?.interview_summary) return;
    void navigator.clipboard?.writeText(session.interview_summary);
  }

  const micDisabled = busy || !session || !!session.ended_at || voice.busy;
  const titleMicDisabled = busy || titleVoice.busy;
  const scenarioMicDisabled = busy || scenarioVoice.busy;
  const voiceStatus = voice.listening ? "Nagrywanie…" : voice.busy ? "Transkrypcja…" : "";
  const titleVoiceStatus = titleVoice.listening ? "Nagrywanie…" : titleVoice.busy ? "Transkrypcja…" : "";
  const scenarioVoiceStatus = scenarioVoice.listening ? "Nagrywanie…" : scenarioVoice.busy ? "Transkrypcja…" : "";
  const banner = voice.error ?? titleVoice.error ?? scenarioVoice.error ?? error;

  return (
    <main className="app-page flex min-h-0 flex-1 flex-col">
      <h1 className="text-3xl">Wywiad</h1>
      {!session ? (
        <>
          <div className="mt-5 inline-flex gap-1 rounded border border-[var(--color-divider)] p-0.5" role="group" aria-label="Sposób dodania wywiadu">
            <button className={`classical-btn text-sm ${mode === "recorded" ? "classical-btn-primary" : ""}`} type="button" aria-pressed={mode === "recorded"} onClick={() => setMode("recorded")}>Nagranie</button>
            <button className={`classical-btn text-sm ${mode === "manual" ? "classical-btn-primary" : ""}`} type="button" aria-pressed={mode === "manual"} onClick={() => setMode("manual")}>Ręcznie</button>
          </div>
          {mode === "recorded" ? (
            <div className="classical-card mt-4 max-w-2xl space-y-3 p-4">
              <div>
                <h2 className="text-xl">Transkrypcja nagrania</h2>
                <p className="mt-1 text-sm text-[var(--color-soft)]">Nagraj rozmowę lekarza z pacjentem. Nagranie jest przetwarzane w pamięci i nie jest zapisywane.</p>
              </div>
              <label className="block text-sm" htmlFor="recording-title">Tytuł (opcjonalny)</label>
              <input id="recording-title" className="min-h-11 w-full rounded border border-[var(--color-divider)] bg-[var(--color-bg)] px-3" value={title} maxLength={120} onChange={(event) => setTitle(event.target.value)} placeholder="np. Wizyta kontrolna" />
              <RecordedRecorder
                token={token ?? ""}
                title={title}
                onCreated={(created) => {
                  setSession(created);
                  setMessages([]);
                }}
                onError={(message) => setError(message)}
              />
            </div>
          ) : (
        <form className="classical-card mt-4 max-w-2xl space-y-4 p-4" onSubmit={(event) => void start(event)}>
          <p className="text-sm text-[var(--color-soft)]">
            Utwórz prywatny przypadek ćwiczeniowy. Nie wpisuj danych umożliwiających identyfikację pacjenta.
          </p>
          <div className="block text-sm">
            <div className="flex items-center gap-1">
              <label htmlFor="case-title">Tytuł przypadku</label>
              <button
                type="button"
                className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded text-[var(--color-soft)] hover:text-[var(--color-accent)] disabled:opacity-50"
                aria-pressed={titleVoice.listening}
                aria-label={titleVoice.listening ? "Wyślij nagranie — tytuł przypadku" : "Dyktuj tytuł przypadku"}
                title={titleVoice.listening ? "Wyślij nagranie" : "Dyktuj"}
                disabled={titleMicDisabled}
                onClick={titleVoice.toggle}
              >
                <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <rect x="9" y="2" width="6" height="12" rx="3" />
                  <path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8" />
                </svg>
              </button>
            </div>
            <input
              id="case-title"
              className="mt-1 min-h-11 w-full rounded border border-[var(--color-divider)] bg-[var(--color-bg)] px-3"
              value={title}
              maxLength={120}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="np. Ból w klatce piersiowej"
            />
            {titleVoiceStatus ? (
              <p className="mt-1 text-xs text-[var(--color-soft)]" role="status">
                {titleVoiceStatus}
              </p>
            ) : null}
          </div>
          <div className="block text-sm">
            <div className="flex items-center gap-1">
              <label htmlFor="case-scenario">Opis pacjenta i sytuacji</label>
              <button
                type="button"
                className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded text-[var(--color-soft)] hover:text-[var(--color-accent)] disabled:opacity-50"
                aria-pressed={scenarioVoice.listening}
                aria-label={scenarioVoice.listening ? "Wyślij nagranie — opis pacjenta" : "Dyktuj opis pacjenta"}
                title={scenarioVoice.listening ? "Wyślij nagranie" : "Dyktuj"}
                disabled={scenarioMicDisabled}
                onClick={scenarioVoice.toggle}
              >
                <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <rect x="9" y="2" width="6" height="12" rx="3" />
                  <path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8" />
                </svg>
              </button>
            </div>
            <textarea
              id="case-scenario"
              className="mt-1 min-h-48 w-full rounded border border-[var(--color-divider)] bg-[var(--color-bg)] p-3"
              value={scenario}
              minLength={20}
              maxLength={12000}
              required
              onChange={(event) => setScenario(event.target.value)}
              placeholder="Opisz objawy, wiek oraz istotny kontekst medyczny…"
            />
            {scenarioVoiceStatus ? (
              <p className="mt-1 text-xs text-[var(--color-soft)]" role="status">
                {scenarioVoiceStatus}
              </p>
            ) : null}
          </div>
          <button className="classical-btn classical-btn-primary" disabled={busy || scenario.trim().length < 20} type="submit">
            Rozpocznij wywiad
          </button>
        </form>
          )}
        </>
      ) : (
        <>
          <div className="mt-4 flex items-center justify-between">
            <p className="text-sm text-[var(--color-soft)]">{session.title}</p>
            <button
              className="classical-btn text-sm"
              type="button"
              onClick={() => {
                setSession(null);
                setMessages([]);
                setScenario("");
                setTitle("");
              }}
            >
              Nowy przypadek
            </button>
          </div>
          <section className="mt-5 min-h-0 flex-1 overflow-y-auto">
            {session.recorded_transcript ? (
              <div className="space-y-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h2 className="text-xl">Rozmowa</h2>
                    <p className="mt-1 text-xs text-[var(--color-soft)]">Role rozmówców rozpoznano przez AI na podstawie tekstu. Zweryfikuj je z pełną transkrypcją.</p>
                  </div>
                </div>
                <ol className="space-y-3">
                  {session.recorded_transcript.turns.map((turn, index) => (
                    <li className="classical-card p-3" key={`${index}-${turn.text.slice(0, 16)}`}>
                      <p className="text-xs font-semibold text-[var(--color-soft)]">{turn.speaker === "doctor" ? "Lekarz" : turn.speaker === "patient" ? "Pacjent" : "Rozmówca"}</p>
                      <p className="mt-1 whitespace-pre-wrap">{turn.text}</p>
                    </li>
                  ))}
                </ol>
                <details className="classical-card p-4">
                  <summary className="cursor-pointer font-semibold">Pełna transkrypcja źródłowa</summary>
                  <p className="mt-3 whitespace-pre-wrap text-sm">{session.recorded_transcript.raw_text}</p>
                </details>
              </div>
            ) : <ol className="space-y-3">
              {messages.map((message) => (
                <li className="classical-card p-3" key={message.id}>
                  <p className="text-xs font-semibold text-[var(--color-soft)]">
                    {speakerLabel("doctor_asks", message.role)}
                  </p>
                  <p className="mt-1 whitespace-pre-wrap">{message.content}</p>
                </li>
              ))}
            </ol>}
          </section>
          {session.interview_summary ? (
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
          ) : (
            <section className="classical-card mt-4 flex items-center justify-between gap-2 p-4" aria-label="Opis i plan">
              <div>
                <h2 className="text-xl">Opis i plan</h2>
                <p className="text-sm text-[var(--color-soft)]">
                  Wygeneruj wersję roboczą: wywiad, różnicowanie, badania, leki, zalecenia, objawy alarmowe i dalszy plan.
                </p>
                <p className="mt-1 text-xs text-[var(--color-soft)]">Treść wygenerowana przez AI wymaga weryfikacji lekarza.</p>
              </div>
              <button
                className="classical-btn classical-btn-primary text-sm"
                type="button"
                disabled={planBusy}
                onClick={() => void generatePlan()}
              >
                {planBusy ? "Generowanie…" : "Generuj opis i plan"}
              </button>
            </section>
          )}
          {!session.recorded_transcript ? <form className="mt-4 flex gap-2" onSubmit={(event) => void send(event)}>
            <input
              className="min-h-11 min-w-0 flex-1 rounded border border-[var(--color-divider)] bg-[var(--color-bg)] px-3"
              value={draft}
              disabled={busy || voice.listening}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Zadaj pytanie pacjentowi…"
            />
            <button
              type="button"
              className="classical-btn inline-flex items-center gap-2"
              aria-pressed={voice.listening}
              aria-label="Mikrofon"
              disabled={micDisabled}
              onClick={voice.toggle}
            >
              <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                <rect x="9" y="2" width="6" height="12" rx="3" />
                <path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8" />
              </svg>
              {voice.listening ? "Wyślij nagranie" : "Mikrofon"}
            </button>
            <button className="classical-btn classical-btn-primary" type="submit" disabled={busy || !draft.trim() || voice.listening}>
              Wyślij
            </button>
          </form> : null}
          {voiceStatus ? (
            <p className="mt-1 text-xs text-[var(--color-soft)]" role="status">
              {voiceStatus}
            </p>
          ) : null}
        </>
      )}
      {banner ? (
        <p className="mt-4 text-sm text-amber-200" role="alert">
          {banner}
        </p>
      ) : null}
    </main>
  );
}
