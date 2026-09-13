"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { apiUrl } from "@/lib/api";
import { composerPlaceholder, fetchNextPatient, fetchConversation, postTurn, speakerLabel, type SimMode, type SimulationSession } from "@/lib/simulation/api";
import { cardRowsForDisplay } from "@/lib/simulation/cardDisplay";
import { useAbortableAction } from "@/lib/useAbortableAction";
import { useVoiceController } from "@/lib/voice/useVoiceController";
import { ConversationComposer } from "./ConversationComposer";
import { InterviewActions, InterviewEvaluation } from "./InterviewActions";

const MODES: { id: SimMode; label: string }[] = [
  { id: "doctor_asks", label: "Lekarz pyta" },
  { id: "patient_asks", label: "Pacjent pyta" },
  { id: "meta_ask", label: "Pytaj AI" },
];

export function SimulationClient() {
  const { token, getAccessToken } = useAuth();
  const searchParams = useSearchParams();
  const [session, setSession] = useState<SimulationSession | null>(null);
  const [mode, setMode] = useState<SimMode>("doctor_asks");
  const [draft, setDraft] = useState("");
  const [cardOpen, setCardOpen] = useState(true);
  const [keywords, setKeywords] = useState("");
  const [patientVoice, setPatientVoice] = useState(true);
  const patientVoiceRef = useRef(true);
  const { busy, error, cancelled, run, cancel } = useAbortableAction();
  const voice = useVoiceController();
  const stopVoice = voice.stop;
  const bearer = useCallback(async () => token ?? await getAccessToken(), [token, getAccessToken]);
  const stop = useCallback(() => { cancel(); stopVoice(); }, [cancel, stopVoice]);

  useEffect(() => {
    const id = searchParams.get("conversation");
    if (!id) return;
    stopVoice();
    void run(async (signal) => {
      const access = await bearer();
      signal.throwIfAborted();
      if (!access) throw new Error("Missing token");
      return fetchConversation(access, id, signal);
    }, (saved) => { setSession(saved); setMode(saved.mode); setCardOpen(true); });
  }, [bearer, run, searchParams, stopVoice]);

  function nextPatient() {
    stop();
    void run(async (signal) => {
      const access = await bearer();
      signal.throwIfAborted();
      if (!access) throw new Error("Missing token");
      return fetchNextPatient(access, undefined, signal);
    }, (next) => { setSession(next); setMode(next.mode); setDraft(""); setCardOpen(true); });
  }

  function generateFromKeywords() {
    stop();
    void run(async (signal) => {
      const access = await bearer();
      signal.throwIfAborted();
      if (!access) throw new Error("Missing token");
      return fetchNextPatient(access, keywords.trim() || undefined, signal);
    }, (next) => { setSession(next); setMode(next.mode); setDraft(""); setCardOpen(true); setKeywords(""); });
  }

  function submitTurn(text: string, audio?: Blob) {
    if (!session || session.ended_at || (!text && !audio)) return;
    stop();
    void run(async (signal) => {
      const access = await bearer();
      signal.throwIfAborted();
      if (!access) throw new Error("Missing token");
      if (audio) {
        const form = new FormData(); form.append("audio", audio, "doctor-turn.webm");
        const response = await fetch(apiUrl("/api/voice/transcribe"), { method: "POST", signal, headers: { Authorization: `Bearer ${access}` }, body: form });
        if (!response.ok) throw new Error("Transcription failed");
        text = (await response.json() as { text: string }).text;
      }
      signal.throwIfAborted();
      const result = await postTurn(access, session.conversation_id, text, mode, signal);
      return { result, access };
    }, ({ result, access }) => {
      setSession(result); setDraft("");
      if (patientVoiceRef.current && result.assistant?.content) void voice.play(result.assistant.content, access);
    });
  }

  const active = busy || voice.speaking || voice.listening;
  return <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
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
    {session && !session.ended_at ? <div className="flex shrink-0 gap-1 overflow-x-auto border-b border-[var(--color-divider)] px-3 py-1">
      {MODES.map((item) => <button key={item.id} type="button" disabled={busy} className={`classical-btn shrink-0 px-3 text-sm ${mode === item.id ? "classical-btn-primary" : ""}`} aria-pressed={mode === item.id} onClick={() => { stop(); setMode(item.id); }}>{item.label}</button>)}
    </div> : null}
    <section className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-4 text-sm" aria-label="Rozmowa">
      {!session ? <p className="text-[var(--color-soft)]">Najpierw wygeneruj pacjenta, aby móc rozpocząć wywiad.</p> : <>
        <button type="button" className="mb-3 min-h-11 w-full text-left font-semibold" onClick={() => setCardOpen((open) => !open)} aria-expanded={cardOpen}>Karta pacjenta {cardOpen ? "▾" : "▸"}</button>
        {cardOpen ? <dl className="classical-card mb-4 space-y-1 p-3">{cardRowsForDisplay(session.card).map((row) => <div key={row.label} className="flex justify-between gap-3"><dt className="text-[var(--color-soft)]">{row.label}</dt><dd>{row.value}</dd></div>)}</dl> : null}
        <ol className="space-y-3">{session.messages?.map((msg) => <li key={msg.id}><p className="text-xs text-[var(--color-soft)]">{msg.content.startsWith("Wynik badania:") ? "Wynik badania · AI" : `${speakerLabel(mode, msg.role)}${msg.role === "assistant" ? " · AI" : ""}`}</p><p className="whitespace-pre-wrap break-words">{msg.content}</p></li>)}</ol>
        <InterviewEvaluation session={session} />
      </>}
    </section>
    {error || voice.error ? <p className="shrink-0 px-3 py-2 text-sm" role="alert">{error || voice.error}</p> : null}
    {session ? <InterviewActions key={session.conversation_id} session={session} active={active} busy={busy} error={error} cancelled={cancelled} getToken={bearer} onStop={stop} onUpdate={setSession} run={run} /> : busy ? <button className="classical-btn m-3" type="button" onClick={stop}>Zatrzymaj</button> : null}
    <ConversationComposer draft={draft} onDraftChange={setDraft} onSend={(event) => { event.preventDefault(); submitTurn(draft.trim()); }} disabled={busy || !session || !!session.ended_at} placeholder={session?.ended_at ? "Wywiad zakończony" : composerPlaceholder(mode)} patientVoice={patientVoice} speaking={voice.speaking} listening={voice.listening} onVoiceChange={(enabled) => { patientVoiceRef.current = enabled; setPatientVoice(enabled); if (!enabled) stopVoice(); }} onListeningChange={(enabled) => { if (enabled) void voice.startRecording((blob) => submitTurn("", blob)); else voice.finishRecording(); }} />
  </main>;
}
