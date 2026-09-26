"use client";

import { FormEvent, useCallback, useEffect, useId, useRef, useState } from "react";
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

const KEYWORDS_MAX = 500;

export function SimulationClient() {
  const { token, getAccessToken } = useAuth();
  const searchParams = useSearchParams();
  const [session, setSession] = useState<SimulationSession | null>(null);
  const [mode, setMode] = useState<SimMode>("doctor_asks");
  const [draft, setDraft] = useState("");
  const [cardOpen, setCardOpen] = useState(true);
  const [keywords, setKeywords] = useState("");
  const [keywordsOpen, setKeywordsOpen] = useState(false);
  const keywordsDialog = useRef<HTMLDialogElement>(null);
  const transcriptRef = useRef<HTMLElement>(null);
  const transcriptAtBottom = useRef(true);
  const [newerBelow, setNewerBelow] = useState(false);
  const keywordsFieldId = useId();
  const keywordsTitleId = useId();
  const keywordsHintId = useId();
  const [patientVoice, setPatientVoice] = useState(true);
  const [inputMode, setInputMode] = useState<"messages" | "conversation">("messages");
  const patientVoiceRef = useRef(true);
  const { busy, error, cancelled, run, cancel } = useAbortableAction();
  const voice = useVoiceController();
  const stopVoice = voice.stop;
  const bearer = useCallback(async () => token ?? await getAccessToken(), [token, getAccessToken]);
  const stop = useCallback(() => { cancel(); stopVoice(); setInputMode("messages"); }, [cancel, stopVoice]);

  useEffect(() => {
    if (keywordsOpen) keywordsDialog.current?.showModal();
    else keywordsDialog.current?.close();
  }, [keywordsOpen]);

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

  useEffect(() => {
    const transcript = transcriptRef.current;
    if (!transcript || !transcriptAtBottom.current) return;
    transcript.scrollTop = transcript.scrollHeight;
    setNewerBelow(false);
  }, [session?.messages?.length]);

  function onTranscriptScroll() {
    const transcript = transcriptRef.current;
    if (!transcript) return;
    transcriptAtBottom.current = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 24;
    setNewerBelow(!transcriptAtBottom.current);
  }

  function nextPatient() {
    stop();
    void run(async (signal) => {
      const access = await bearer();
      signal.throwIfAborted();
      if (!access) throw new Error("Missing token");
      return fetchNextPatient(access, undefined, signal);
    }, (next) => { setSession(next); setMode(next.mode); setDraft(""); setCardOpen(true); });
  }

  function openKeywordsDialog() {
    stop();
    setKeywordsOpen(true);
  }

  function closeKeywordsDialog() {
    setKeywords("");
    setKeywordsOpen(false);
  }

  function generateFromKeywords(event: FormEvent) {
    event.preventDefault();
    stop();
    void run(async (signal) => {
      const access = await bearer();
      signal.throwIfAborted();
      if (!access) throw new Error("Missing token");
      return fetchNextPatient(access, keywords.trim() || undefined, signal);
    }, (next) => { setSession(next); setMode(next.mode); setDraft(""); setCardOpen(true); setKeywords(""); setKeywordsOpen(false); });
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
        <button type="button" className="classical-btn text-sm" disabled={busy} onClick={nextPatient}>Następny pacjent</button>
        <button type="button" className="classical-btn classical-btn-primary text-sm" disabled={busy} onClick={openKeywordsDialog}>Wygeneruj pacjenta</button>
      </div>
    </header>
    {session && !session.ended_at ? <div className="flex shrink-0 gap-1 overflow-x-auto border-b border-[var(--color-divider)] px-3 py-1">
      {MODES.map((item) => <button key={item.id} type="button" disabled={busy} className={`classical-btn shrink-0 px-3 text-sm ${mode === item.id ? "classical-btn-primary" : ""}`} aria-pressed={mode === item.id} onClick={() => { stop(); setMode(item.id); }}>{item.label}</button>)}
    </div> : null}
    <section ref={transcriptRef} onScroll={onTranscriptScroll} className="simulation-transcript min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-4 text-sm" aria-label="Rozmowa" tabIndex={0}>
      {!session ? <p className="text-[var(--color-soft)]">Najpierw wygeneruj pacjenta, aby móc rozpocząć wywiad.</p> : <>
        <button type="button" className="mb-3 min-h-11 w-full text-left font-semibold" onClick={() => setCardOpen((open) => !open)} aria-expanded={cardOpen}>Karta pacjenta {cardOpen ? "▾" : "▸"}</button>
        {cardOpen ? <dl className="classical-card mb-4 space-y-1 p-3">{cardRowsForDisplay(session.card).map((row) => <div key={row.label} className="flex justify-between gap-3"><dt className="text-[var(--color-soft)]">{row.label}</dt><dd>{row.value}</dd></div>)}</dl> : null}
        <ol className="space-y-3">{session.messages?.map((msg) => <li key={msg.id}><p className="text-xs text-[var(--color-soft)]">{msg.content.startsWith("Wynik badania:") ? "Wynik badania · AI" : `${speakerLabel(mode, msg.role)}${msg.role === "assistant" ? " · AI" : ""}`}</p><p className="whitespace-pre-wrap break-words">{msg.content}</p></li>)}</ol>
        <InterviewEvaluation session={session} />
      </>}
    </section>
    {newerBelow ? <button type="button" className="classical-btn mx-3 mb-2 self-end text-sm" onClick={() => { const transcript = transcriptRef.current; if (transcript) { transcript.scrollTop = transcript.scrollHeight; transcriptAtBottom.current = true; setNewerBelow(false); } }}>Przejdź do najnowszej wiadomości</button> : null}
    {error || voice.error ? <p className="shrink-0 px-3 py-2 text-sm" role="alert">{error || voice.error}</p> : null}
    {session ? <InterviewActions key={session.conversation_id} session={session} active={active} busy={busy} error={error} cancelled={cancelled} getToken={bearer} onStop={stop} onUpdate={setSession} run={run} /> : busy ? <button className="classical-btn m-3" type="button" onClick={stop}>Zatrzymaj</button> : null}
    <dialog
      ref={keywordsDialog}
      className="interview-dialog classical-card"
      aria-labelledby={keywordsTitleId}
      aria-describedby={keywordsHintId}
    >
      <form onSubmit={generateFromKeywords} className="space-y-4">
        <h2 id={keywordsTitleId} className="text-xl">Wygeneruj pacjenta</h2>
        <p id={keywordsHintId} className="text-sm text-[var(--color-soft)]">
          Podaj słowa kluczowe, np. <em>zaburzenia neurologiczne</em>, <em>ból w klatce</em>. Puste pole = losowy pacjent z katalogu.
        </p>
        <label className="block text-sm" htmlFor={keywordsFieldId}>Słowa kluczowe (opcjonalne)</label>
        <textarea
          id={keywordsFieldId}
          aria-label="Słowa kluczowe pacjenta"
          autoFocus
          rows={2}
          maxLength={KEYWORDS_MAX}
          className="w-full rounded border border-[var(--color-divider)] bg-[var(--color-bg)] p-3"
          value={keywords}
          disabled={busy}
          onChange={(event) => setKeywords(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Escape") { event.preventDefault(); closeKeywordsDialog(); } }}
          placeholder="np. zaburzenia neurologiczne, ból w klatce"
        />
        <div className="flex flex-wrap justify-end gap-2">
          <button type="button" className="classical-btn" onClick={closeKeywordsDialog}>Anuluj</button>
          <button type="submit" className="classical-btn classical-btn-primary" disabled={busy}>
            {busy ? "Generowanie…" : "Generuj"}
          </button>
        </div>
      </form>
    </dialog>
    <ConversationComposer draft={draft} onDraftChange={setDraft} onSend={(event) => { event.preventDefault(); submitTurn(draft.trim()); }} disabled={busy || !session || !!session.ended_at} placeholder={session?.ended_at ? "Wywiad zakończony" : composerPlaceholder(mode)} patientVoice={patientVoice} speaking={voice.speaking} listening={voice.listening} onVoiceChange={(enabled) => { patientVoiceRef.current = enabled; setPatientVoice(enabled); if (!enabled) stopVoice(); }} onListeningChange={(enabled) => { if (enabled) void voice.startRecording((blob) => submitTurn("", blob)); else voice.finishRecording(); }} inputMode={inputMode} onInputModeChange={(next) => { if (next === "messages") voice.stopConversationTurn(); setInputMode(next); }} onConversationStart={() => void bearer().then((access) => { if (access) void voice.startConversationTurn(access, submitTurn); })} onConversationStop={() => voice.stopConversationTurn()} />
  </main>;
}
