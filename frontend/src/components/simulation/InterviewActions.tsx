"use client";

import { FormEvent, useCallback, useEffect, useId, useRef, useState } from "react";
import { apiUrl } from "@/lib/api";
import { finishInterview, requestExamination, type SimulationSession } from "@/lib/simulation/api";
import type { RunAction } from "@/lib/useAbortableAction";
import { useVoiceController } from "@/lib/voice/useVoiceController";

type Props = {
  session: SimulationSession;
  active: boolean;
  busy: boolean;
  error: string | null;
  cancelled: boolean;
  getToken: () => Promise<string | null>;
  onStop: () => void;
  onUpdate: (session: SimulationSession) => void;
  run: RunAction;
};

const EXAMINATION_MAX = 2000;
const PLAN_MAX = 8000;

export function InterviewActions({ session, active, busy, error, cancelled, getToken, onStop, onUpdate, run }: Props) {
  const [kind, setKind] = useState<"examination" | "finish" | null>(null);
  const [examination, setExamination] = useState("");
  const [plan, setPlan] = useState("");
  const [transcribing, setTranscribing] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const fieldId = useId();
  const hintId = useId();
  const voice = useVoiceController();

  useEffect(() => {
    if (kind) dialog.current?.showModal();
    else dialog.current?.close();
  }, [kind]);

  const close = useCallback(() => {
    voice.stop();
    setTranscribing(false);
    onStop();
    setKind(null);
  }, [voice, onStop]);

  function open(next: "examination" | "finish") {
    voice.stop();
    setTranscribing(false);
    onStop();
    setKind(next);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const value = (kind === "examination" ? examination : plan).trim();
    if (!value || busy) return;
    await run(async (signal) => {
      const token = await getToken();
      signal.throwIfAborted();
      if (!token) throw new Error("Missing token");
      return kind === "examination"
        ? requestExamination(token, session.conversation_id, value, signal)
        : finishInterview(token, session.conversation_id, value, signal);
    }, (updated) => {
      onUpdate(updated);
      setKind(null);
      setExamination("");
      setPlan("");
    });
  }

  const targetValue = kind === "examination" ? examination : plan;
  const targetMax = kind === "examination" ? EXAMINATION_MAX : PLAN_MAX;
  const targetSetter = kind === "examination" ? setExamination : setPlan;
  const voiceError = voice.error;
  const voiceDisabled = busy || transcribing || (voice.speaking ?? false);

  async function appendTranscription(blob: Blob) {
    if (!kind) return;
    setTranscribing(true);
    try {
      const token = await getToken();
      if (!token) return;
      const form = new FormData();
      form.append("audio", blob, `${kind}-dictation.webm`);
      const response = await fetch(apiUrl("/api/voice/transcribe"), {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });
      if (!response.ok) {
        voice.stop();
        return;
      }
      const { text } = (await response.json()) as { text: string };
      const cleaned = text.trim();
      if (!cleaned) return;
      const setter = kind === "examination" ? setExamination : setPlan;
      const current = kind === "examination" ? examination : plan;
      const next = (current ? `${current.trimEnd()} ` : "") + cleaned;
      setter(next.length > targetMax ? next.slice(0, targetMax) : next);
    } finally {
      setTranscribing(false);
    }
  }

  function toggleMic() {
    if (voiceDisabled) return;
    if (voice.listening) voice.finishRecording();
    else voice.startRecording((blob) => void appendTranscription(blob));
  }

  if (session.ended_at) return <p className="px-3 py-2 text-sm font-semibold text-[var(--color-accent)]" role="status">Wywiad zakończony · tylko do odczytu</p>;

  const showError = !cancelled && (error || voiceError);
  const status = busy
    ? "Trwa wysyłanie…"
    : transcribing
      ? "Transkrypcja…"
      : voice.listening
        ? "Nagrywanie dyktowania…"
        : null;

  return <>
    <div className="flex shrink-0 flex-wrap gap-2 border-b border-[var(--color-divider)] px-3 py-2" aria-label="Działania w wywiadzie">
      <button className="classical-btn text-sm" type="button" disabled={!active} onClick={onStop}>Zatrzymaj</button>
      <button className="classical-btn text-sm" type="button" onClick={() => open("examination")}>Zrób badanie</button>
      <button className="classical-btn text-sm" type="button" onClick={() => open("finish")}>Zakończ wywiad</button>
    </div>
    <dialog ref={dialog} className="interview-dialog classical-card" aria-labelledby={titleId} aria-describedby={hintId} onCancel={(event) => { event.preventDefault(); close(); }}>
      <form onSubmit={(event) => void submit(event)} className="space-y-4">
        <h2 id={titleId} className="text-xl">{kind === "examination" ? "Zrób badanie" : "Zakończ wywiad"}</h2>
        <p id={hintId} className="text-sm text-[var(--color-soft)]">{kind === "examination" ? "Wynik symulowanego badania pojawi się w rozmowie." : "Zapisz rozpoznanie i plan postępowania. Po ocenie wywiad będzie dostępny tylko do odczytu."}</p>
        <label className="block text-sm" htmlFor={fieldId}>{kind === "examination" ? "Jakie badanie chcesz wykonać?" : "Rozpoznanie i plan leczenia"}</label>
        <textarea
          id={fieldId}
          autoFocus
          required
          maxLength={targetMax}
          rows={6}
          className="w-full rounded border border-[var(--color-divider)] bg-[var(--color-bg)] p-3"
          value={targetValue}
          disabled={busy || transcribing}
          onChange={(event) => targetSetter(event.target.value)}
          placeholder={kind === "examination" ? "Np. osłuchiwanie płuc, pomiar ciśnienia, morfologia…" : "Rozpoznanie, leki, zalecenia, dalsza diagnostyka…"}
        />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <button
            type="button"
            className="classical-btn inline-flex items-center gap-2 text-sm"
            disabled={voiceDisabled}
            aria-pressed={voice.listening}
            onClick={toggleMic}
          >
            <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <rect x="9" y="2" width="6" height="12" rx="3" />
              <path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8" />
            </svg>
            {voice.listening ? "Wyślij nagranie" : "Dyktuj"}
          </button>
          <span className="text-xs text-[var(--color-soft)]" role="status" aria-live="polite">{status}</span>
        </div>
        {showError ? <p role="alert" className="text-sm text-[var(--color-text)]">{error || voiceError}</p> : null}
        <div className="flex flex-wrap justify-end gap-2">
          <button type="button" className="classical-btn" onClick={close}>Anuluj</button>
          <button type="submit" className="classical-btn classical-btn-primary" disabled={busy || transcribing || !targetValue.trim()}>
            {busy ? "Przetwarzanie…" : kind === "examination" ? "Wykonaj badanie" : "Zakończ i oceń"}
          </button>
        </div>
      </form>
    </dialog>
  </>;
}

export function InterviewEvaluation({ session }: { session: SimulationSession }) {
  if (!session.ended_at) return null;
  return <section className="classical-card mt-5 space-y-3 p-4" aria-label="Ocena wywiadu">
    <h2 className="text-xl">Ocena wywiadu</h2>
    <h3>Twój plan</h3>
    <p className="whitespace-pre-wrap text-sm">{session.user_treatment_response}</p>
    <h3>Informacja zwrotna</h3>
    <p className="whitespace-pre-wrap text-sm">{session.diagnosis_evaluation || "Ocena nie jest dostępna."}</p>
  </section>;
}