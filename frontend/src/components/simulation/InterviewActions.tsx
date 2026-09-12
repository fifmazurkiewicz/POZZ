"use client";

import { FormEvent, useEffect, useId, useRef, useState } from "react";
import { finishInterview, requestExamination, type SimulationSession } from "@/lib/simulation/api";
import type { RunAction } from "@/lib/useAbortableAction";

type Props = {
  session: SimulationSession;
  active: boolean;
  busy: boolean;
  error: string | null;
  getToken: () => Promise<string | null>;
  onStop: () => void;
  onUpdate: (session: SimulationSession) => void;
  run: RunAction;
};

export function InterviewActions({ session, active, busy, error, getToken, onStop, onUpdate, run }: Props) {
  const [kind, setKind] = useState<"examination" | "finish" | null>(null);
  const [examination, setExamination] = useState("");
  const [plan, setPlan] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const fieldId = useId();
  const hintId = useId();

  useEffect(() => {
    if (kind) dialog.current?.showModal();
    else dialog.current?.close();
  }, [kind]);

  function open(next: "examination" | "finish") {
    onStop();
    setKind(next);
  }
  function close() {
    onStop();
    setKind(null);
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
    });
  }

  if (session.ended_at) return <p className="px-3 py-2 text-sm font-semibold text-[var(--color-accent)]" role="status">Wywiad zakończony · tylko do odczytu</p>;

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
        <textarea id={fieldId} autoFocus required maxLength={kind === "examination" ? 2000 : 8000} rows={6} className="w-full rounded border border-[var(--color-divider)] bg-[var(--color-bg)] p-3" value={kind === "examination" ? examination : plan} disabled={busy} onChange={(event) => kind === "examination" ? setExamination(event.target.value) : setPlan(event.target.value)} placeholder={kind === "examination" ? "Np. osłuchiwanie płuc, pomiar ciśnienia, morfologia…" : "Rozpoznanie, leki, zalecenia, dalsza diagnostyka…"} />
        {error ? <p role="alert" className="text-sm text-[var(--color-text)]">{error}</p> : null}
        <div className="flex flex-wrap justify-end gap-2">
          <button type="button" className="classical-btn" onClick={close}>{busy ? "Zatrzymaj" : "Anuluj"}</button>
          <button type="submit" className="classical-btn classical-btn-primary" disabled={busy || !(kind === "examination" ? examination : plan).trim()}>{busy ? "Przetwarzanie…" : kind === "examination" ? "Wykonaj badanie" : "Zakończ i oceń"}</button>
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
