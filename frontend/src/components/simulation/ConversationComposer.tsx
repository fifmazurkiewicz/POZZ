"use client";

import { FormEvent, useId } from "react";

export function ConversationComposer({ draft, onDraftChange, onSend, disabled, placeholder, patientVoice, listening, speaking, onVoiceChange, onListeningChange }: {
  draft: string; onDraftChange: (value: string) => void; onSend: (event: FormEvent) => void;
  disabled: boolean; placeholder: string; patientVoice: boolean; listening: boolean; speaking: boolean;
  onVoiceChange: (value: boolean) => void; onListeningChange: (value: boolean) => void;
}) {
  const id = useId();
  return <form className="shrink-0 space-y-2 border-t border-[var(--color-divider)] bg-[var(--color-surface)] p-3" onSubmit={onSend}>
    <div className="flex gap-2">
      <label className="sr-only" htmlFor={id}>Wiadomość</label>
      <input id={id} className="min-h-11 min-w-0 flex-1 rounded border border-[var(--color-divider)] bg-[var(--color-bg)] px-3" value={draft} onChange={(event) => onDraftChange(event.target.value)} disabled={disabled} placeholder={placeholder} />
      <button type="submit" className="classical-btn classical-btn-primary" disabled={disabled || !draft.trim() || listening}>Wyślij</button>
    </div>
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" className="classical-btn inline-flex items-center gap-2 text-sm" disabled={disabled || speaking} aria-pressed={listening} onClick={() => onListeningChange(!listening)}>
        <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="9" y="2" width="6" height="12" rx="3" /><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8" /></svg>
        {listening ? "Wyślij nagranie" : "Mikrofon"}
      </button>
      <button type="button" className="classical-btn inline-flex items-center gap-2 text-sm" aria-pressed={patientVoice} onClick={() => onVoiceChange(!patientVoice)}>
        <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="m11 4-6 5H2v6h3l6 5z" />{patientVoice ? <path d="M15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14" /> : <path d="m16 9 6 6m0-6-6 6" />}</svg>
        {patientVoice ? "Głos włączony" : "Głos wyłączony"}
      </button>
      <span className="text-xs text-[var(--color-soft)]" role="status">{speaking ? "Głos pacjenta…" : listening ? "Nagrywanie…" : ""}</span>
    </div>
  </form>;
}
