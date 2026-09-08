"use client";

import { FormEvent, useState } from "react";
import { useAuth } from "@/components/AuthProvider";
import { SimulationStatusRow } from "@/components/simulation/SimulationStatusRow";
import { ApiError } from "@/lib/api";
import {
  composerPlaceholder,
  fetchNextPatient,
  postTurn,
  speakerLabel,
  type SimMessage,
  type SimMode,
  type SimulationSession,
} from "@/lib/simulation/api";
import { cardRowsForDisplay } from "@/lib/simulation/cardDisplay";

const MODES: { id: SimMode; label: string }[] = [
  { id: "doctor_asks", label: "Lekarz pyta" },
  { id: "patient_asks", label: "Pacjent pyta" },
  { id: "meta_ask", label: "Pytaj AI" },
];

export function SimulationClient() {
  const { token, getAccessToken } = useAuth();
  const [session, setSession] = useState<SimulationSession | null>(null);
  const [messages, setMessages] = useState<SimMessage[]>([]);
  const [mode, setMode] = useState<SimMode>("doctor_asks");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cardOpen, setCardOpen] = useState(true);

  async function bearer(): Promise<string | null> {
    return token ?? (await getAccessToken());
  }

  async function onNextPatient() {
    const access = await bearer();
    if (!access) return;
    setBusy(true);
    setError(null);
    try {
      const next = await fetchNextPatient(access);
      setSession(next);
      setMessages(next.messages ?? []);
      setMode(next.mode);
      setCardOpen(true);
      setDraft("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Nie udało się wczytać pacjenta.");
    } finally {
      setBusy(false);
    }
  }

  async function onSend(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || !session) return;
    const access = await bearer();
    if (!access) return;
    setBusy(true);
    setError(null);
    try {
      const result = await postTurn(access, session.conversation_id, text, mode);
      setSession(result);
      setMessages(result.messages ?? []);
      setDraft("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Nie udało się wysłać wiadomości.");
    } finally {
      setBusy(false);
    }
  }

  const cardRows = session ? cardRowsForDisplay(session.card) : [];

  return (
    <main className="flex min-h-0 flex-1 flex-col">
      <header className="flex h-11 shrink-0 items-center justify-between border-b border-[var(--color-divider)] px-4">
        <h1 className="text-lg">Symulacja</h1>
        <button type="button" className="classical-btn text-sm" disabled={busy} onClick={() => void onNextPatient()}>
          Następny pacjent
        </button>
      </header>
      <SimulationStatusRow />
      {session ? (
        <div className="flex gap-1 overflow-x-auto border-b border-[var(--color-divider)] px-2 py-1">
          {MODES.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`classical-btn shrink-0 px-3 text-sm ${mode === item.id ? "classical-btn-primary" : ""}`}
              aria-pressed={mode === item.id}
              onClick={() => setMode(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
      ) : null}
      <section className="min-h-0 flex-1 overflow-y-auto px-4 py-4 text-sm">
        {!session ? (
          <p className="text-[var(--color-soft)]">Najpierw wygeneruj pacjenta, aby móc rozpocząć wywiad.</p>
        ) : (
          <>
            <button
              type="button"
              className="mb-3 w-full text-left font-serif text-base"
              onClick={() => setCardOpen((open) => !open)}
              aria-expanded={cardOpen}
            >
              Karta pacjenta {cardOpen ? "▾" : "▸"}
            </button>
            {cardOpen ? (
              <dl className="classical-card mb-4 space-y-1 p-3">
                {cardRows.map((row) => (
                  <div key={row.label} className="flex justify-between gap-3">
                    <dt className="text-[var(--color-soft)]">{row.label}</dt>
                    <dd>{row.value}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
            <ol className="space-y-3">
              {messages.map((msg) => (
                <li key={msg.id}>
                  <p className="text-xs text-[var(--color-soft)]">{speakerLabel(mode, msg.role)}</p>
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                </li>
              ))}
            </ol>
          </>
        )}
        {error ? (
          <p className="mt-3 text-sm text-amber-200/90" role="alert">
            {error}
          </p>
        ) : null}
      </section>
      <form className="shrink-0 border-t border-[var(--color-divider)] p-3" onSubmit={(e) => void onSend(e)}>
        <label className="sr-only" htmlFor="sim-composer">
          Wiadomość
        </label>
        <input
          id="sim-composer"
          className="min-h-11 w-full rounded border border-[var(--color-divider)] bg-[var(--color-bg)] px-3"
          placeholder={composerPlaceholder(mode)}
          value={draft}
          disabled={busy || !session}
          onChange={(e) => setDraft(e.target.value)}
        />
      </form>
    </main>
  );
}
