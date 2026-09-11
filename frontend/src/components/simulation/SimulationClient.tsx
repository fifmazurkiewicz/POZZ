"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useAuth } from "@/components/AuthProvider";
import { SimulationStatusRow } from "@/components/simulation/SimulationStatusRow";
import { ApiError } from "@/lib/api";
import { apiUrl } from "@/lib/api";
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
  const [patientVoice, setPatientVoice] = useState(true);
  const [listening, setListening] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => () => streamRef.current?.getTracks().forEach((track) => track.stop()), []);

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

  async function submitTurn(text: string) {
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
      if (patientVoice && result.assistant?.content) void speakPatient(result.assistant.content, access);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Nie udało się wysłać wiadomości.");
    } finally {
      setBusy(false);
    }
  }

  async function onSend(event: FormEvent) { event.preventDefault(); await submitTurn(draft.trim()); }

  async function speakPatient(text: string, access: string) {
    try {
      const response = await fetch(apiUrl("/api/voice/speech"), { method: "POST", headers: { Authorization: `Bearer ${access}`, "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
      if (response.ok) {
        const url = URL.createObjectURL(await response.blob());
        const audio = new Audio(url);
        setSpeaking(true);
        audio.onended = () => { URL.revokeObjectURL(url); setSpeaking(false); };
        audio.onerror = () => { URL.revokeObjectURL(url); setSpeaking(false); };
        await audio.play();
        return;
      }
    } catch { /* Browser speech is the intentional no-key fallback. */ }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "pl-PL";
      setSpeaking(true);
      utterance.onend = () => setSpeaking(false);
      utterance.onerror = () => setSpeaking(false);
      window.speechSynthesis.speak(utterance);
    }
  }

  async function onListeningChange(next: boolean) {
    if (!next) { recorderRef.current?.stop(); return; }
    if (!session) { setError("Najpierw wybierz pacjenta."); return; }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { setError("Ta przeglądarka nie obsługuje nagrywania głosu."); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream; chunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => { if (event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop = () => { stream.getTracks().forEach((track) => track.stop()); streamRef.current = null; recorderRef.current = null; setListening(false); void transcribeAndSend(new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" })); };
      recorder.start(); setListening(true); setError(null);
    } catch { setError("Brak dostępu do mikrofonu."); }
  }

  async function transcribeAndSend(audio: Blob) {
    const access = await bearer(); if (!access || !audio.size) return;
    setBusy(true); setError(null);
    try {
      const form = new FormData(); form.append("audio", audio, "doctor-turn.webm");
      const response = await fetch(apiUrl("/api/voice/transcribe"), { method: "POST", headers: { Authorization: `Bearer ${access}` }, body: form });
      if (!response.ok) throw new Error("Nie udało się rozpoznać wypowiedzi.");
      const { text } = await response.json() as { text: string };
      await submitTurn(text);
    } catch (err) { setError(err instanceof Error ? err.message : "Nie udało się rozpoznać wypowiedzi."); }
    finally { setBusy(false); }
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
      <SimulationStatusRow patientVoice={patientVoice} listening={listening} disabled={busy || speaking} onPatientVoiceChange={setPatientVoice} onListeningChange={(next) => void onListeningChange(next)} />
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
