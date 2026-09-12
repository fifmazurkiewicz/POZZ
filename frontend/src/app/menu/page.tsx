"use client";

import Link from "next/link";
import { useState } from "react";
import { useAuth } from "@/components/AuthProvider";
import { readVoiceId, saveVoiceId } from "@/lib/voice/voicePreference";

export default function MenuPage() {
  const { email, signOut, isAdmin } = useAuth();
  const [voiceId, setVoiceId] = useState(() => readVoiceId());
  const [voiceMessage, setVoiceMessage] = useState<string | null>(null);

  function persistVoiceId(value: string | null) {
    try {
      saveVoiceId(value ?? "");
      setVoiceId(value ?? "");
      setVoiceMessage(value && value.trim() ? "Zapisano głos dla tej przeglądarki." : "Używasz domyślnego głosu serwera.");
    } catch {
      setVoiceMessage("Nieprawidłowy identyfikator. Dozwolone litery, cyfry, podkreślniki i myślniki (do 128 znaków).");
    }
  }

  return (
    <main className="app-page flex-1">
      <p className="text-sm font-semibold text-[var(--color-accent)]">POZZ</p>
      <h1 className="mt-1 text-3xl">Menu</h1>
      {email ? <p className="mt-3 text-sm text-[var(--color-soft)]">{email}</p> : null}
      <div className="mt-5 flex flex-col gap-2">
        <Link className="classical-btn flex items-center" href="/menu/sessions">
          Historia rozmów
        </Link>
        <Link className="classical-btn flex items-center" href="/menu/privacy">
          Prywatność i moje dane
        </Link>
        {isAdmin ? (
          <Link className="classical-btn flex items-center" href="/menu/admin">
            Administracja
          </Link>
        ) : null}
      </div>
      <section className="classical-card mt-6 max-w-xl space-y-3 p-4">
        <h2 className="text-xl">Głos</h2>
        <p className="text-sm text-[var(--color-soft)]">Preferencja jest zapisana w tej przeglądarce. Puste pole oznacza domyślny głos serwera.</p>
        <label htmlFor="voice-id" className="block text-sm text-[var(--color-soft)]">Identyfikator głosu ElevenLabs (opcjonalny)</label>
        <input
          id="voice-id"
          className="min-h-11 w-full rounded border border-[var(--color-divider)] bg-[var(--color-bg)] px-3"
          value={voiceId}
          onChange={(event) => {
            setVoiceId(event.target.value);
            if (voiceMessage) setVoiceMessage(null);
          }}
        />
        <div className="flex flex-wrap gap-2">
          <button type="button" className="classical-btn" onClick={() => persistVoiceId(voiceId)}>Zapisz głos</button>
          <button type="button" className="classical-btn" onClick={() => persistVoiceId("")}>Przywróć domyślny</button>
        </div>
        {voiceMessage ? <p className="text-sm" role="status">{voiceMessage}</p> : null}
      </section>
      <button type="button" className="classical-btn mt-8" onClick={() => void signOut()}>
        Wyloguj
      </button>
      <p className="mt-6 text-xs text-[var(--color-soft)]">To jest symulator treningowy, nie urządzenie medyczne.</p>
    </main>
  );
}
