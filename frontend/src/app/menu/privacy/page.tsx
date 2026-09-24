"use client";

import { useState } from "react";
import { useAuth } from "@/components/AuthProvider";
import { apiFetch, apiUrl } from "@/lib/api";
import { MenuBackLink } from "@/components/MenuBackLink";

export default function PrivacyControlsPage() {
  const { token, getAccessToken } = useAuth();
  const [confirmation, setConfirmation] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const access = async () => token ?? await getAccessToken();

  async function downloadExport() {
    setBusy(true); setMessage(null);
    try {
    const bearer = await access();
    if (!bearer) return;
    const data = await apiFetch<Record<string, unknown>>("/api/privacy/export", { token: bearer });
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = "pozz-data-export.json"; anchor.click();
    URL.revokeObjectURL(url);
    setMessage("Pobrano kopię danych.");
    } catch { setMessage("Nie udało się pobrać danych."); }
    finally { setBusy(false); }
  }

  async function eraseContent() {
    setBusy(true); setMessage(null);
    try {
    const bearer = await access();
    if (!bearer) return;
    const response = await fetch(apiUrl("/api/privacy/content"), {
      method: "DELETE",
      headers: { Authorization: `Bearer ${bearer}`, "Content-Type": "application/json" },
      body: JSON.stringify({ confirmation }),
    });
    setMessage(response.ok ? "Treść została usunięta. Konto logowania pozostało aktywne." : "Nie udało się usunąć danych.");
    if (response.ok) setConfirmation("");
    } catch { setMessage("Nie udało się usunąć danych."); }
    finally { setBusy(false); }
  }

  return <main className="app-page flex-1 overflow-y-auto">
    <MenuBackLink />
    <h1 className="text-3xl">Prywatność</h1>
    <p className="mt-3 text-sm text-[var(--color-soft)]">Pobierz dane zapisane przez POZZ albo usuń rozmowy, transkrypcje, oceny i prywatne przypadki.</p>
    <button className="classical-btn mt-5" type="button" disabled={busy} onClick={() => void downloadExport()}>{busy ? "Przetwarzanie…" : "Pobierz moje dane"}</button>
    <section className="classical-card mt-6 max-w-xl space-y-3 p-4">
      <h2 className="text-xl">Usuń treść</h2>
      <p className="text-sm text-[var(--color-soft)]">Ta operacja nie usuwa konta Google/Supabase ani kopii zapasowych. Wpisz dokładnie: <strong>USUŃ MOJE DANE</strong></p>
      <input className="min-h-11 w-full rounded border border-[var(--color-divider)] bg-[var(--color-bg)] px-3" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} />
      <button className="classical-btn" type="button" disabled={busy || confirmation !== "USUŃ MOJE DANE"} onClick={() => void eraseContent()}>Usuń moją treść</button>
    </section>
    {message ? <p className="mt-4 text-sm" role="status">{message}</p> : null}
  </main>;
}
