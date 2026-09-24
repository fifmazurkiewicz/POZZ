"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useAuth } from "@/components/AuthProvider";
import { ApiError, apiFetch } from "@/lib/api";
import type { SimulationSession } from "@/lib/simulation/api";
import { MenuBackLink } from "@/components/MenuBackLink";

type Conversation = SimulationSession & { created_at: string | null; ended_at: string | null };

function kindLabel(kind: string): string {
  if (kind === "recorded_interview") return "Wywiad z nagrania";
  if (kind === "manual_interview") return "Wywiad ręczny";
  return "Symulacja";
}

function conversationHref(item: Conversation): string {
  const base = item.kind === "simulation" ? "/simulation" : "/interview";
  return `${base}?conversation=${item.conversation_id}`;
}

export default function SessionsPage() {
  const { token, getAccessToken } = useAuth();
  const [items, setItems] = useState<Conversation[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const load = async () => {
    setLoading(true); setError(null);
    const access = token ?? await getAccessToken();
    if (!access) { setLoading(false); return; }
    try { setItems((await apiFetch<{ conversations: Conversation[] }>("/api/conversations", { token: access })).conversations); }
    catch (err) { setError(err instanceof ApiError ? err.message : "Nie udało się pobrać historii."); }
    finally { setLoading(false); }
  };

  useEffect(() => {
    void (async () => {
      const access = token ?? await getAccessToken();
      if (!access) { setLoading(false); return; }
      try { setItems((await apiFetch<{ conversations: Conversation[] }>("/api/conversations", { token: access })).conversations); }
      catch (err) { setError(err instanceof ApiError ? err.message : "Nie udało się pobrać historii."); }
      finally { setLoading(false); }
    })();
  }, [getAccessToken, token]);

  async function removeConversation(item: Conversation) {
    if (!window.confirm(`Usunąć rozmowę „${item.title || "Rozmowa z pacjentem"}”? Tej operacji nie można cofnąć.`)) return;
    const access = token ?? await getAccessToken();
    if (!access) return;
    setDeletingId(item.conversation_id);
    setError(null);
    try {
      await apiFetch<void>(`/api/conversations/${item.conversation_id}`, { method: "DELETE", token: access });
      setItems((current) => current.filter((entry) => entry.conversation_id !== item.conversation_id));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Nie udało się usunąć rozmowy.");
    } finally {
      setDeletingId(null);
    }
  }

  return <main className="app-page flex-1"><MenuBackLink /><h1 className="text-3xl">Historia rozmów</h1>{loading ? <p className="mt-4 text-sm" role="status">Wczytywanie historii…</p> : error ? <div className="mt-4 space-y-2" role="alert"><p className="text-sm text-amber-200">{error}</p><button className="classical-btn" type="button" onClick={() => void load()}>Spróbuj ponownie</button></div> : items.length === 0 ? <p className="classical-card mt-5 p-4 text-sm text-[var(--color-soft)]">Nie masz jeszcze zapisanych rozmów.</p> : <ul className="mt-5 space-y-2">{items.map((item) => <li className="classical-card flex items-center gap-3 p-3" key={item.conversation_id}><Link className="min-w-0 flex-1" href={conversationHref(item)}><p className="truncate font-semibold">{item.title || "Rozmowa z pacjentem"}</p><p className="mt-1 text-xs font-semibold text-[var(--color-accent)]">{kindLabel(item.kind)}</p><p className="mt-1 text-xs text-[var(--color-soft)]">{item.kind === "recorded_interview" ? "Pełna transkrypcja" : `${item.messages?.length ?? 0} wiadomości`}{item.created_at ? ` · ${new Intl.DateTimeFormat("pl-PL", { dateStyle: "medium", timeStyle: "short" }).format(new Date(item.created_at))}` : ""}</p></Link><button className="classical-btn shrink-0 text-sm" type="button" disabled={deletingId === item.conversation_id} onClick={() => void removeConversation(item)} aria-label={`Usuń rozmowę ${item.title || "Rozmowa z pacjentem"}`}>{deletingId === item.conversation_id ? "Usuwanie…" : "Usuń"}</button></li>)}</ul>}</main>;
}
