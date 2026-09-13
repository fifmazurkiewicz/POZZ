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

  useEffect(() => { void (async () => {
    const access = token ?? await getAccessToken();
    if (!access) return;
    try { setItems((await apiFetch<{ conversations: Conversation[] }>("/api/conversations", { token: access })).conversations); }
    catch (err) { setError(err instanceof ApiError ? err.message : "Nie udało się pobrać historii."); }
  })(); }, [getAccessToken, token]);

  return <main className="app-page flex-1"><MenuBackLink /><h1 className="text-3xl">Historia rozmów</h1>{error ? <p className="mt-4 text-sm text-amber-200" role="alert">{error}</p> : null}{items.length === 0 && !error ? <p className="classical-card mt-5 p-4 text-sm text-[var(--color-soft)]">Nie masz jeszcze zapisanych rozmów.</p> : <ul className="mt-5 space-y-2">{items.map((item) => <li className="classical-card p-3" key={item.conversation_id}><Link className="block" href={conversationHref(item)}><p className="font-semibold">{item.title || "Rozmowa z pacjentem"}</p><p className="mt-1 text-xs font-semibold text-[var(--color-accent)]">{kindLabel(item.kind)}</p><p className="mt-1 text-xs text-[var(--color-soft)]">{item.kind === "recorded_interview" ? "Pełna transkrypcja" : `${item.messages?.length ?? 0} wiadomości`}{item.created_at ? ` · ${new Intl.DateTimeFormat("pl-PL", { dateStyle: "medium", timeStyle: "short" }).format(new Date(item.created_at))}` : ""}</p></Link></li>)}</ul>}</main>;
}
