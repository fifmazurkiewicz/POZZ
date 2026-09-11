"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useAuth } from "@/components/AuthProvider";
import { ApiError, apiFetch } from "@/lib/api";
import type { SimulationSession } from "@/lib/simulation/api";

type Conversation = SimulationSession & { created_at: string | null; ended_at: string | null };

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

  return <main className="app-page flex-1"><div className="flex items-center justify-between"><h1 className="text-3xl">Historia rozmów</h1><Link className="classical-btn min-h-9 py-1 text-sm" href="/menu">Menu</Link></div>{error ? <p className="mt-4 text-sm text-amber-200" role="alert">{error}</p> : null}{items.length === 0 && !error ? <p className="classical-card mt-5 p-4 text-sm text-[var(--color-soft)]">Nie masz jeszcze zapisanych rozmów.</p> : <ul className="mt-5 space-y-2">{items.map((item) => <li className="classical-card p-3" key={item.conversation_id}><Link className="block" href={`/simulation?conversation=${item.conversation_id}`}><p className="font-semibold">{item.title || "Rozmowa z pacjentem"}</p><p className="mt-1 text-xs text-[var(--color-soft)]">{item.messages?.length ?? 0} wiadomości{item.created_at ? ` · ${new Intl.DateTimeFormat("pl-PL", { dateStyle: "medium", timeStyle: "short" }).format(new Date(item.created_at))}` : ""}</p></Link></li>)}</ul>}</main>;
}
