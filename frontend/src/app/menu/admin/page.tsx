"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { ApiError } from "@/lib/api";
import { fetchAdminUsers, type AdminUser, updateUser } from "@/lib/admin/api";
import { canOpenAdmin, canRevokeApproval, partitionAdminUsers } from "@/lib/admin/access";

export default function AdminPage() {
  const { isAdmin, userId, token, getAccessToken } = useAuth();
  const router = useRouter();
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  useEffect(() => {
    if (!canOpenAdmin(isAdmin)) { router.replace("/menu"); return; }
    void (async () => {
      const access = token ?? (await getAccessToken());
      if (!access) return;
      try { setUsers(await fetchAdminUsers(access)); }
      catch (err) { setError(err instanceof ApiError ? err.message : "Nie udało się pobrać kont."); }
    })();
  }, [getAccessToken, isAdmin, router, token]);

  async function setApproval(row: AdminUser, approved: boolean) {
    const access = token ?? (await getAccessToken());
    if (!access) return;
    setBusyId(row.id); setError(null);
    try {
      const updated = await updateUser(access, row.id, { is_approved: approved });
      setUsers((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (err) { setError(err instanceof ApiError ? err.message : "Nie udało się zmienić statusu."); }
    finally { setBusyId(null); }
  }

  async function setSpendCap(row: AdminUser, value: string) {
    const cap = Number(value);
    if (!Number.isFinite(cap) || cap < 0) { setError("Limit musi być liczbą równą lub większą od 0."); return; }
    const access = token ?? (await getAccessToken());
    if (!access) return;
    setBusyId(row.id); setError(null);
    try {
      const updated = await updateUser(access, row.id, { spend_cap_usd: cap });
      setUsers((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (err) { setError(err instanceof ApiError ? err.message : "Nie udało się zapisać limitu."); }
    finally { setBusyId(null); }
  }

  const { pending, approved } = partitionAdminUsers(users);
  return <main className="flex-1 px-4 py-6">
    <div className="flex items-center justify-between"><h1 className="text-2xl">Administracja</h1><Link className="text-sm underline" href="/menu">Menu</Link></div>
    {error ? <p className="mt-4 text-sm text-amber-700" role="alert">{error}</p> : null}
    <p className="mt-3 text-sm text-[var(--color-soft)]">Ustaw miesięczny limit wydatków dla każdego konta. Wartość 0 oznacza brak limitu.</p>
    <UserSection title="Oczekujące konta" empty="Brak kont oczekujących na akceptację." rows={pending} selfId={userId} busyId={busyId} onSetApproval={setApproval} onSetSpendCap={setSpendCap} />
    <UserSection title="Zaakceptowane konta" empty="Brak zaakceptowanych kont." rows={approved} selfId={userId} busyId={busyId} onSetApproval={setApproval} onSetSpendCap={setSpendCap} />
  </main>;
}

function UserSection({ title, empty, rows, selfId, busyId, onSetApproval, onSetSpendCap }: { title: string; empty: string; rows: AdminUser[]; selfId: string | null; busyId: string | null; onSetApproval: (row: AdminUser, next: boolean) => Promise<void>; onSetSpendCap: (row: AdminUser, value: string) => Promise<void> }) {
  return <section className="mt-7"><h2 className="text-xl">{title}</h2>{rows.length === 0 ? <p className="mt-2 text-sm text-[var(--color-soft)]">{empty}</p> : <ul className="mt-3 space-y-2">{rows.map((row) => <li className="classical-card flex flex-wrap items-center justify-between gap-3 p-3" key={row.id}><div><p>{row.display_name || row.email || "Konto bez nazwy"}{row.is_admin ? " · administrator" : ""}</p><p className="text-xs text-[var(--color-soft)]">{row.email || row.id}</p></div><div className="flex items-center gap-2"><label className="text-xs text-[var(--color-soft)]" htmlFor={`cap-${row.id}`}>Limit USD / mies.</label><input id={`cap-${row.id}`} className="w-20 rounded border border-[var(--color-divider)] bg-[var(--color-bg)] px-2 py-1 text-sm" type="number" min="0" step="0.01" defaultValue={row.spend_cap_usd ?? 10} disabled={busyId === row.id} onBlur={(event) => { if (event.target.value !== String(row.spend_cap_usd ?? 10)) void onSetSpendCap(row, event.target.value); }} /></div>{!row.is_approved ? <button className="classical-btn classical-btn-primary" disabled={busyId === row.id} onClick={() => void onSetApproval(row, true)}>Zaakceptuj</button> : canRevokeApproval(selfId, row.id) ? <button className="classical-btn" disabled={busyId === row.id} onClick={() => void onSetApproval(row, false)}>Cofnij</button> : null}</li>)}</ul>}</section>;
}
