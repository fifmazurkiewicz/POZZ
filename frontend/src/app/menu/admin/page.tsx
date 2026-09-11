"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { ApiError } from "@/lib/api";
import { fetchAdminUsers, type AdminUser, updateUserApproval } from "@/lib/admin/api";
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
      const updated = await updateUserApproval(access, row.id, approved);
      setUsers((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (err) { setError(err instanceof ApiError ? err.message : "Nie udało się zmienić statusu."); }
    finally { setBusyId(null); }
  }

  const { pending, approved } = partitionAdminUsers(users);
  return <main className="flex-1 px-4 py-6">
    <div className="flex items-center justify-between"><h1 className="text-2xl">Administracja</h1><Link className="text-sm underline" href="/menu">Menu</Link></div>
    {error ? <p className="mt-4 text-sm text-amber-700" role="alert">{error}</p> : null}
    <UserSection title="Oczekujące konta" empty="Brak kont oczekujących na akceptację." rows={pending} selfId={userId} busyId={busyId} onSetApproval={setApproval} />
    <UserSection title="Zaakceptowane konta" empty="Brak zaakceptowanych kont." rows={approved} selfId={userId} busyId={busyId} onSetApproval={setApproval} />
  </main>;
}

function UserSection({ title, empty, rows, selfId, busyId, onSetApproval }: { title: string; empty: string; rows: AdminUser[]; selfId: string | null; busyId: string | null; onSetApproval: (row: AdminUser, next: boolean) => Promise<void> }) {
  return <section className="mt-7"><h2 className="text-xl">{title}</h2>{rows.length === 0 ? <p className="mt-2 text-sm text-[var(--color-soft)]">{empty}</p> : <ul className="mt-3 space-y-2">{rows.map((row) => <li className="classical-card flex items-center justify-between gap-3 p-3" key={row.id}><div><p>{row.display_name || row.email || "Konto bez nazwy"}{row.is_admin ? " · administrator" : ""}</p><p className="text-xs text-[var(--color-soft)]">{row.email || row.id}</p></div>{!row.is_approved ? <button className="classical-btn classical-btn-primary" disabled={busyId === row.id} onClick={() => void onSetApproval(row, true)}>Zaakceptuj</button> : canRevokeApproval(selfId, row.id) ? <button className="classical-btn" disabled={busyId === row.id} onClick={() => void onSetApproval(row, false)}>Cofnij</button> : null}</li>)}</ul>}</section>;
}
