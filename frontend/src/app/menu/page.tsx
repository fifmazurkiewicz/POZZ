"use client";

import Link from "next/link";
import { useAuth } from "@/components/AuthProvider";

export default function MenuPage() {
  const { email, signOut, isAdmin } = useAuth();

  return (
    <main className="flex-1 px-4 py-6">
      <h1 className="text-lg">Menu</h1>
      {email ? <p className="mt-3 text-sm text-[var(--color-soft)]">{email}</p> : null}
      <div className="mt-5 flex flex-col gap-2">
        <Link className="classical-btn flex items-center" href="/menu/sessions">
          Historia rozmów
        </Link>
        {isAdmin ? (
          <Link className="classical-btn flex items-center" href="/menu/admin">
            Administracja
          </Link>
        ) : null}
      </div>
      <button type="button" className="classical-btn mt-6" onClick={() => void signOut()}>
        Wyloguj
      </button>
      <p className="mt-6 text-xs text-[var(--color-soft)]">To jest symulator treningowy, nie urządzenie medyczne.</p>
    </main>
  );
}
