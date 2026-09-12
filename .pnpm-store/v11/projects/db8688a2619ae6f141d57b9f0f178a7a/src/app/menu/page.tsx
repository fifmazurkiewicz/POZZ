"use client";

import Link from "next/link";
import { useAuth } from "@/components/AuthProvider";

export default function MenuPage() {
  const { email, signOut, isAdmin } = useAuth();

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
      <button type="button" className="classical-btn mt-8" onClick={() => void signOut()}>
        Wyloguj
      </button>
      <p className="mt-6 text-xs text-[var(--color-soft)]">To jest symulator treningowy, nie urządzenie medyczne.</p>
    </main>
  );
}
