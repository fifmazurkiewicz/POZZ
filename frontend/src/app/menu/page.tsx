"use client";

import { useAuth } from "@/components/AuthProvider";

export default function MenuPage() {
  const { email, signOut, isAdmin } = useAuth();

  return (
    <main className="flex-1 px-4 py-6">
      <h1 className="text-lg">Menu</h1>
      {email ? <p className="mt-3 text-sm text-[var(--color-soft)]">{email}</p> : null}
      <p className="mt-3 text-sm text-[var(--color-soft)]">
        Sesje i Admin pojawią się w kolejnym pakiecie.
        {isAdmin ? " Masz uprawnienia administratora." : ""}
      </p>
      <button type="button" className="classical-btn mt-6" onClick={() => void signOut()}>
        Wyloguj
      </button>
      <p className="mt-6 text-xs text-[var(--color-soft)]">To jest symulator treningowy, nie urządzenie medyczne.</p>
    </main>
  );
}
