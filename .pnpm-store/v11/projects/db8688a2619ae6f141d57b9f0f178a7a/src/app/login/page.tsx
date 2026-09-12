"use client";

import { useState } from "react";
import Link from "next/link";
import { useAuth } from "@/components/AuthProvider";

export default function LoginPage() {
  const { signInWithGoogle } = useAuth();
  const [oauthError] = useState(() => {
    if (typeof window === "undefined") return false;
    return new URLSearchParams(window.location.search).get("error") === "oauth";
  });

  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-6 p-6">
      <div className="classical-card max-w-sm w-full p-8 text-center">
        <h1 className="text-3xl mb-2">POZZ</h1>
        <p className="text-sm opacity-80 mb-6">Symulator pacjenta POZ — narzędzie treningowe, nie urządzenie medyczne.</p>
        {oauthError && (
          <p className="mb-4 text-sm text-amber-200/90" role="alert">
            Logowanie Google nie powiodło się. Spróbuj ponownie. Adres powrotu musi zawierać{" "}
            <code className="text-xs">/auth/callback</code>.
          </p>
        )}
        <button type="button" className="classical-btn classical-btn-primary w-full" onClick={() => void signInWithGoogle()}>
          Kontynuuj z Google
        </button>
        <p className="mt-4 text-xs opacity-60">Tryb deweloperski: bez Supabase UI wysyła dev-token.</p>
        <p className="mt-4 text-xs"><Link className="underline" href="/privacy">Prywatność i przetwarzanie danych</Link></p>
      </div>
    </main>
  );
}
