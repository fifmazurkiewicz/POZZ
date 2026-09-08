"use client";

import { useEffect, useRef } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { APPROVAL_POLL_MS, resolveRedirect } from "@/lib/auth/routePolicy";

function Splash({ label, action }: { label: string; action?: React.ReactNode }) {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-4 p-6">
      <p className="font-serif text-lg text-[var(--color-soft)]">{label}</p>
      {action}
    </main>
  );
}

function PendingApprovalScreen() {
  const { refreshProfile, signOut } = useAuth();

  useEffect(() => {
    const id = window.setInterval(() => {
      void refreshProfile();
    }, APPROVAL_POLL_MS);
    return () => window.clearInterval(id);
  }, [refreshProfile]);

  return (
    <main className="flex flex-1 flex-col items-center justify-center p-6">
      <div className="classical-card w-full max-w-sm space-y-4 p-6">
        <h1 className="font-serif text-2xl text-[var(--color-text)]">Konto oczekuje na akceptację</h1>
        <p className="text-sm text-[var(--color-soft)]">
          Administrator musi zaakceptować to konto, zanim będzie można korzystać z POZZ.
        </p>
        <div className="flex flex-col gap-2">
          <button type="button" className="classical-btn classical-btn-primary" onClick={() => void refreshProfile()}>
            Sprawdź status
          </button>
          <button type="button" className="classical-btn" onClick={() => void signOut()}>
            Wyloguj
          </button>
        </div>
      </div>
    </main>
  );
}

export function AuthGate({ children }: { children: React.ReactNode }) {
  const { status, refreshProfile } = useAuth();
  const pathname = usePathname();
  const router = useRouter();
  const redirectTo = resolveRedirect(status, pathname);
  const navigatedRef = useRef<string | null>(null);

  useEffect(() => {
    if (!redirectTo) {
      navigatedRef.current = null;
      return;
    }
    if (navigatedRef.current === redirectTo) return;
    navigatedRef.current = redirectTo;
    router.replace(redirectTo);
  }, [redirectTo, router]);

  if (redirectTo) return <Splash label="Przekierowanie…" />;

  if (status === "initializing") return <Splash label="Wczytywanie POZZ…" />;

  if (status === "profile_unknown") {
    return (
      <Splash
        label="Uruchamianie API…"
        action={
          <button type="button" className="classical-btn" onClick={() => void refreshProfile()}>
            Spróbuj teraz
          </button>
        }
      />
    );
  }

  if (status === "pending_approval") {
    return <PendingApprovalScreen />;
  }

  return <>{children}</>;
}
