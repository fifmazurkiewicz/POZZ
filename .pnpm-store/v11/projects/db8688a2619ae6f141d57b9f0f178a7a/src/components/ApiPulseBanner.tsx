"use client";

import { useApiPulse } from "@/components/ApiPulseProvider";

export function ApiPulseBanner() {
  const { isWaking, status, checkNow } = useApiPulse();

  if (!isWaking) return null;

  return (
    <div
      role="status"
      className="border-b border-[var(--color-accent)] bg-[color-mix(in_srgb,var(--color-accent)_10%,var(--color-bg))] px-4 py-2 text-center text-sm"
    >
      {status === "checking" || status === "unknown" ? (
        <span>Uruchamianie API…</span>
      ) : (
        <span className="inline-flex flex-wrap items-center justify-center gap-2">
          API niedostępne — akcje mogą nie działać, dopóki serwer nie wstanie.
          <button type="button" className="classical-btn px-2 py-1 text-xs" onClick={() => void checkNow()}>
            Spróbuj teraz
          </button>
        </span>
      )}
    </div>
  );
}
