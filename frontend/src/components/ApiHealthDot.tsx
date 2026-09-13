"use client";

import { useApiPulse } from "@/components/ApiPulseProvider";

const STATUS_COPY = {
  healthy: "API działa prawidłowo",
  unhealthy: "API jest niedostępne",
  checking: "Sprawdzanie API",
  unknown: "Sprawdzanie API",
} as const;

export function ApiHealthDot() {
  const { status } = useApiPulse();
  const color =
    status === "healthy"
      ? "bg-emerald-400"
      : status === "unhealthy"
        ? "bg-red-400"
        : "bg-amber-300";

  return (
    <span
      className="absolute right-3 top-1/2 inline-flex -translate-y-1/2 items-center"
      role="status"
      aria-label={STATUS_COPY[status]}
      title={STATUS_COPY[status]}
    >
      <span
        className={`h-2.5 w-2.5 rounded-full border border-black/20 shadow-[0_0_0_2px_color-mix(in_srgb,var(--color-surface)_75%,transparent)] ${color}`}
        aria-hidden="true"
      />
    </span>
  );
}
