"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  fetchApiPulse,
  PULSE_INTERVAL_FAST_MS,
  PULSE_INTERVAL_SLOW_MS,
  type PulseStatus,
} from "@/lib/api/pulse";

type ApiPulseContextValue = {
  status: PulseStatus;
  isHealthy: boolean;
  isWaking: boolean;
  lastCheckedAt: number | null;
  checkNow: () => Promise<boolean>;
};

const ApiPulseContext = createContext<ApiPulseContextValue | null>(null);

export function ApiPulseProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<PulseStatus>("unknown");
  const [lastCheckedAt, setLastCheckedAt] = useState<number | null>(null);
  const runningRef = useRef(false);

  const checkNow = useCallback(async () => {
    if (runningRef.current) return status === "healthy";
    runningRef.current = true;
    const ok = await fetchApiPulse();
    setStatus(ok ? "healthy" : "unhealthy");
    setLastCheckedAt(Date.now());
    runningRef.current = false;
    return ok;
  }, [status]);

  useEffect(() => {
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    async function tick() {
      if (cancelled) return;
      const ok = await fetchApiPulse();
      if (cancelled) return;
      setStatus(ok ? "healthy" : "unhealthy");
      setLastCheckedAt(Date.now());
      const delay = ok ? PULSE_INTERVAL_SLOW_MS : PULSE_INTERVAL_FAST_MS;
      timeoutId = setTimeout(tick, delay);
    }

    void tick();
    return () => {
      cancelled = true;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, []);

  const value = useMemo(
    () => ({
      status,
      isHealthy: status === "healthy",
      isWaking: status !== "healthy",
      lastCheckedAt,
      checkNow,
    }),
    [status, lastCheckedAt, checkNow]
  );

  return <ApiPulseContext.Provider value={value}>{children}</ApiPulseContext.Provider>;
}

export function useApiPulse(): ApiPulseContextValue {
  const ctx = useContext(ApiPulseContext);
  if (!ctx) throw new Error("useApiPulse must be used within ApiPulseProvider");
  return ctx;
}
