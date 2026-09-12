"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "@/lib/api";

export type RunAction = <T>(action: (signal: AbortSignal) => Promise<T>, success: (value: T) => void, fallback?: string) => Promise<boolean>;

/** Only the current request may update the UI, even if a transport ignores abort. */
export function useAbortableAction() {
  const controller = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cancel = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
    setBusy(false);
    setError(null);
  }, []);
  useEffect(() => () => {
    controller.current?.abort();
    controller.current = null;
  }, []);
  const run: RunAction = useCallback(async (action, success, fallback = "Nie udało się wykonać operacji. Spróbuj ponownie.") => {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setBusy(true);
    setError(null);
    try {
      const result = await action(current.signal);
      if (controller.current !== current || current.signal.aborted) return false;
      success(result);
      return true;
    } catch (err) {
      if (controller.current === current && !current.signal.aborted) {
        setError(err instanceof ApiError ? err.message : fallback);
      }
      return false;
    } finally {
      if (controller.current === current) {
        controller.current = null;
        setBusy(false);
      }
    }
  }, []);
  return { busy, error, run, cancel, setError };
}
