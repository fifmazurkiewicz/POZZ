"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "@/lib/api";

export type RunAction = <T>(
  action: (signal: AbortSignal) => Promise<T>,
  success: (value: T) => void,
  fallback?: string
) => Promise<boolean>;

export type LatestError = { message: string; code?: string } | null;

/** Only the current request may update the UI, even if a transport ignores abort. */
export function useAbortableAction() {
  const controller = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cancelled, setCancelled] = useState(false);
  const [latestError, setLatestError] = useState<LatestError>(null);

  const reset = useCallback(() => {
    setError(null);
    setCancelled(false);
    setLatestError(null);
  }, []);

  const cancel = useCallback(() => {
    controller.current?.abort("user_cancelled");
    controller.current = null;
    setBusy(false);
    setError(null);
    setCancelled(true);
  }, []);

  useEffect(() => () => {
    controller.current?.abort();
    controller.current = null;
  }, []);

  const run: RunAction = useCallback(async (action, success, fallback = "Nie udało się wykonać operacji. Spróbuj ponownie.") => {
    controller.current?.abort("superseded");
    const current = new AbortController();
    controller.current = current;
    setBusy(true);
    setError(null);
    setCancelled(false);
    setLatestError(null);
    try {
      const result = await action(current.signal);
      if (controller.current !== current || current.signal.aborted) return false;
      success(result);
      return true;
    } catch (err) {
      if (controller.current !== current) return false;
      // User pressed Cancel → no error surface.
      if (current.signal.aborted) {
        setCancelled(true);
        return false;
      }
      if (err instanceof ApiError) {
        setError(err.message);
        setLatestError({ message: err.message, code: err.code });
      } else {
        setError(fallback);
        setLatestError({ message: fallback });
      }
      return false;
    } finally {
      if (controller.current === current) {
        controller.current = null;
        setBusy(false);
      }
    }
  }, []);

  const getError = useCallback(() => latestError, [latestError]);

  return { busy, error, cancelled, run, cancel, reset, getError, setError };
}