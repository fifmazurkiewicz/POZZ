"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError } from "@/lib/api";
import { uploadRecordedInterview } from "@/lib/interview/api";
import { useVoiceController } from "@/lib/voice/useVoiceController";
import type { SimulationSession } from "@/lib/simulation/api";

type Phase = "idle" | "recording" | "submitting" | "error";

type RecordedRecorderProps = {
  token: string;
  title?: string;
  onCreated: (session: SimulationSession) => void;
  onError: (message: string) => void;
  /** Optional: rendered when recording is paused to provide a "Cancel" affordance. */
  onCancel?: () => void;
};

function formatTimer(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

/**
 * Live recorder for the Wywiad "Nagranie" tab. Uses
 * {@link useVoiceController} for mic lifecycle; on stop, submits the
 * resulting `Blob` to `POST /api/interviews/recordings` (the same
 * endpoint the legacy upload form used). The blob stays in memory —
 * nothing is persisted client-side.
 *
 * Out of scope: pause/resume, waveform, streaming, audio storage.
 */
export function RecordedRecorder({ token, title, onCreated, onError, onCancel }: RecordedRecorderProps) {
  const voice = useVoiceController();
  const [phase, setPhase] = useState<Phase>("idle");
  const [elapsed, setElapsed] = useState(0);
  const startedAt = useRef<number | null>(null);

  // Local timer so the doctor sees how long they have been recording.
  useEffect(() => {
    if (phase !== "recording") return;
    startedAt.current = Date.now();
    const id = window.setInterval(() => {
      if (startedAt.current === null) return;
      setElapsed(Math.floor((Date.now() - startedAt.current) / 1000));
    }, 1000);
    return () => {
      window.clearInterval(id);
      startedAt.current = null;
    };
  }, [phase]);

  const busy = phase === "recording" || phase === "submitting";

  async function start() {
    setPhase("recording");
    try {
      await voice.startRecording(async (blob) => {
        setPhase("submitting");
        try {
          const session = await uploadRecordedInterview(token, blob, title);
          setPhase("idle");
          onCreated(session);
        } catch (error) {
          setPhase("error");
          const message =
            error instanceof ApiError
              ? error.message
              : error instanceof Error
                ? error.message
                : "Nie udało się przetworzyć nagrania.";
          onError(message);
        }
      });
    } catch {
      // useVoiceController surfaces its own error via the `error` field;
      // the UI does not need a separate banner — keep the recorder idle.
      setPhase("idle");
    }
  }

  function stop() {
    voice.finishRecording();
  }

  function cancel() {
    voice.stop();
    setPhase("idle");
    setElapsed(0);
    onCancel?.();
  }

  if (phase === "submitting") {
    return (
      <div className="classical-card flex items-center gap-3 p-3" data-testid="recorder-submitting">
        <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-[var(--color-divider)] border-t-[var(--color-accent)]" aria-hidden="true" />
        <p className="text-sm">Transkrypcja i rozpoznawanie rozmówców…</p>
      </div>
    );
  }

  if (phase === "recording") {
    return (
      <div className="classical-card flex items-center justify-between gap-3 p-3" data-testid="recorder-recording">
        <div className="flex items-center gap-3">
          <span
            className="inline-block h-3 w-3 shrink-0 rounded-full bg-red-500 motion-safe:animate-pulse"
            aria-hidden="true"
          />
          <div>
            <p className="text-sm font-semibold" role="status">Nagrywanie</p>
            <p className="text-xs text-[var(--color-soft)]" aria-live="polite">{formatTimer(elapsed)}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button type="button" className="classical-btn text-sm" onClick={cancel} aria-label="Anuluj nagrywanie">
            Anuluj
          </button>
          <button
            type="button"
            className="classical-btn classical-btn-primary text-sm"
            onClick={stop}
            aria-label="Zatrzymaj i transkrybuj"
          >
            Zatrzymaj i transkrybuj
          </button>
        </div>
      </div>
    );
  }

  return (
    <button
      type="button"
      className="classical-btn classical-btn-primary w-full"
      onClick={() => {
        void start();
      }}
      disabled={busy}
      aria-label="Rozpocznij nagrywanie"
    >
      Rozpocznij nagrywanie
    </button>
  );
}