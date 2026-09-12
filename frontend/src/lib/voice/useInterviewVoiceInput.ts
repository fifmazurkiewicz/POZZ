"use client";

import { useCallback } from "react";
import { useAbortableAction } from "../useAbortableAction";
import { useVoiceController } from "./useVoiceController";
import { transcribeAudio } from "./transcribeAudio";

export type InterviewVoiceInputOptions = {
  token: string;
  appendTranscript: (text: string) => void;
};

export type InterviewVoiceInput = {
  listening: boolean;
  speaking: boolean;
  busy: boolean;
  error: string | null;
  toggle: () => void;
  stop: () => void;
};

/**
 * Mic-only speech-to-text input for the Wywiad (Interview) manual case
 * composer. Reuses the shared {@link useVoiceController} for microphone
 * lifecycle and calls `POST /api/voice/transcribe`; the recognised text is
 * handed off via `appendTranscript` (typically appended to the existing
 * draft input — the doctor reviews / edits before tapping "Wyślij").
 *
 * No patient TTS, no Live lamp — see ADR
 * `docs/technical/decisions/2026-09-13-interview-voice-input.md`.
 */
export function useInterviewVoiceInput(options: InterviewVoiceInputOptions): InterviewVoiceInput {
  const voice = useVoiceController();
  const { token, appendTranscript } = options;
  const { run, busy, error: actionError, reset } = useAbortableAction();

  const toggle = useCallback(() => {
    if (voice.listening) {
      // Finalize the current recording — VoiceController will call onBlob,
      // which we set up below to submit to STT.
      voice.finishRecording();
      return;
    }
    reset();
    void voice.startRecording((blob) => {
      void run(
        async (signal) => transcribeAudio(token, blob, signal),
        (result) => {
          const text = result.text.trim();
          if (text) appendTranscript(text);
        }
      );
    });
  }, [voice, run, token, appendTranscript, reset]);

  // The action's error wins when present; cancellation is silent (we
  // simply did not append anything).
  const error = actionError ?? voice.error ?? null;

  return {
    listening: voice.listening,
    speaking: voice.speaking,
    busy,
    error,
    toggle,
    stop: voice.stop,
  };
}