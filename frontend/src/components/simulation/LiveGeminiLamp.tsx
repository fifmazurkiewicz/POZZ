"use client";

import { voiceDotFillClass } from "@/components/simulation/voiceDotFill";

type Props = {
  on: boolean;
  disabled?: boolean;
  onToggle: () => void;
};

/** Status-row lamp: lit = Live Gemini, unlit = TTS-only path. */
export function LiveGeminiLamp({ on, disabled, onToggle }: Props) {
  return (
    <button
      type="button"
      className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent)] disabled:opacity-50"
      aria-label="Live Gemini"
      aria-pressed={on}
      disabled={disabled}
      onClick={onToggle}
      title={on ? "Live Gemini" : "Tylko TTS"}
    >
      <span
        className={`h-3.5 w-3.5 rounded-full ring-1 ring-[var(--color-divider)] ${voiceDotFillClass(on)}${
          on ? " shadow-[0_0_8px_color-mix(in_srgb,var(--color-accent)_55%,transparent)]" : ""
        }`}
      />
    </button>
  );
}
