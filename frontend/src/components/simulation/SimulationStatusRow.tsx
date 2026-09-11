"use client";

import { useEffect, useState } from "react";
import { LiveGeminiLamp } from "@/components/simulation/LiveGeminiLamp";
import { voiceDotFillClass } from "@/components/simulation/voiceDotFill";
import {
  readLiveGeminiPreference,
  writeLiveGeminiPreference,
} from "@/lib/voice/liveGeminiPreference";

type VoiceConfig = {
  live_available: boolean;
};

type Props = {
  patientVoice: boolean;
  listening: boolean;
  disabled?: boolean;
  onPatientVoiceChange: (enabled: boolean) => void;
  onListeningChange: (enabled: boolean) => void;
};

export function SimulationStatusRow({ patientVoice, listening, disabled, onPatientVoiceChange, onListeningChange }: Props) {
  const [liveGemini, setLiveGemini] = useState(() => readLiveGeminiPreference());
  const [liveAvailable, setLiveAvailable] = useState(true);

  useEffect(() => {
    const api = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    void fetch(`${api}/api/voice/config`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((body: VoiceConfig | null) => {
        if (body && body.live_available === false) {
          setLiveAvailable(false);
          setLiveGemini(false);
        }
      })
      .catch(() => {
        /* ApiPulse covers wake; lamp stays on default */
      });
  }, []);

  function toggleLamp() {
    if (!liveAvailable) return;
    const next = !liveGemini;
    setLiveGemini(next);
    writeLiveGeminiPreference(next);
  }

  return (
    <div className="flex h-11 shrink-0 items-center justify-between border-b border-[var(--color-divider)] px-2">
      <LiveGeminiLamp on={liveGemini && liveAvailable} disabled={!liveAvailable} onToggle={toggleLamp} />
      <p className="text-sm text-[var(--color-soft)]">{listening ? "Słuchanie" : "Gotowe"}</p>
      <div className="flex items-center gap-2 pr-1">
        <button
          type="button"
          className="flex h-11 w-11 items-center justify-center"
          aria-label="Głos pacjenta"
          aria-pressed={patientVoice}
          disabled={disabled}
          onClick={() => onPatientVoiceChange(!patientVoice)}
          title="Głos pacjenta"
        >
          <span className={`h-2.5 w-2.5 rounded-full ${voiceDotFillClass(patientVoice)}`} />
        </button>
        <button
          type="button"
          className="flex h-11 w-11 items-center justify-center"
          aria-label="Słuchanie"
          aria-pressed={listening}
          disabled={disabled}
          onClick={() => onListeningChange(!listening)}
          title="Słuchanie"
        >
          <span className={`h-2.5 w-2.5 rounded-full ${voiceDotFillClass(listening)}`} />
        </button>
      </div>
    </div>
  );
}
