"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { VoiceController } from "./voiceController";

export function useVoiceController() {
  const controller = useRef<VoiceController | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [listening, setListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const voice = new VoiceController({ onSpeakingChange: setSpeaking, onListeningChange: setListening, onError: setError });
    controller.current = voice;
    return () => { voice.dispose(); controller.current = null; };
  }, []);
  const stop = useCallback(() => { controller.current?.stop(); setError(null); }, []);
  const play = useCallback((text: string, token: string) => controller.current?.play(text, token), []);
  const startRecording = useCallback((onBlob: (blob: Blob) => void) => { setError(null); return controller.current?.startRecording(onBlob); }, []);
  const finishRecording = useCallback(() => controller.current?.finishRecording(), []);
  const startConversationTurn = useCallback((token: string, onComplete: (text: string) => void) => { setError(null); return controller.current?.startConversationTurn(token, onComplete); }, []);
  const stopConversationTurn = useCallback(() => { controller.current?.stopConversationTurn(); setError(null); }, []);
  return { speaking, listening, error, stop, play, startRecording, finishRecording, startConversationTurn, stopConversationTurn };
}
