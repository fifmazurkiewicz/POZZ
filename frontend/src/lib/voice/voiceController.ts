import { apiUrl } from "../api";
import { readVoiceId } from "./voicePreference";

type VoiceControllerCallbacks = {
  onSpeakingChange: (speaking: boolean) => void;
  onListeningChange: (listening: boolean) => void;
  onError: (message: string) => void;
};

type ApiErrorBody = {
  code?: string;
  message?: string;
  detail?: string | { code?: string; message?: string };
};

export class VoiceController {
  private disposed = false;
  private operation = 0;
  private abortController: AbortController | null = null;
  private audio: HTMLAudioElement | null = null;
  private audioUrl: string | null = null;
  private playbackDone: (() => void) | null = null;
  private stream: MediaStream | null = null;
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private recordingDone: ((blob: Blob) => void) | null = null;
  private submitRecording = false;

  constructor(private readonly callbacks: VoiceControllerCallbacks) {}

  async play(text: string, token: string): Promise<void> {
    this.stop();
    if (this.disposed) return;
    const operation = this.operation;
    this.speaking(true);
    const controller = new AbortController();
    this.abortController = controller;

    try {
      const voiceId = readVoiceId();
      const response = await fetch(apiUrl("/api/voice/speech"), {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice_id: voiceId || undefined }),
        signal: controller.signal,
      });
      if (!this.current(operation)) return;

      if (!response.ok) {
        const error = await readApiError(response);
        if (!this.current(operation)) return;
        if (response.status === 503 && error.code === "tts_unavailable") {
          await this.playBrowserSpeech(text, operation);
          return;
        }
        throw new Error(error.message || "Nie udało się odtworzyć głosu pacjenta.");
      }

      const blob = await response.blob();
      if (!this.current(operation)) return;
      const url = URL.createObjectURL(blob);
      if (!this.current(operation)) {
        URL.revokeObjectURL(url);
        return;
      }
      this.audioUrl = url;
      const audio = new Audio(url);
      this.audio = audio;
      await new Promise<void>((resolve, reject) => {
        let settled = false;
        const finish = (error?: Error) => {
          if (settled) return;
          settled = true;
          this.playbackDone = null;
          if (error) reject(error);
          else resolve();
        };
        this.playbackDone = () => finish();
        audio.onended = () => finish();
        audio.onerror = () => finish(new Error("Nie udało się odtworzyć głosu pacjenta."));
        Promise.resolve(audio.play()).catch((error) => finish(error instanceof Error ? error : new Error(String(error))));
      });
    } catch (error) {
      if (this.current(operation) && !isAbortError(error)) {
        this.error(error instanceof Error ? error.message : "Nie udało się odtworzyć głosu pacjenta.");
      }
    } finally {
      if (this.current(operation)) {
        this.abortController = null;
        this.cleanupPlayback();
        this.speaking(false);
      }
    }
  }

  async startRecording(onBlob: (blob: Blob) => void): Promise<void> {
    this.stop();
    if (this.disposed) return;
    const operation = this.operation;
    const mediaDevices = globalThis.navigator?.mediaDevices;
    const Recorder = globalThis.MediaRecorder;
    if (!mediaDevices?.getUserMedia || !Recorder) {
      this.error("Ta przeglądarka nie obsługuje nagrywania głosu.");
      return;
    }
    try {
      const stream = await mediaDevices.getUserMedia({ audio: true });
      if (!this.current(operation)) {
        stopTracks(stream);
        return;
      }
      this.stream = stream;
      const recorder = new Recorder(stream);
      this.recorder = recorder;
      this.chunks = [];
      this.recordingDone = onBlob;
      this.submitRecording = false;
      recorder.ondataavailable = (event) => {
        if (this.current(operation) && event.data.size) this.chunks.push(event.data);
      };
      recorder.onstop = () => {
        const shouldSubmit = this.current(operation) && this.submitRecording;
        const callback = this.recordingDone;
        const blob = new Blob(this.chunks, { type: recorder.mimeType || "audio/webm" });
        this.cleanupRecording();
        if (shouldSubmit) {
          this.listening(false);
          if (blob.size) callback?.(blob);
        }
      };
      recorder.start();
      this.listening(true);
    } catch {
      if (this.current(operation)) {
        this.cleanupRecording();
        this.error("Brak dostępu do mikrofonu.");
      }
    }
  }

  finishRecording(): void {
    if (this.disposed || !this.recorder) return;
    this.submitRecording = true;
    if (this.recorder.state !== "inactive") this.recorder.stop();
  }

  stop(): void {
    this.operation += 1;
    this.abortController?.abort();
    this.abortController = null;
    this.cleanupPlayback();
    const wasRecording = this.recorder !== null || this.stream !== null;
    this.submitRecording = false;
    const recorder = this.recorder;
    this.recorder = null;
    this.recordingDone = null;
    this.chunks = [];
    if (recorder && recorder.state !== "inactive") recorder.stop();
    this.cleanupRecording();
    if (!this.disposed) {
      this.speaking(false);
      if (wasRecording) this.listening(false);
    }
    if (typeof window !== "undefined") window.speechSynthesis?.cancel();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.stop();
  }

  private async playBrowserSpeech(text: string, operation: number): Promise<void> {
    if (typeof window === "undefined" || !window.speechSynthesis || typeof SpeechSynthesisUtterance === "undefined") {
      throw new Error("Ta przeglądarka nie obsługuje syntezy mowy.");
    }
    await new Promise<void>((resolve, reject) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "pl-PL";
      this.playbackDone = resolve;
      utterance.onend = () => resolve();
      utterance.onerror = () => reject(new Error("Nie udało się odtworzyć głosu pacjenta."));
      if (this.current(operation)) window.speechSynthesis.speak(utterance);
      else resolve();
    });
  }

  private current(operation: number): boolean {
    return !this.disposed && operation === this.operation;
  }

  private cleanupPlayback(): void {
    const done = this.playbackDone;
    this.playbackDone = null;
    this.audio?.pause();
    if (this.audio) this.audio.src = "";
    this.audio = null;
    if (this.audioUrl) URL.revokeObjectURL(this.audioUrl);
    this.audioUrl = null;
    done?.();
  }

  private cleanupRecording(): void {
    stopTracks(this.stream);
    this.stream = null;
    this.recorder = null;
    this.recordingDone = null;
    this.chunks = [];
    this.submitRecording = false;
  }

  private speaking(value: boolean): void {
    if (!this.disposed) this.callbacks.onSpeakingChange(value);
  }

  private listening(value: boolean): void {
    if (!this.disposed) this.callbacks.onListeningChange(value);
  }

  private error(message: string): void {
    if (!this.disposed) this.callbacks.onError(message);
  }
}

function stopTracks(stream: MediaStream | null): void {
  stream?.getTracks().forEach((track) => track.stop());
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

async function readApiError(response: Response): Promise<{ code?: string; message?: string }> {
  try {
    const body = (await response.json()) as ApiErrorBody;
    if (body.detail && typeof body.detail === "object") return body.detail;
    if (typeof body.detail === "string") return { message: body.detail };
    return { code: body.code, message: body.message };
  } catch {
    return { message: response.statusText };
  }
}
