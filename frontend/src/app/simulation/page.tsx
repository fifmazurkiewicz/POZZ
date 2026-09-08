import { SimulationStatusRow } from "@/components/simulation/SimulationStatusRow";

export default function SimulationPage() {
  return (
    <main className="flex min-h-0 flex-1 flex-col">
      <header className="flex h-11 shrink-0 items-center justify-between border-b border-[var(--color-divider)] px-4">
        <h1 className="text-lg">Symulacja</h1>
        <button type="button" className="classical-btn text-sm" disabled>
          Następny pacjent
        </button>
      </header>
      <SimulationStatusRow />
      <section className="min-h-0 flex-1 overflow-y-auto px-4 py-6 text-sm text-[var(--color-soft)]">
        <p>Najpierw wygeneruj pacjenta, aby móc rozpocząć wywiad.</p>
        <p className="mt-3">
          Lampa: ON = Gemini Live, OFF = TTS. Głos i transkrypt pojawią się w kolejnym pakiecie.
        </p>
      </section>
      <div className="shrink-0 border-t border-[var(--color-divider)] p-3">
        <label className="sr-only" htmlFor="sim-composer">
          Wiadomość
        </label>
        <input
          id="sim-composer"
          className="min-h-11 w-full rounded border border-[var(--color-divider)] bg-[var(--color-bg)] px-3"
          placeholder="Zadaj pytanie pacjentowi…"
          disabled
        />
      </div>
    </main>
  );
}
