import Link from "next/link";

export default function PrivacyPage() {
  return (
    <main className="app-page flex-1 overflow-y-auto">
      <p className="text-sm font-semibold text-[var(--color-accent)]">POZZ</p>
      <h1 className="mt-1 text-3xl">Prywatność i AI</h1>
      <p className="mt-3 rounded border border-amber-400/40 p-3 text-sm text-amber-100">
        Wersja robocza do testów lokalnych. Nie jest jeszcze zatwierdzoną polityką prywatności do publikacji.
      </p>
      <section className="mt-6 max-w-3xl space-y-3 text-sm text-[var(--color-soft)]">
        <h2 className="text-xl text-[var(--color-text)]">Jak działa POZZ</h2>
        <p>POZZ jest symulatorem szkoleniowym, a nie urządzeniem medycznym. Pacjent, odpowiedzi, wyniki badań i oceny mogą być generowane przez AI i mogą zawierać błędy.</p>
        <h2 className="pt-3 text-xl text-[var(--color-text)]">Jakie dane są przetwarzane</h2>
        <p>Konto może obejmować adres e-mail i nazwę profilu. Zapisywane mogą być rozmowy, opisy przypadków, plany postępowania, oceny oraz informacje o wykorzystaniu usługi.</p>
        <p>Nagranie krótkiej wypowiedzi jest wysyłane do dostawcy transkrypcji i przetwarzane w pamięci; POZZ nie zapisuje pliku audio w bazie. Powstały tekst może zostać zapisany jako część rozmowy.</p>
        <h2 className="pt-3 text-xl text-[var(--color-text)]">Dostawcy</h2>
        <p>Aplikacja korzysta docelowo z Supabase, Render i Vercel oraz może przekazywać niezbędną treść do OpenRouter, Groq, ElevenLabs i Langfuse. Dokładne regiony, okresy przechowywania i mechanizmy transferu muszą zostać zatwierdzone przed produkcją.</p>
        <h2 className="pt-3 text-xl text-[var(--color-text)]">Nie wpisuj danych prawdziwego pacjenta</h2>
        <p>Używaj wyłącznie danych fikcyjnych albo skutecznie zanonimizowanych. Nie wpisuj imion, nazwisk, numerów PESEL, adresów, danych kontaktowych, numerów dokumentacji ani wyjątkowych kombinacji informacji pozwalających rozpoznać osobę.</p>
        <h2 className="pt-3 text-xl text-[var(--color-text)]">Twoje dane</h2>
        <p>Po zalogowaniu możesz pobrać kopię danych aplikacji albo usunąć swoją treść w sekcji Prywatność w Menu. Usunięcie treści nie usuwa jeszcze konta uwierzytelniania ani kopii zapasowych.</p>
      </section>
      <Link className="classical-btn mt-8 inline-flex" href="/login">Wróć</Link>
    </main>
  );
}
