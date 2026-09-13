import Link from "next/link";

export function MenuBackLink() {
  return (
    <Link
      className="classical-btn mb-4 inline-flex min-h-11 items-center gap-2 text-sm"
      href="/menu"
      aria-label="Wróć do menu"
    >
      <span aria-hidden="true">←</span>
      Wróć do menu
    </Link>
  );
}
