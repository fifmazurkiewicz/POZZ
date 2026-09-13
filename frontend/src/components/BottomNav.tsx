"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ApiHealthDot } from "@/components/ApiHealthDot";

const TABS = [
  { href: "/simulation", label: "Symulacja" },
  { href: "/interview", label: "Wywiad" },
  { href: "/menu", label: "Menu" },
] as const;

export function BottomNav() {
  const pathname = usePathname();

  return (
    <nav
      className="relative flex shrink-0 border-t border-[var(--color-divider)] bg-[var(--color-surface)] pb-[env(safe-area-inset-bottom)]"
      aria-label="Główne"
    >
      {TABS.map((tab) => {
        const active = pathname === tab.href || pathname.startsWith(`${tab.href}/`) || (tab.href === "/simulation" && pathname === "/");
        return (
          <Link
            key={tab.href}
            href={tab.href}
            className={`nav-tab-link flex min-h-11 flex-1 items-center justify-center text-sm ${
              active ? "text-[var(--color-accent)]" : "text-[var(--color-soft)]"
            }`}
            aria-current={active ? "page" : undefined}
          >
            {tab.label}
          </Link>
        );
      })}
      <ApiHealthDot />
    </nav>
  );
}
