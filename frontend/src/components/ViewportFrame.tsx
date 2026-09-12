"use client";

import { usePathname } from "next/navigation";

export function ViewportFrame({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const conversation = path === "/simulation" || path === "/interview";
  return <div className={`flex min-h-0 flex-1 flex-col ${conversation ? "overflow-hidden" : "overflow-y-auto"}`}>{children}</div>;
}
