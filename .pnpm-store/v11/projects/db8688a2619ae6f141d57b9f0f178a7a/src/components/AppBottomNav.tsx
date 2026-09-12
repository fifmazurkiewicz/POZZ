"use client";

import { useAuth } from "@/components/AuthProvider";
import { BottomNav } from "@/components/BottomNav";

export function AppBottomNav() {
  const { status } = useAuth();
  if (status !== "ready") return null;
  return <BottomNav />;
}
