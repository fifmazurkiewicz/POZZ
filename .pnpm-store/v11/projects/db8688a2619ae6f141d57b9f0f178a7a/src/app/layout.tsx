import type { Metadata, Viewport } from "next";
import "./globals.css";
import { ApiPulseBanner } from "@/components/ApiPulseBanner";
import { ApiPulseProvider } from "@/components/ApiPulseProvider";
import { AppBottomNav } from "@/components/AppBottomNav";
import { AuthGate } from "@/components/AuthGate";
import { AuthProvider } from "@/components/AuthProvider";
import { ViewportFrame } from "@/components/ViewportFrame";

export const metadata: Metadata = {
  title: "POZZ",
  description: "Symulator pacjenta POZ",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "POZZ",
  },
};

export const viewport: Viewport = {
  themeColor: "#16130f",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  interactiveWidget: "resizes-content",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pl" className="h-full dark" suppressHydrationWarning>
      <body className="flex h-dvh min-h-0 flex-col overflow-hidden bg-[var(--color-bg)] text-[var(--color-text)]">
        <ApiPulseProvider>
          <AuthProvider>
            <ApiPulseBanner />
            <AuthGate>
              <ViewportFrame>{children}</ViewportFrame>
            </AuthGate>
            <AppBottomNav />
          </AuthProvider>
        </ApiPulseProvider>
      </body>
    </html>
  );
}
