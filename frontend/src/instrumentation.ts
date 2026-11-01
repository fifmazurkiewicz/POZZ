import type { Instrumentation } from "next";
import * as Sentry from "@sentry/nextjs";

export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs" || process.env.NEXT_RUNTIME === "edge") {
    await import("./sentry.server.config");
  }
}

export const onRequestError: Instrumentation.onRequestError = async (error, request, context) => {
  try { Sentry.captureRequestError(error, request, context); await Sentry.flush(2000); } catch { /* Preserve the original failure. */ }
};
