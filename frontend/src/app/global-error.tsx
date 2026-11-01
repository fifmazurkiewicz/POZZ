"use client";

import { useEffect } from "react";
import NextError from "next/error";
import * as Sentry from "@sentry/nextjs";

export default function GlobalError({ error }: { error: Error & { digest?: string } }) {
  useEffect(() => { try { Sentry.captureException(error); } catch { /* Preserve fallback UI. */ } }, [error]);
  return <html><body><NextError statusCode={0} /></body></html>;
}
