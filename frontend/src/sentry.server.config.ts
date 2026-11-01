import * as Sentry from "@sentry/nextjs";
import { monitoringOptions } from "./lib/monitoring-options";

const options = monitoringOptions("pozz-web", {
  enabled: process.env.SENTRY_ENABLED,
  dsn: process.env.SENTRY_DSN,
  environment: process.env.SENTRY_ENVIRONMENT,
  release: process.env.SENTRY_RELEASE || process.env.VERCEL_GIT_COMMIT_SHA,
});
if (options) {
  try {
    Sentry.init({ ...options, enableOpenTelemetrySetup: false });
  } catch { /* Monitoring must never prevent application startup. */ }
}
