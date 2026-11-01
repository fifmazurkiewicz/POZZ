import * as Sentry from "@sentry/nextjs";
import { monitoringOptions } from "./lib/monitoring-options";

const options = monitoringOptions("pozz-web", {
  enabled: process.env.NEXT_PUBLIC_SENTRY_ENABLED,
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NEXT_PUBLIC_SENTRY_ENVIRONMENT,
  release: process.env.NEXT_PUBLIC_SENTRY_RELEASE,
});
if (options) {
  try {
    Sentry.init({ ...options, integrations: [Sentry.globalHandlersIntegration(), Sentry.browserApiErrorsIntegration(), Sentry.dedupeIntegration()] });
  } catch { /* Monitoring must never prevent application startup. */ }
}
