import { afterEach, describe, expect, it } from "vitest";
import * as Sentry from "@sentry/nextjs";
import { monitoringOptions } from "./monitoring-options";
import { onRequestError } from "../instrumentation";

type Envelope = [Record<string, unknown>, Array<[{ type: string }, unknown]>];

const dsn = "https://0123456789abcdef0123456789abcdef@o1.ingest.de.sentry.io/123";
const privateData = "PRIVATE-CONTENT-123 patient transcript prompt token /Users/Private/file.txt";
afterEach(async () => { await Sentry.close(100); });

describe("Sentry privacy and activation", () => {
  it.each([
    { enabled: "false", dsn }, { enabled: "", dsn }, { enabled: "true", dsn: "" },
    { enabled: "true", dsn: "invalid" }, { enabled: "true", dsn: "https://private@example.org/1" },
  ])("does not initialize for disabled or invalid configuration: %j", (config) => {
    expect(monitoringOptions("pozz-web", config)).toBeUndefined();
  });

  it("emits one real SDK envelope with technical fields only, including when attachments exist", async () => {
    const envelopes: Envelope[] = [];
    const options = monitoringOptions("pozz-web", { enabled: "true", dsn, environment: "test" });
    expect(options).toBeDefined();
    Sentry.init({ ...options, transport: () => ({
      send: (envelope: Envelope) => { envelopes.push(envelope); return Promise.resolve({ statusCode: 200 }); },
      flush: () => Promise.resolve(true),
    }) });
    Sentry.setUser({ email: privateData });
    Sentry.setContext("private", { nested: { content: privateData } });
    Sentry.setExtra("private", privateData);
    Sentry.addBreadcrumb({ message: privateData });
    Sentry.getCurrentScope().addAttachment({ filename: "private.txt", data: privateData });
    const error = new TypeError(privateData);
    Sentry.captureException(error);
    Sentry.captureException(error);
    Sentry.captureMessage(privateData);
    await Sentry.flush(100);
    expect(envelopes).toHaveLength(1);
    expect(envelopes[0][1].map(([header]) => header.type)).toEqual(["event"]);
    const serialized = JSON.stringify(envelopes);
    expect(serialized).not.toContain(privateData);
    expect(serialized).not.toContain("/Users/");
    const event = envelopes[0][1][0][1] as Record<string, unknown>;
    expect(event.tags).toEqual({ component: "pozz-web" });
    expect(event.exception).toMatchObject({ values: [{ type: "TypeError", value: "Application error; private details removed" }] });
    for (const field of ["request", "user", "contexts", "extra", "breadcrumbs", "transaction"]) {
      expect(event).not.toHaveProperty(field);
    }
  });

  it("removes private stack URLs, variables and messages while retaining bundle positions", () => {
    const options = monitoringOptions("pozz-web", { enabled: "true", dsn, environment: privateData, release: privateData })!;
    const event = options.beforeSend({ type: undefined, exception: { values: [{ type: privateData, value: privateData, stacktrace: { frames: [
      { filename: "https://private.example/assets/main-123.js?token=" + privateData, lineno: 5, colno: 2, vars: { privateData } },
      { filename: "/Users/Private/file.txt", function: privateData },
    ] } }] } }, { originalException: new Error(privateData) });
    expect(JSON.stringify(event)).not.toContain(privateData);
    expect(JSON.stringify(event)).not.toContain("private.example");
    expect(event?.exception?.values?.[0]?.stacktrace?.frames).toEqual([{ filename: "/assets/main-123.js", lineno: 5, colno: 2 }]);
    expect(event?.environment).toBe("unknown");
    expect(event).not.toHaveProperty("release");
  });

  it("retains sanitized Debug IDs in the real envelope for private source maps", async () => {
    const envelopes: Envelope[] = [];
    Sentry.init({ ...monitoringOptions("pozz-web", { enabled: "true", dsn }), transport: () => ({
      send: (envelope: Envelope) => { envelopes.push(envelope); return Promise.resolve({ statusCode: 200 }); },
      flush: () => Promise.resolve(true),
    }) });
    Sentry.captureEvent({ type: undefined, exception: { values: [{ type: "Error", value: privateData }] },
      debug_meta: { images: [{ type: "sourcemap", code_file: "https://private.example/_next/static/chunks/app.js?secret=" + privateData, debug_id: "12345678-1234-1234-1234-123456789abc" }] },
    }, { originalException: new Error(privateData) });
    await Sentry.flush(100);
    expect(envelopes).toHaveLength(1);
    expect(envelopes[0][1][0][1]).toMatchObject({ debug_meta: { images: [{ type: "sourcemap", code_file: "/_next/static/chunks/app.js", debug_id: "12345678-1234-1234-1234-123456789abc" }] } });
    expect(JSON.stringify(envelopes)).not.toContain(privateData);
  });

  it("awaits bounded delivery before the Next.js request-error hook completes", async () => {
    const envelopes: Envelope[] = [];
    let flushed = false;
    Sentry.init({ ...monitoringOptions("pozz-web", { enabled: "true", dsn }), transport: () => ({
      send: (envelope: Envelope) => { envelopes.push(envelope); return Promise.resolve({ statusCode: 200 }); },
      flush: async () => { await Promise.resolve(); flushed = true; return true; },
    }) });
    await onRequestError(new Error(privateData), { path: "/private?token=" + privateData, method: "POST", headers: { authorization: privateData } }, { routerKind: "App Router", routePath: "/private", routeType: "route", revalidateReason: undefined });
    expect(flushed).toBe(true);
    expect(envelopes).toHaveLength(1);
    expect(JSON.stringify(envelopes)).not.toContain(privateData);
  });

  it("drops expected cancellation errors", async () => {
    const envelopes: Envelope[] = [];
    Sentry.init({ ...monitoringOptions("pozz-web", { enabled: "true", dsn }), transport: () => ({
      send: (envelope: Envelope) => { envelopes.push(envelope); return Promise.resolve({ statusCode: 200 }); },
      flush: () => Promise.resolve(true),
    }) });
    Sentry.captureException(new DOMException(privateData, "AbortError"));
    await Sentry.flush(100);
    expect(envelopes).toEqual([]);
  });
});
