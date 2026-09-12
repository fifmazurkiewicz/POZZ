import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, apiUrl: (path: string) => `http://localhost:8000${path}` };
});

import { ApiError } from "../api";
import { transcribeAudio } from "./transcribeAudio";

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
    json: async () => body,
  } as unknown as Response;
}

describe("transcribeAudio", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn()));
  afterEach(() => vi.unstubAllGlobals());

  it("POSTs multipart with audio blob and bearer token", async () => {
    const fetchMock = vi.mocked(fetch).mockResolvedValue(jsonResponse(200, { text: "Ból w klatce" }));
    const blob = new Blob(["audio"], { type: "audio/webm" });
    const result = await transcribeAudio("token-xyz", blob);

    expect(result).toEqual({ text: "Ból w klatce" });
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://localhost:8000/api/voice/transcribe");
    expect(init!.method).toBe("POST");
    expect(init!.headers).toEqual({ Authorization: "Bearer token-xyz" });
    expect(init!.body).toBeInstanceOf(FormData);
    const form = init!.body as FormData;
    const file = form.get("audio");
    expect(file).toBeInstanceOf(Blob);
  });

  it("returns the recognised text on 200", async () => {
    vi.mocked(fetch).mockResolvedValue(jsonResponse(200, { text: "Dzień dobry" }));
    const result = await transcribeAudio("token", new Blob(["x"]));
    expect(result.text).toBe("Dzień dobry");
  });

  it("throws ApiError with Polish message on 4xx with { code, message }", async () => {
    vi.mocked(fetch).mockResolvedValue(
      jsonResponse(403, { detail: { code: "account_pending_approval", message: "Konto oczekuje na akceptację." } })
    );
    await expect(transcribeAudio("token", new Blob(["x"]))).rejects.toBeInstanceOf(ApiError);
    vi.mocked(fetch).mockResolvedValue(
      jsonResponse(403, { detail: { code: "account_pending_approval", message: "Konto oczekuje na akceptację." } })
    );
    await expect(transcribeAudio("token", new Blob(["x"]))).rejects.toMatchObject({
      status: 403,
      code: "account_pending_approval",
      message: "Konto oczekuje na akceptację.",
    });
  });

  it("throws ApiError with fallback message on plain-text error", async () => {
    vi.mocked(fetch).mockResolvedValue(jsonResponse(500, "Internal Server Error"));
    await expect(transcribeAudio("token", new Blob(["x"]))).rejects.toMatchObject({
      status: 500,
      message: "Internal Server Error",
    });
  });

  it("does not throw when the caller aborts before fetch resolves", async () => {
    const abortError = Object.assign(new Error("aborted"), { name: "AbortError" });
    vi.mocked(fetch).mockImplementation((_url, init?: RequestInit) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(abortError));
      })
    );
    const controller = new AbortController();
    const pending = transcribeAudio("token", new Blob(["x"]), controller.signal);
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });

  it("passes signal through to fetch", async () => {
    const fetchMock = vi.mocked(fetch).mockResolvedValue(jsonResponse(200, { text: "ok" }));
    const controller = new AbortController();
    await transcribeAudio("token", new Blob(["x"]), controller.signal);
    expect((fetchMock.mock.calls[0]![1] as RequestInit).signal).toBe(controller.signal);
  });
});