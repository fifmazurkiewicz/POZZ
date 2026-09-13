import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, apiUrl: (path: string) => `http://localhost:8000${path}` };
});

import { uploadRecordedInterview } from "./api";

describe("uploadRecordedInterview", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn()));
  afterEach(() => vi.unstubAllGlobals());

  it("uploads audio and optional title as authenticated multipart data", async () => {
    const fetchMock = vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ conversation_id: "recorded-1" }),
    } as Response);
    const audio = new File(["audio"], "wizyta.webm", { type: "audio/webm" });

    const result = await uploadRecordedInterview("token", audio, "Wizyta kontrolna");

    expect(result.conversation_id).toBe("recorded-1");
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://localhost:8000/api/interviews/recordings");
    expect(init?.headers).toEqual({ Authorization: "Bearer token" });
    const form = init?.body as FormData;
    expect(form.get("audio")).toBeInstanceOf(File);
    expect(form.get("title")).toBe("Wizyta kontrolna");
  });

  it("accepts a raw Blob (e.g. MediaRecorder output) as the audio field", async () => {
    const fetchMock = vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ conversation_id: "recorded-2" }),
    } as Response);
    const audio = new Blob(["recorder-output"], { type: "audio/webm" });

    const result = await uploadRecordedInterview("token", audio);

    expect(result.conversation_id).toBe("recorded-2");
    const form = fetchMock.mock.calls[0]?.[1]?.body as FormData;
    expect(form.get("audio")).toBeInstanceOf(Blob);
    expect((form.get("audio") as Blob).size).toBe(audio.size);
    expect(form.get("title")).toBeNull();
  });
});
