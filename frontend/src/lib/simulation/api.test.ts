import { describe, expect, it, vi } from "vitest";
import { composerPlaceholder, speakerLabel } from "./labels";

describe("simulation labels", () => {
  it("labels doctor-asks turns as doctor then patient", () => {
    expect(speakerLabel("doctor_asks", "user")).toBe("Lekarz");
    expect(speakerLabel("doctor_asks", "assistant")).toBe("Pacjent");
    expect(composerPlaceholder("doctor_asks")).toContain("pacjentowi");
  });

  it("labels patient-asks and meta modes", () => {
    expect(speakerLabel("patient_asks", "user")).toBe("Pacjent");
    expect(speakerLabel("patient_asks", "assistant")).toBe("Lekarz");
    expect(speakerLabel("meta_ask", "assistant")).toBe("AI");
  });
});

describe("generateCasePlan", () => {
  it("POSTs to /plan with auth and returns interview_summary", async () => {
    const { generateCasePlan } = await import("./api");
    const captured: { value: { url: string; init: RequestInit } | null } = { value: null };
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      captured.value = { url, init };
      return {
        ok: true,
        status: 200,
        json: async () => ({ interview_summary: "**WYWIAD**\n- Skargi" }),
      } as unknown as Response;
    }));
    try {
      const result = await generateCasePlan("token", "conv-1");
      expect(captured.value?.url).toContain("/api/conversations/conv-1/plan");
      expect(captured.value?.init.method).toBe("POST");
      expect(captured.value?.init.headers).toEqual({ "Content-Type": "application/json", Authorization: "Bearer token" });
      expect(result.interview_summary).toContain("WYWIAD");
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
