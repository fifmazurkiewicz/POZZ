import { describe, expect, it } from "vitest";
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
