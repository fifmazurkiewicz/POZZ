import { describe, expect, it } from "vitest";
import { PULSE_INTERVAL_FAST_MS, PULSE_INTERVAL_SLOW_MS, PULSE_TIMEOUT_MS } from "./pulse";

describe("api pulse contract", () => {
  it("uses 5s waking / 30s healthy / 8s timeout", () => {
    expect(PULSE_INTERVAL_FAST_MS).toBe(5_000);
    expect(PULSE_INTERVAL_SLOW_MS).toBe(30_000);
    expect(PULSE_TIMEOUT_MS).toBe(8_000);
  });
});
