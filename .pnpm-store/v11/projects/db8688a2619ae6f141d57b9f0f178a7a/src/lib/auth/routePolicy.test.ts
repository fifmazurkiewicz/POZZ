import { describe, expect, it } from "vitest";
import { APPROVAL_POLL_MS, resolveRedirect } from "./routePolicy";

describe("resolveRedirect", () => {
  it("sends anonymous visitors to login", () => {
    expect(resolveRedirect("anonymous", "/simulation")).toBe("/login");
    expect(resolveRedirect("anonymous", "/login")).toBeNull();
    expect(resolveRedirect("anonymous", "/privacy")).toBeNull();
  });

  it("does not guess while profile is unknown", () => {
    expect(resolveRedirect("initializing", "/simulation")).toBeNull();
    expect(resolveRedirect("profile_unknown", "/login")).toBeNull();
    expect(resolveRedirect("pending_approval", "/simulation")).toBeNull();
  });

  it("sends ready users off the login entry routes", () => {
    expect(resolveRedirect("ready", "/login")).toBe("/simulation");
    expect(resolveRedirect("ready", "/")).toBe("/simulation");
    expect(resolveRedirect("ready", "/simulation")).toBeNull();
    expect(resolveRedirect("ready", "/menu")).toBeNull();
    expect(resolveRedirect("ready", "/menu/admin")).toBeNull();
  });
});

describe("approval waiting screen", () => {
  it("polls /api/auth/me every 15s", () => {
    expect(APPROVAL_POLL_MS).toBe(15_000);
  });
});
