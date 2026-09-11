import { describe, expect, it } from "vitest";
import { canOpenAdmin, canRevokeApproval, partitionAdminUsers } from "./access";
import type { AdminUser } from "./api";

function user(partial: Partial<AdminUser> & Pick<AdminUser, "id" | "is_approved">): AdminUser {
  return {
    email: null,
    display_name: null,
    is_admin: false,
    created_at: null,
    ...partial,
  };
}

describe("admin access", () => {
  it("opens Admin only for admins", () => {
    expect(canOpenAdmin(true)).toBe(true);
    expect(canOpenAdmin(false)).toBe(false);
  });

  it("hides revoke on the admin own row", () => {
    expect(canRevokeApproval("admin-1", "admin-1")).toBe(false);
    expect(canRevokeApproval("admin-1", "other-2")).toBe(true);
    expect(canRevokeApproval(null, "other-2")).toBe(false);
  });
});

describe("partitionAdminUsers", () => {
  it("puts pending accounts first as their own group", () => {
    const rows = [
      user({ id: "ok", email: "ok@example.com", is_approved: true }),
      user({ id: "wait", email: "wait@example.com", is_approved: false }),
    ];
    const { pending, approved } = partitionAdminUsers(rows);
    expect(pending.map((row) => row.email)).toEqual(["wait@example.com"]);
    expect(approved.map((row) => row.email)).toEqual(["ok@example.com"]);
  });
});
