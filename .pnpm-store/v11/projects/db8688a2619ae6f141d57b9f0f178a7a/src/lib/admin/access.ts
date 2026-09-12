import type { AdminUser } from "./api";

export function canOpenAdmin(isAdmin: boolean): boolean {
  return isAdmin;
}

export function canRevokeApproval(currentUserId: string | null, targetUserId: string): boolean {
  return currentUserId !== null && currentUserId !== targetUserId;
}

export function partitionAdminUsers(users: AdminUser[]): { pending: AdminUser[]; approved: AdminUser[] } {
  return { pending: users.filter((row) => !row.is_approved), approved: users.filter((row) => row.is_approved) };
}
