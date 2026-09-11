import { apiFetch } from "@/lib/api";

export type AdminUser = {
  id: string;
  email: string | null;
  display_name: string | null;
  is_admin: boolean;
  is_approved: boolean;
  spend_cap_usd?: number;
  created_at: string | null;
};

export async function fetchAdminUsers(token: string): Promise<AdminUser[]> {
  return (await apiFetch<{ users: AdminUser[] }>("/api/admin/users", { token })).users;
}

export function updateUser(token: string, userId: string, update: Partial<Pick<AdminUser, "is_approved" | "spend_cap_usd">>): Promise<AdminUser> {
  return apiFetch<AdminUser>(`/api/admin/users/${userId}`, {
    method: "PATCH",
    token,
    body: update,
  });
}
