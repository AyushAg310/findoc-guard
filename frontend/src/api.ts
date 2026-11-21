// Typed client for the FinDoc Guard REST API (see /docs on the server).

export type Role = "submitter" | "reviewer" | "admin";

export interface User {
  id: string;
  email: string;
  name: string;
  role: Role;
  active: boolean;
  created_at: string;
}

export interface DocSummary {
  id: string;
  filename: string;
  kind: string;
  status: string;
  stage: string | null;
  progress: number;
  attempts: number;
  doc_type: string | null;
  risk_score: number | null;
  risk_band: "low" | "medium" | "high" | null;
  owner_email: string;
  created_at: string;
  finished_at: string | null;
}

export interface FieldView {
  name: string;
  value: string | null;
  normalized: string | null;
  corrected: string | null;
  confidence: number;
  bbox: [number, number, number, number] | null;
  method: string | null;
}

export interface CheckView {
  id: string;
  fired: boolean;
  severity: number;
  confidence: number;
  reason: string;
  contribution: number | null;
  evidence: Record<string, unknown>;
}

export interface DocDetail extends DocSummary {
  media_type: string;
  size_bytes: number;
  error: string | null;
  images: string[];
  page_frame: "template" | "flat";
  fields: FieldView[];
  line_items: Record<string, unknown>[];
  checks: CheckView[];
  risk: { score?: number; band?: string; forced_by?: string | null };
  quality: Record<string, number>;
  messages: string[];
  stages: { stage: string; ms: number; ok: boolean; note: string }[];
  flattening: string | null;
  decisions: { decision: string; comment: string; user_email: string; created_at: string }[];
  field_edits: { field: string; old_value: string | null; new_value: string; user_email: string; created_at: string }[];
  allowed_decisions: string[];
}

export interface Stats {
  total: number;
  by_status: Record<string, number>;
  by_band: Record<string, number>;
  by_type: Record<string, number>;
  average_risk: number | null;
  open_reviews: number;
}

export interface AuditEntry {
  id: number;
  at: string;
  actor_email: string | null;
  action: string;
  target_type: string;
  target_id: string | null;
  detail: Record<string, unknown>;
}

export interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  revoked: boolean;
  key: string | null;
}

const TOKEN = "findoc.token";

export const session = {
  get token(): string | null {
    return sessionStorage.getItem(TOKEN);
  },
  set token(t: string | null) {
    if (t) sessionStorage.setItem(TOKEN, t);
    else sessionStorage.removeItem(TOKEN);
  },
};

/** An RFC 9457 problem from the server, with its human-readable detail. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (session.token) headers.set("Authorization", `Bearer ${session.token}`);
  if (init.body && typeof init.body === "string") headers.set("Content-Type", "application/json");
  const res = await fetch(path, { ...init, headers });
  if (res.status === 401 && !path.endsWith("/auth/login")) {
    session.token = null;
    onUnauthorized();
  }
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const p = await res.json();
      msg = typeof p.detail === "string" ? p.detail : p.errors?.map((e: { msg: string }) => e.msg).join("; ") || p.title;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (body: unknown) => JSON.stringify(body);

export const api = {
  async login(email: string, password: string) {
    const body = new URLSearchParams({ username: email, password });
    const r = await request<{ access_token: string }>("/api/v1/auth/login", { method: "POST", body });
    session.token = r.access_token;
  },
  me: () => request<User>("/api/v1/auth/me"),

  documents(params: URLSearchParams) {
    return request<{ items: DocSummary[]; total: number }>(`/api/v1/documents?${params}`);
  },
  stats: () => request<Stats>("/api/v1/documents/stats"),
  document: (id: string) => request<DocDetail>(`/api/v1/documents/${id}`),
  upload(file: File, register: boolean, account: string) {
    const form = new FormData();
    form.append("file", file);
    form.append("register", String(register));
    if (account) form.append("account", account);
    return request<{ id: string; status: string; duplicate_of: string | null }>("/api/v1/documents", {
      method: "POST",
      body: form,
    });
  },
  correct: (id: string, fields: Record<string, string>) =>
    request<DocDetail>(`/api/v1/documents/${id}/fields`, { method: "PATCH", body: json({ fields }) }),
  decide: (id: string, decision: string, comment: string) =>
    request<DocDetail>(`/api/v1/documents/${id}/decision`, { method: "POST", body: json({ decision, comment }) }),
  retry: (id: string) => request<DocSummary>(`/api/v1/documents/${id}/retry`, { method: "POST" }),

  /** Images need the Authorization header, so they're fetched as blobs, not plain <img src>. */
  async image(id: string, variant: string): Promise<string> {
    const res = await fetch(`/api/v1/documents/${id}/image?variant=${variant}`, {
      headers: session.token ? { Authorization: `Bearer ${session.token}` } : {},
    });
    if (!res.ok) throw new ApiError(res.status, "Image not available");
    return URL.createObjectURL(await res.blob());
  },

  async exportFile(format: "json" | "xlsx", filters: { status?: string[]; band?: string[] }) {
    const ex = await request<{ id: string; count: number; download: string }>("/api/v1/exports", {
      method: "POST",
      body: json({ format, ...filters }),
    });
    const res = await fetch(ex.download, { headers: { Authorization: `Bearer ${session.token}` } });
    const url = URL.createObjectURL(await res.blob());
    const a = Object.assign(document.createElement("a"), { href: url, download: `findoc-export.${format}` });
    a.click();
    URL.revokeObjectURL(url);
    return ex.count;
  },

  users: () => request<User[]>("/api/v1/users"),
  createUser: (body: { email: string; name: string; role: Role; password: string }) =>
    request<User>("/api/v1/users", { method: "POST", body: json(body) }),
  updateUser: (id: string, body: Partial<{ role: Role; active: boolean }>) =>
    request<User>(`/api/v1/users/${id}`, { method: "PATCH", body: json(body) }),
  keys: () => request<ApiKey[]>("/api/v1/api-keys"),
  createKey: (name: string) => request<ApiKey>("/api/v1/api-keys", { method: "POST", body: json({ name }) }),
  revokeKey: (id: string) => request<void>(`/api/v1/api-keys/${id}`, { method: "DELETE" }),
  audit: (params = "") => request<AuditEntry[]>(`/api/v1/audit?limit=200${params}`),
};
