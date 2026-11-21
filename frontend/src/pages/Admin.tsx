import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api, type ApiKey, type AuditEntry, type Role, type User } from "../api";
import { useAuth } from "../auth";
import { ago } from "../format";

/** Users and roles, your API keys, and the audit trail (SRS S7, FR-APP-02, FR-APP-07). */
export default function Admin() {
  const { user: me } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [log, setLog] = useState<AuditEntry[]>([]);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({ email: "", name: "", role: "reviewer" as Role, password: "" });

  const load = useCallback(() => {
    Promise.all([api.users(), api.keys(), api.audit()])
      .then(([u, k, a]) => {
        setUsers(u);
        setKeys(k);
        setLog(a);
      })
      .catch((e) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  async function run(fn: () => Promise<unknown>) {
    setError(null);
    try {
      await fn();
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "That didn't work");
    }
  }

  async function addUser(e: FormEvent) {
    e.preventDefault();
    await run(async () => {
      await api.createUser(form);
      setForm({ email: "", name: "", role: "reviewer", password: "" });
    });
  }

  return (
    <>
      <div className="page-head">
        <h1>Users and audit</h1>
      </div>
      {error && (
        <p className="error-box" role="alert" style={{ marginBottom: 16 }}>
          {error}
        </p>
      )}
      <div className="admin-grid">
        <div className="stack">
          <table className="grid">
            <thead>
              <tr>
                <th>User</th>
                <th>Role</th>
                <th>Access</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id}>
                  <td>
                    {u.email}
                    {u.name && <span className="muted small"> · {u.name}</span>}
                  </td>
                  <td>
                    <select
                      aria-label={`Role of ${u.email}`}
                      value={u.role}
                      disabled={u.id === me?.id}
                      onChange={(e) => run(() => api.updateUser(u.id, { role: e.target.value as Role }))}
                    >
                      <option value="submitter">Submitter</option>
                      <option value="reviewer">Reviewer</option>
                      <option value="admin">Admin</option>
                    </select>
                  </td>
                  <td>
                    <button className="btn" disabled={u.id === me?.id} onClick={() => run(() => api.updateUser(u.id, { active: !u.active }))}>
                      {u.active ? "Disable" : "Enable"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <h2 style={{ marginTop: 16 }}>Audit trail</h2>
          <table className="grid">
            <thead>
              <tr>
                <th>When</th>
                <th>Who</th>
                <th>What</th>
                <th className="hide-sm">Details</th>
              </tr>
            </thead>
            <tbody>
              {log.map((a) => (
                <tr key={a.id}>
                  <td className="small muted" title={a.at}>
                    {ago(a.at)}
                  </td>
                  <td className="small">{a.actor_email ?? <span className="muted">system</span>}</td>
                  <td className="small">
                    {a.action}
                    {a.target_id && <span className="muted"> {a.target_id.slice(0, 8)}</span>}
                  </td>
                  <td className="small muted hide-sm" style={{ maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis" }}>
                    {Object.entries(a.detail)
                      .map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`)
                      .join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="stack">
          <form className="options" onSubmit={addUser}>
            <h2>Add a user</h2>
            <label className="field">
              Email
              <input type="email" required value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
            </label>
            <label className="field">
              Name
              <input type="text" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </label>
            <label className="field">
              Role
              <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as Role })}>
                <option value="submitter">Submitter: uploads, sees own documents</option>
                <option value="reviewer">Reviewer: reviews and decides</option>
                <option value="admin">Admin: also users and escalations</option>
              </select>
            </label>
            <label className="field">
              Temporary password
              <input type="password" minLength={8} required value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
            </label>
            <button className="btn primary" type="submit">
              Add user
            </button>
          </form>

          <div className="options">
            <h2>Your API keys</h2>
            <p className="small muted">For systems that upload documents. A key acts as you, with your role.</p>
            {newKey && (
              <div className="note-box small">
                Copy this key now; it won't be shown again.
                <code className="key">{newKey}</code>
              </div>
            )}
            {keys.map((k) => (
              <div key={k.id} className="small" style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                <span>
                  {k.name || "Unnamed"} <span className="muted">fdk_{k.prefix}_…</span>
                  <br />
                  <span className="muted">{k.revoked ? "Revoked" : k.last_used_at ? `Used ${ago(k.last_used_at)}` : "Never used"}</span>
                </span>
                {!k.revoked && (
                  <button className="btn" onClick={() => run(() => api.revokeKey(k.id))}>
                    Revoke
                  </button>
                )}
              </div>
            ))}
            <button
              className="btn"
              onClick={() =>
                run(async () => {
                  const k = await api.createKey(`key ${keys.length + 1}`);
                  setNewKey(k.key);
                })
              }
            >
              Create a key
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
