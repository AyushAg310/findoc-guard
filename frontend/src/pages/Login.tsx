import { useState, type FormEvent } from "react";
import { api } from "../api";
import { Logo } from "../components/bits";
import { useAuth } from "../auth";

export default function Login() {
  const { refresh } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.login(email, password);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login">
      <form onSubmit={submit} aria-labelledby="login-title">
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span style={{ background: "var(--ink)", padding: 6, borderRadius: 4, display: "flex" }}>
            <Logo />
          </span>
          <h1 id="login-title">FinDoc Guard</h1>
        </div>
        <div className="micr" aria-hidden="true" />
        <p className="muted">Sign in to review cheques and invoices.</p>
        <label className="field">
          Email
          <input type="email" autoComplete="username" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label className="field">
          Password
          <input
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        {error && (
          <p className="error-box" role="alert">
            {error}
          </p>
        )}
        <button className="btn primary" disabled={busy} type="submit" style={{ justifyContent: "center" }}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
