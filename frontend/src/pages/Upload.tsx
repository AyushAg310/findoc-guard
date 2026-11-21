import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, type DocSummary } from "../api";
import { RiskBadge, StatusBadge } from "../components/bits";
import { STAGE_LABEL, ago } from "../format";

const ACCEPT = "image/jpeg,image/png,image/webp,image/bmp,image/tiff,video/mp4,video/quicktime,video/x-msvideo,video/webm";

/** Upload photos or videos and follow their processing (SRS S1, S2). */
export default function Upload() {
  const [over, setOver] = useState(false);
  const [register, setRegister] = useState(false);
  const [account, setAccount] = useState("");
  const [messages, setMessages] = useState<{ kind: "error" | "note"; text: string }[]>([]);
  const [rows, setRows] = useState<DocSummary[]>([]);
  const input = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    api
      .documents(new URLSearchParams({ limit: "15" }))
      .then((r) => setRows(r.items))
      .catch(() => {});
  }, []);

  const busy = rows.some((r) => r.status === "queued" || r.status === "processing");
  useEffect(() => {
    load();
    const t = setInterval(load, busy ? 1500 : 8000);
    return () => clearInterval(t);
  }, [load, busy]);

  async function send(files: FileList | null) {
    if (!files?.length) return;
    const notes: typeof messages = [];
    for (const f of Array.from(files)) {
      try {
        const r = await api.upload(f, register, account.trim());
        if (r.duplicate_of) notes.push({ kind: "note", text: `${f.name}: you uploaded this exact file before.` });
      } catch (e) {
        notes.push({ kind: "error", text: `${f.name}: ${e instanceof Error ? e.message : "upload failed"}` });
      }
    }
    setMessages(notes);
    if (input.current) input.current.value = "";
    load();
  }

  return (
    <>
      <div className="page-head">
        <h1>Upload documents</h1>
      </div>
      <div className="upload-grid">
        <div className="stack">
          <label
            className={`drop ${over ? "over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setOver(false);
              send(e.dataTransfer.files);
            }}
          >
            <svg width="44" height="30" viewBox="0 0 44 30" aria-hidden="true">
              <rect x="1" y="1" width="42" height="28" rx="3" fill="none" stroke="var(--ink-3)" strokeWidth="2" />
              <path d="M7 9h20 M7 15h12 M7 23h3 M12 23h2 M16 23h5 M23 23h2" stroke="var(--teal)" strokeWidth="2.5" />
            </svg>
            <strong>Drop cheque or invoice photos here, or choose files</strong>
            <span className="muted small">
              Photos (JPEG, PNG, WebP) or a 3–5 second video sweep of a folded cheque. Up to 200 MB each.
            </span>
            <input ref={input} type="file" multiple accept={ACCEPT} className="visually-hidden" onChange={(e) => send(e.target.files)} />
          </label>
          {messages.map((m, i) => (
            <p key={i} className={m.kind === "error" ? "error-box" : "note-box"} role={m.kind === "error" ? "alert" : "status"}>
              {m.text}
            </p>
          ))}
        </div>
        <div className="options">
          <h2>Cheque options</h2>
          <label style={{ display: "flex", gap: 8, alignItems: "start" }}>
            <input type="checkbox" checked={register} onChange={(e) => setRegister(e.target.checked)} />
            <span>
              Add to the duplicate index
              <span className="muted small" style={{ display: "block" }}>
                Later uploads of the same cheque will be flagged.
              </span>
            </span>
          </label>
          <label className="field">
            Account for the signature check
            <input type="text" inputMode="numeric" value={account} maxLength={34} onChange={(e) => setAccount(e.target.value)} />
            <span className="muted">Optional; used when the account number can't be read.</span>
          </label>
        </div>
      </div>

      <h2 style={{ margin: "32px 0 12px" }}>Recent uploads</h2>
      <table className="grid">
        <thead>
          <tr>
            <th>Document</th>
            <th>Status</th>
            <th>Progress</th>
            <th>Risk</th>
            <th className="hide-sm">Uploaded</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const running = r.status === "queued" || r.status === "processing";
            return (
              <tr key={r.id}>
                <td className="file">
                  <Link to={`/documents/${r.id}`}>{r.filename}</Link>
                </td>
                <td>
                  <StatusBadge status={r.status} />
                </td>
                <td className="small">
                  {running ? (
                    <div style={{ display: "grid", gap: 4 }}>
                      <div className="progress" role="progressbar" aria-valuenow={Math.round(r.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
                        <span style={{ width: `${Math.round(r.progress * 100)}%` }} />
                      </div>
                      <span className="muted">
                        {r.status === "queued" ? (r.attempts ? `Retrying (attempt ${r.attempts + 1})` : "Waiting for a worker") : STAGE_LABEL[r.stage ?? ""] ?? r.stage}
                      </span>
                    </div>
                  ) : (
                    <span className="muted">Done</span>
                  )}
                </td>
                <td>
                  <RiskBadge band={r.risk_band} score={r.risk_score} />
                </td>
                <td className="hide-sm small muted">{ago(r.created_at)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows.length === 0 && <p className="empty">No uploads yet.</p>}
    </>
  );
}
