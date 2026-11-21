import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, type DocSummary } from "../api";
import { RiskBadge, StatusBadge } from "../components/bits";
import { STAGE_LABEL, ago, bytes } from "../format";
import { useToast } from "../ui";

interface Sending {
  key: number;
  name: string;
  size: number;
  thumb: string | null;
  progress: number;
  state: "sending" | "done" | "error";
  note?: string;
}

const ACCEPT = "image/jpeg,image/png,image/webp,image/bmp,image/tiff,video/mp4,video/quicktime,video/x-msvideo,video/webm";

/** Upload photos or videos and follow their processing (SRS S1, S2). */
export default function Upload() {
  const [over, setOver] = useState(false);
  const [register, setRegister] = useState(false);
  const [account, setAccount] = useState("");
  const [sending, setSending] = useState<Sending[]>([]);
  const toast = useToast();
  const counter = useRef(0);
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

  function patch(key: number, p: Partial<Sending>) {
    setSending((all) => all.map((s) => (s.key === key ? { ...s, ...p } : s)));
  }

  async function send(files: FileList | null) {
    if (!files?.length) return;
    const list = Array.from(files).map((f) => ({
      file: f,
      item: {
        key: ++counter.current,
        name: f.name,
        size: f.size,
        thumb: f.type.startsWith("image/") ? URL.createObjectURL(f) : null,
        progress: 0,
        state: "sending" as const,
      },
    }));
    setSending((s) => [...list.map((x) => x.item), ...s].slice(0, 12));
    if (input.current) input.current.value = "";
    let ok = 0;
    await Promise.all(
      list.map(async ({ file, item }) => {
        try {
          const r = await api.uploadWithProgress(file, register, account.trim(), (p) => patch(item.key, { progress: p }));
          ok++;
          patch(item.key, { state: "done", progress: 1, note: r.duplicate_of ? "You uploaded this exact file before" : "Queued for processing" });
        } catch (e) {
          patch(item.key, { state: "error", note: e instanceof Error ? e.message : "Upload failed" });
        }
      }),
    );
    if (ok) toast.show(`Uploaded ${ok} file${ok > 1 ? "s" : ""}; processing has started`);
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
          <label className="btn camera">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
              <path d="M2 5h3l1.5-2h3L11 5h3v8H2z" />
              <circle cx="8" cy="9" r="2.5" />
            </svg>
            Take a photo
            <input type="file" accept="image/*" capture="environment" className="visually-hidden" onChange={(e) => send(e.target.files)} />
          </label>
          {sending.length > 0 && (
            <ul className="sending" aria-label="Uploads in progress">
              {sending.map((s) => (
                <li key={s.key} className={s.state}>
                  <span className="thumb">{s.thumb ? <img src={s.thumb} alt="" /> : <span className="vid">▶</span>}</span>
                  <span className="meta">
                    <span className="file">{s.name}</span>
                    <span className="small muted">
                      {bytes(s.size)}
                      {s.note ? ` · ${s.note}` : s.state === "sending" ? ` · ${Math.round(s.progress * 100)}%` : ""}
                    </span>
                    <span className="progress" style={{ width: "100%" }}>
                      <span style={{ width: `${Math.round(s.progress * 100)}%` }} />
                    </span>
                  </span>
                  <span className="state" aria-label={s.state}>
                    {s.state === "done" ? "✓" : s.state === "error" ? "!" : ""}
                  </span>
                </li>
              ))}
            </ul>
          )}
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
