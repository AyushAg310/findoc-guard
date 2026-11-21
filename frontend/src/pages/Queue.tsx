import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, type DocSummary, type Stats } from "../api";
import { RiskBar, StatusBadge } from "../components/bits";
import { ago } from "../format";

const STATUS_FILTERS: [string, string[]][] = [
  ["To review", ["needs_review", "processed", "escalated"]],
  ["Processing", ["queued", "processing"]],
  ["Decided", ["approved", "rejected"]],
  ["Problems", ["failed", "rejected_quality"]],
  ["All", []],
];
const BANDS = ["high", "medium", "low"];
const PAGE = 25;

/** The review queue and batch dashboard (SRS S6, FR-APP-03/04). */
export default function Queue() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const view = params.get("view") ?? "To review";
  const bands = params.getAll("band");
  const docType = params.get("type") ?? "";
  const q = params.get("q") ?? "";
  const page = Number(params.get("page") ?? 0);

  const [rows, setRows] = useState<DocSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const statuses = useMemo(() => STATUS_FILTERS.find(([n]) => n === view)?.[1] ?? [], [view]);

  const load = useCallback(() => {
    const p = new URLSearchParams({ limit: String(PAGE), offset: String(page * PAGE) });
    statuses.forEach((s) => p.append("status", s));
    bands.forEach((b) => p.append("band", b));
    if (docType) p.set("doc_type", docType);
    if (q) p.set("q", q);
    api
      .documents(p)
      .then((r) => {
        setRows(r.items);
        setTotal(r.total);
        setError(null);
      })
      .catch((e) => setError(e.message));
    api.stats().then(setStats).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statuses, bands.join(), docType, q, page]);

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [load]);

  function set(key: string, value: string | string[] | null) {
    const next = new URLSearchParams(params);
    next.delete(key);
    next.delete("page");
    if (Array.isArray(value)) value.forEach((v) => next.append(key, v));
    else if (value) next.set(key, value);
    setParams(next, { replace: true });
  }

  async function exportAs(format: "json" | "xlsx") {
    setExporting(true);
    try {
      await api.exportFile(format, { status: statuses.length ? statuses : undefined, band: bands.length ? bands : undefined });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Export failed");
    } finally {
      setExporting(false);
    }
  }

  const s = stats?.by_status ?? {};
  return (
    <>
      <div className="page-head">
        <h1>Review queue</h1>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn" disabled={exporting} onClick={() => exportAs("xlsx")}>
            Export to Excel
          </button>
          <button className="btn" disabled={exporting} onClick={() => exportAs("json")}>
            Export JSON
          </button>
        </div>
      </div>

      {stats && (
        <div className="counts" aria-label="Summary">
          <div>
            <strong>{stats.open_reviews}</strong>
            <span>waiting for a decision</span>
          </div>
          <div>
            <strong style={{ color: "var(--high)" }}>{stats.by_band.high ?? 0}</strong>
            <span>high risk</span>
          </div>
          <div>
            <strong>{(s.queued ?? 0) + (s.processing ?? 0)}</strong>
            <span>in processing</span>
          </div>
          <div>
            <strong>{stats.average_risk != null ? Math.round(stats.average_risk) : "–"}</strong>
            <span>average risk score</span>
          </div>
          <div>
            <strong>{(s.approved ?? 0) + (s.rejected ?? 0)}</strong>
            <span>decided of {stats.total}</span>
          </div>
        </div>
      )}

      <div className="filters">
        <fieldset>
          <legend>Show</legend>
          <div className="chips">
            {STATUS_FILTERS.map(([name]) => (
              <button key={name} className="chip" aria-pressed={view === name} onClick={() => set("view", name)}>
                {name}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>Risk</legend>
          <div className="chips">
            {BANDS.map((b) => (
              <button
                key={b}
                className="chip"
                aria-pressed={bands.includes(b)}
                onClick={() => set("band", bands.includes(b) ? bands.filter((x) => x !== b) : [...bands, b])}
              >
                {b.charAt(0).toUpperCase() + b.slice(1)}
              </button>
            ))}
          </div>
        </fieldset>
        <label className="field">
          Type
          <select value={docType} onChange={(e) => set("type", e.target.value)}>
            <option value="">Any</option>
            <option value="cheque">Cheques</option>
            <option value="invoice">Invoices</option>
            <option value="receipt">Receipts</option>
          </select>
        </label>
        <label className="field">
          File name
          <input type="search" defaultValue={q} onChange={(e) => set("q", e.target.value)} placeholder="Search" />
        </label>
      </div>

      {error && <p className="error-box">{error}</p>}
      <table className="grid">
        <thead>
          <tr>
            <th>Document</th>
            <th>Type</th>
            <th>Status</th>
            <th>Risk</th>
            <th className="hide-sm">Submitted by</th>
            <th className="hide-sm">Received</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.id}
              className="link"
              tabIndex={0}
              onClick={() => navigate(`/documents/${r.id}`)}
              onKeyDown={(e) => e.key === "Enter" && navigate(`/documents/${r.id}`)}
            >
              <td className="file" title={r.filename}>
                {r.filename}
                {r.kind === "video" && <span className="muted small"> · video</span>}
              </td>
              <td>{r.doc_type && r.doc_type !== "unknown" ? r.doc_type : <span className="muted">–</span>}</td>
              <td>
                <StatusBadge status={r.status} />
              </td>
              <td>
                <RiskBar band={r.risk_band} score={r.risk_score} />
              </td>
              <td className="hide-sm small">{r.owner_email}</td>
              <td className="hide-sm small muted">{ago(r.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && !error && (
        <p className="empty">
          {view === "To review" ? "Nothing waiting for review. New uploads appear here once processed." : "No documents match these filters."}
        </p>
      )}
      {total > PAGE && (
        <div className="pager">
          <span>
            {page * PAGE + 1}–{Math.min(total, (page + 1) * PAGE)} of {total}
          </span>
          <span style={{ display: "flex", gap: 8 }}>
            <button className="btn" disabled={page === 0} onClick={() => set("page", String(page - 1))}>
              Previous
            </button>
            <button className="btn" disabled={(page + 1) * PAGE >= total} onClick={() => set("page", String(page + 1))}>
              Next
            </button>
          </span>
        </div>
      )}
    </>
  );
}
