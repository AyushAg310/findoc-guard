import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, type DocSummary, type FlagBrief, type Stats } from "../api";
import { RiskBar, StatusBadge } from "../components/bits";
import { ago, checkLabel, compactRupees } from "../format";
import { Dialog, reviewList, useToast } from "../ui";

const STATUS_FILTERS: [string, string[]][] = [
  ["To review", ["needs_review", "processed", "escalated"]],
  ["Processing", ["queued", "processing"]],
  ["Decided", ["approved", "rejected"]],
  ["Problems", ["failed", "rejected_quality"]],
  ["All", []],
];
const BANDS = ["high", "medium", "low"];
const PAGE = 25;
type SortKey = "received" | "risk" | "name";

/** The review queue (SRS S6, FR-APP-03/04): filter, sort, select, decide in bulk, export. */
export default function Queue() {
  const navigate = useNavigate();
  const toast = useToast();
  const [params, setParams] = useSearchParams();
  const view = params.get("view") ?? "To review";
  const bands = params.getAll("band");
  const docType = params.get("type") ?? "";
  const q = params.get("q") ?? "";
  const page = Number(params.get("page") ?? 0);
  const sort = (params.get("sort") as SortKey) ?? "risk";
  const desc = params.get("dir") !== "asc";

  const [rows, setRows] = useState<DocSummary[] | null>(null);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulk, setBulk] = useState<string | null>(null);
  const [comment, setComment] = useState("");
  const statuses = useMemo(() => STATUS_FILTERS.find(([n]) => n === view)?.[1] ?? [], [view]);
  const bandKey = bands.join();

  const load = useCallback(() => {
    const p = new URLSearchParams({ limit: String(PAGE), offset: String(page * PAGE) });
    statuses.forEach((s) => p.append("status", s));
    bandKey.split(",").filter(Boolean).forEach((b) => p.append("band", b));
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
  }, [statuses, bandKey, docType, q, page]);

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [load]);

  const sorted = useMemo(() => {
    if (!rows) return null;
    const key = (r: DocSummary) =>
      sort === "risk" ? r.risk_score ?? -1 : sort === "name" ? r.filename.toLowerCase() : r.created_at;
    return [...rows].sort((a, b) => {
      const x = key(a),
        y = key(b);
      return (x < y ? -1 : x > y ? 1 : 0) * (desc ? -1 : 1);
    });
  }, [rows, sort, desc]);

  // the review screen steps through the queue in the order shown here
  useEffect(() => {
    if (sorted) reviewList.set(sorted.map((r) => r.id));
  }, [sorted]);
  useEffect(() => setSelected(new Set()), [view, bandKey, docType, q, page]);

  function set(key: string, value: string | string[] | null) {
    const next = new URLSearchParams(params);
    next.delete(key);
    if (key !== "page") next.delete("page");
    if (Array.isArray(value)) value.forEach((v) => next.append(key, v));
    else if (value) next.set(key, value);
    setParams(next, { replace: true });
  }
  function sortBy(k: SortKey) {
    const next = new URLSearchParams(params);
    next.set("sort", k);
    next.set("dir", sort === k && desc ? "asc" : "desc");
    setParams(next, { replace: true });
  }

  async function exportAs(format: "json" | "xlsx") {
    try {
      const n = await api.exportFile(format, { status: statuses.length ? statuses : undefined, band: bands.length ? bands : undefined });
      toast.show(`Exported ${n} document${n === 1 ? "" : "s"} to ${format === "xlsx" ? "Excel" : "JSON"}`);
    } catch (e) {
      toast.show(e instanceof Error ? e.message : "Export failed", "error");
    }
  }

  async function runBulk(decision: string) {
    try {
      const r = await api.bulkDecide([...selected], decision, comment);
      const skipped = Object.keys(r.skipped).length;
      const verb = { approved: "Approved", rejected: "Rejected", escalated: "Escalated" }[decision];
      toast.show(`${verb} ${r.done.length}${skipped ? `; ${skipped} skipped (already decided or not open)` : ""}`, skipped ? "info" : "ok");
      setSelected(new Set());
      setBulk(null);
      setComment("");
      load();
    } catch (e) {
      toast.show(e instanceof Error ? e.message : "That didn't work", "error");
    }
  }

  const decidable = (sorted ?? []).filter((r) => ["needs_review", "processed"].includes(r.status));
  const allChecked = decidable.length > 0 && decidable.every((r) => selected.has(r.id));
  const s = stats?.by_status ?? {};
  const head = (k: SortKey, label: string) => (
    <th aria-sort={sort === k ? (desc ? "descending" : "ascending") : "none"}>
      <button className="sort" onClick={() => sortBy(k)}>
        {label} <span aria-hidden="true">{sort === k ? (desc ? "↓" : "↑") : "↕"}</span>
      </button>
    </th>
  );

  return (
    <>
      <div className="page-head">
        <h1>Review queue</h1>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn" onClick={() => exportAs("xlsx")}>
            Export to Excel
          </button>
          <button className="btn" onClick={() => exportAs("json")}>
            Export JSON
          </button>
        </div>
      </div>

      <div className="counts" aria-label="Summary">
        {stats ? (
          <>
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
          </>
        ) : (
          Array.from({ length: 5 }, (_, i) => (
            <div key={i}>
              <span className="skel" style={{ width: 44, height: 32 }} />
              <span className="skel" style={{ width: 100, height: 12, marginTop: 6 }} />
            </div>
          ))
        )}
      </div>

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

      {selected.size > 0 && (
        <div className="bulkbar" role="region" aria-label="Selected documents">
          <strong>{selected.size} selected</strong>
          <button className="btn approve" onClick={() => runBulk("approved")}>
            Approve
          </button>
          <button className="btn escalate" onClick={() => setBulk("escalated")}>
            Escalate…
          </button>
          <button className="btn reject" onClick={() => setBulk("rejected")}>
            Reject…
          </button>
          <button className="btn ghost" onClick={() => setSelected(new Set())}>
            Clear
          </button>
        </div>
      )}

      {error && <p className="error-box">{error}</p>}
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>
              <th className="check">
                <input
                  type="checkbox"
                  aria-label="Select every open document on this page"
                  checked={allChecked}
                  disabled={!decidable.length}
                  onChange={() => setSelected(allChecked ? new Set() : new Set(decidable.map((r) => r.id)))}
                />
              </th>
              {head("name", "Document")}
              <th>Status</th>
              {head("risk", "Risk")}
              <th>Why flagged</th>
              <th className="num">Amount</th>
              <th className="hide-md">Submitted by</th>
              {head("received", "Received")}
            </tr>
          </thead>
          <tbody>
            {sorted === null
              ? Array.from({ length: 6 }, (_, i) => (
                  <tr key={i}>
                    {Array.from({ length: 8 }, (_, j) => (
                      <td key={j}>
                        <span className="skel" style={{ width: j === 1 ? 180 : 70, height: 12 }} />
                      </td>
                    ))}
                  </tr>
                ))
              : sorted.map((r) => {
                  const canPick = ["needs_review", "processed"].includes(r.status);
                  return (
                    <tr
                      key={r.id}
                      className={`link ${selected.has(r.id) ? "picked" : ""}`}
                      tabIndex={0}
                      onClick={() => navigate(`/documents/${r.id}`)}
                      onKeyDown={(e) => e.key === "Enter" && navigate(`/documents/${r.id}`)}
                    >
                      <td className="check" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          aria-label={`Select ${r.filename}`}
                          disabled={!canPick}
                          checked={selected.has(r.id)}
                          onChange={() =>
                            setSelected((s0) => {
                              const n = new Set(s0);
                              if (n.has(r.id)) n.delete(r.id);
                              else n.add(r.id);
                              return n;
                            })
                          }
                        />
                      </td>
                      <td className="file" title={r.filename}>
                        {r.filename}
                        {r.kind === "video" && <span className="tag">video</span>}
                        <span className="sub">{r.doc_type && r.doc_type !== "unknown" ? r.doc_type : r.kind}</span>
                      </td>
                      <td>
                        <StatusBadge status={r.status} />
                        {r.status === "processing" && <span className="pulse" aria-hidden="true" />}
                      </td>
                      <td>
                        <RiskBar band={r.risk_band} score={r.risk_score} />
                      </td>
                      <td>
                        <Reasons flags={r.top_flags} />
                      </td>
                      <td className="num">{r.amount ? compactRupees(r.amount) : <span className="muted">–</span>}</td>
                      <td className="hide-md small">{r.owner_email}</td>
                      <td className="small muted">{ago(r.created_at)}</td>
                    </tr>
                  );
                })}
          </tbody>
        </table>
      </div>
      {sorted?.length === 0 && !error && (
        <div className="empty">
          <p>
            {view === "To review"
              ? "Nothing is waiting for review. New uploads appear here as soon as they're processed."
              : "No documents match these filters."}
          </p>
          {view === "To review" && (
            <button className="btn" style={{ marginTop: 10 }} onClick={() => navigate("/upload")}>
              Upload documents
            </button>
          )}
        </div>
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

      {bulk && (
        <Dialog title={`${bulk === "rejected" ? "Reject" : "Escalate"} ${selected.size} document${selected.size > 1 ? "s" : ""}`} onClose={() => setBulk(null)}>
          <label className="field">
            Comment (saved with each document)
            <textarea rows={3} value={comment} onChange={(e) => setComment(e.target.value)} autoFocus />
          </label>
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 12 }}>
            <button className="btn" onClick={() => setBulk(null)}>
              Cancel
            </button>
            <button className={`btn ${bulk === "rejected" ? "reject" : "escalate"}`} disabled={!comment.trim()} onClick={() => runBulk(bulk)}>
              {bulk === "rejected" ? "Reject" : "Escalate"}
            </button>
          </div>
        </Dialog>
      )}
    </>
  );
}

/** The checks that fired, biggest first: two as chips, the rest as a count. */
function Reasons({ flags }: { flags: FlagBrief[] }) {
  if (!flags.length) return <span className="muted small">–</span>;
  return (
    <span className="reasons">
      {flags.slice(0, 2).map((f) => (
        <span key={f.id} className="flag-chip" title={f.reason}>
          {checkLabel(f.id)}
        </span>
      ))}
      {flags.length > 2 && <span className="small muted">+{flags.length - 2}</span>}
    </span>
  );
}
