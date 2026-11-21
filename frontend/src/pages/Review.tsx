import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type CheckView, type DocDetail, type FieldView } from "../api";
import { isReviewer, useAuth } from "../auth";
import { RiskBadge, StatusBadge, useApiImage } from "../components/bits";
import { ZoomCanvas, type ZoomHandle } from "../components/ZoomCanvas";
import {
  MONEY_FIELDS,
  STAGE_LABEL,
  ago,
  bytes,
  checkLabel,
  confidenceOf,
  fieldLabel,
  fieldsForCheck,
  rupees,
} from "../format";
import { reviewList, useToast } from "../ui";

const VIEWS: Record<string, string> = {
  page: "Fields",
  heatmap: "Heatmap on the flat page",
  original: "Capture",
  annotated: "Template boxes",
};
const READ_ONLY = new Set(["account_number", "micr", "signature"]); // masked, or not text
// Fields shown in their checked, normalised form (dates as ISO, amounts as numbers, GSTINs
// repaired); names and descriptions are shown as printed.
const SHOW_NORMALIZED = new Set(["date", "amount_figures", "amount_words", "subtotal", "cgst", "sgst", "igst",
  "total", "vendor_gstin", "buyer_gstin", "invoice_no"]);
const shownValue = (f: FieldView) =>
  f.corrected ?? (SHOW_NORMALIZED.has(f.name) ? f.normalized ?? f.value : f.value) ?? "";
const DECISIONS = [
  { id: "approved", label: "Approve", key: "a", cls: "approve" },
  { id: "escalated", label: "Escalate", key: "e", cls: "escalate" },
  { id: "rejected", label: "Reject", key: "r", cls: "reject" },
];

/** One document: the page with its evidence on the left, what was read and checked on the right,
 * and the decision at the bottom (SRS S4, S5; FR-APP-01, 02). */
export default function Review() {
  const { id = "" } = useParams();
  const { user } = useAuth();
  const navigate = useNavigate();
  const reviewer = isReviewer(user);
  const [doc, setDoc] = useState<DocDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [litCheck, setLitCheck] = useState<string | null>(null);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [heat, setHeat] = useState(0); // heatmap overlay opacity on the Fields view, 0..1
  const [autoNext, setAutoNext] = useState(true);
  const commentRef = useRef<HTMLTextAreaElement>(null);
  const zoom = useRef<ZoomHandle>(null);
  const toast = useToast();

  // the queue this document was opened from: J / K and "approve and next" walk through it
  const list = useMemo(() => reviewList.get(), []);
  const pos = list.indexOf(id);
  const prevId = pos > 0 ? list[pos - 1] : null;
  const nextId = pos >= 0 && pos < list.length - 1 ? list[pos + 1] : null;
  useEffect(() => {
    setSelected(null);
    setLitCheck(null);
    setEdits({});
    setComment("");
    setError(null);
    setDoc(null);
    setView(null);
  }, [id]);

  const load = useCallback(() => {
    api
      .document(id)
      .then((d) => {
        setDoc(d);
        setError(null);
        setView((v) => v ?? (d.images.includes("page") ? "page" : d.images[0] ?? null));
      })
      .catch((e) => setError(e.message));
  }, [id]);

  const running = doc?.status === "queued" || doc?.status === "processing";
  useEffect(() => {
    load();
  }, [load]);
  useEffect(() => {
    if (!running) return;
    const t = setInterval(load, 1500);
    return () => clearInterval(t);
  }, [running, load]);

  const fired = useMemo(
    () => (doc?.checks ?? []).filter((c) => c.fired).sort((a, b) => (b.contribution ?? 0) - (a.contribution ?? 0)),
    [doc],
  );
  const passed = (doc?.checks ?? []).filter((c) => !c.fired);
  const lit = useMemo(() => {
    const c = fired.find((x) => x.id === litCheck);
    return new Set(c ? fieldsForCheck(c.id, c.evidence) : []);
  }, [fired, litCheck]);
  const dirty = Object.keys(edits).length > 0;

  async function saveEdits() {
    if (!doc || !dirty) return;
    setBusy(true);
    try {
      setDoc(await api.correct(doc.id, edits));
      toast.show(`Saved ${Object.keys(edits).length} correction${Object.keys(edits).length > 1 ? "s" : ""}`);
      setEdits({});
    } catch (e) {
      toast.show(e instanceof Error ? e.message : "Saving failed", "error");
    } finally {
      setBusy(false);
    }
  }

  const decide = useCallback(
    async (decision: string) => {
      if (!doc) return;
      if (decision !== "approved" && !comment.trim()) {
        setError(`Add a comment to ${decision === "rejected" ? "reject" : "escalate"}: say what's wrong.`);
        commentRef.current?.focus();
        return;
      }
      setBusy(true);
      try {
        if (dirty) await api.correct(doc.id, edits);
        const d = await api.decide(doc.id, decision, comment);
        const verb = { approved: "Approved", rejected: "Rejected", escalated: "Escalated" }[decision];
        setEdits({});
        setComment("");
        setError(null);
        if (autoNext && nextId) {
          toast.show(`${verb} ${d.filename}`, "ok", { label: "Go back", run: () => navigate(`/documents/${d.id}`) });
          navigate(`/documents/${nextId}`);
        } else {
          setDoc(d);
          toast.show(`${verb} ${d.filename}${nextId ? "" : ". That was the last one in the queue."}`);
        }
      } catch (e) {
        toast.show(e instanceof Error ? e.message : "The decision wasn't saved", "error");
      } finally {
        setBusy(false);
      }
    },
    [doc, comment, dirty, edits, autoNext, nextId, navigate, toast],
  );

  // A approve, E escalate, R reject (SRS 12), unless typing in a field
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement;
      if (e.ctrlKey || e.metaKey || e.altKey || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName)) return;
      const k = e.key.toLowerCase();
      const d = DECISIONS.find((x) => x.key === k);
      if (d && doc?.allowed_decisions.includes(d.id)) {
        e.preventDefault();
        decide(d.id);
      } else if (k === "j" && nextId) navigate(`/documents/${nextId}`);
      else if (k === "k" && prevId) navigate(`/documents/${prevId}`);
      else if (k === "+" || k === "=") zoom.current?.zoomBy(1.4);
      else if (k === "-") zoom.current?.zoomBy(1 / 1.4);
      else if (k === "0") zoom.current?.fit();
      else if (k === "h" && doc?.images.includes("overlay")) {
        setView("page");
        setHeat((h) => (h > 0 ? 0 : 0.6));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [doc, decide, nextId, prevId, navigate]);

  if (error && !doc) return <p className="error-box">{error}</p>;
  if (!doc) return <ReviewSkeleton />;

  return (
    <>
      <div className="page-head">
        <div>
          <p className="small">
            <Link to={reviewer ? "/" : "/upload"}>{reviewer ? "Review queue" : "My documents"}</Link>
          </p>
          <h1 style={{ wordBreak: "break-all" }}>{doc.filename}</h1>
          <p className="muted small">
            {doc.doc_type && doc.doc_type !== "unknown" ? doc.doc_type : doc.kind} · {bytes(doc.size_bytes)} · from{" "}
            {doc.owner_email} · {ago(doc.created_at)}
            {doc.flattening === "3d" && " · flattened in 3D"}
          </p>
        </div>
        <div className="head-side">
          <StatusBadge status={doc.status} />
          {pos >= 0 && list.length > 1 && (
            <div className="stepper" aria-label="Queue position">
              <button className="icon" disabled={!prevId} onClick={() => prevId && navigate(`/documents/${prevId}`)} title="Previous (K)" aria-label="Previous document">
                ‹
              </button>
              <span className="small muted">
                {pos + 1} of {list.length}
              </span>
              <button className="icon" disabled={!nextId} onClick={() => nextId && navigate(`/documents/${nextId}`)} title="Next (J)" aria-label="Next document">
                ›
              </button>
            </div>
          )}
        </div>
      </div>

      {running && <Processing doc={doc} />}
      {doc.status === "failed" && (
        <div className="error-box" style={{ marginBottom: 16 }}>
          <p>
            Processing failed after {doc.attempts} attempt{doc.attempts === 1 ? "" : "s"}: {doc.error?.split("\n")[0]}
          </p>
          {reviewer && (
            <button className="btn" style={{ marginTop: 8 }} onClick={() => api.retry(doc.id).then(load)}>
              Try again
            </button>
          )}
        </div>
      )}
      {doc.status === "rejected_quality" && (
        <p className="error-box" style={{ marginBottom: 16 }}>
          The photo can't be read reliably: {doc.messages.find((m) => /blur|cover|quality|found/i.test(m)) ?? "image quality too low"}.
          Ask for a sharper photo taken straight on, with the whole document in view.
        </p>
      )}
      {!running && doc.decisions.length > 0 && (
        <p className="note-box" style={{ marginBottom: 16 }} role="status">
          {(() => {
            const d = doc.decisions[doc.decisions.length - 1];
            const verb = d.decision === "escalated" ? "Escalated" : d.decision === "approved" ? "Approved" : "Rejected";
            const next = doc.status === "escalated" && doc.allowed_decisions.length === 0 ? " An admin makes the final decision." : "";
            return `${verb} by ${d.user_email} ${ago(d.created_at)}${d.comment ? `: “${d.comment}”` : "."}${next}`;
          })()}
        </p>
      )}
      {error && (
        <p className="error-box" role="alert" style={{ marginBottom: 16 }}>
          {error}
        </p>
      )}

      {!running && (
        <div className="review">
          <div className="canvas-wrap">
            {doc.images.length > 0 && (
              <div className="view-tabs" role="tablist" aria-label="Image">
                {doc.images
                  .filter((v) => v in VIEWS)
                  .map((v) => (
                    <button key={v} role="tab" aria-selected={view === v} onClick={() => setView(v)}>
                      {VIEWS[v]}
                    </button>
                  ))}
              </div>
            )}
            <Canvas
              doc={doc}
              view={view}
              selected={selected}
              lit={lit}
              heat={heat}
              setHeat={setHeat}
              zoomRef={zoom}
              onSelect={(n) => {
                setSelected(n);
                document.getElementById(`f-${n}`)?.focus();
              }}
            />
          </div>

          <div className="panel">
            {doc.risk.score != null && (
              <section className={`section score ${doc.risk.band}`} aria-label="Risk">
                <span className="num">{doc.risk.score}</span>
                <div>
                  <RiskBadge band={doc.risk.band ?? null} />
                  <p className="small muted">
                    {doc.risk.forced_by
                      ? `Forced high by ${checkLabel(doc.risk.forced_by)}`
                      : fired.length
                        ? `${fired.length} check${fired.length > 1 ? "s" : ""} raised a flag`
                        : "No check raised a flag"}
                  </p>
                </div>
              </section>
            )}

            {doc.checks.length > 0 && (
              <section className="section" aria-labelledby="checks-h">
                <header>
                  <h2 id="checks-h">Checks</h2>
                  {fired.length > 0 && <span className="small muted">Select a flag to see it on the page</span>}
                </header>
                <ul className="checks">
                  {fired.map((c) => (
                    <CheckRow key={c.id} c={c} active={litCheck === c.id} onClick={() => {
                      setLitCheck(litCheck === c.id ? null : c.id);
                      if (doc.images.includes("page")) setView("page");
                      if (c.id === "TAMPER_REGION" && doc.images.includes("overlay")) setHeat((h) => (h > 0 && litCheck === c.id ? 0 : 0.6));
                    }} />
                  ))}
                  {passed.length > 0 && (
                    <li>
                      <details className="passed">
                        <summary>
                          {passed.length} check{passed.length > 1 ? "s" : ""} passed
                        </summary>
                        <ul style={{ paddingLeft: 18, margin: "6px 0 0" }}>
                          {passed.map((c) => (
                            <li key={c.id} style={{ border: 0 }}>
                              {checkLabel(c.id)}: {c.reason}
                            </li>
                          ))}
                        </ul>
                      </details>
                    </li>
                  )}
                </ul>
              </section>
            )}

            {doc.fields.length > 0 && (
              <section className="section" aria-labelledby="fields-h">
                <header>
                  <h2 id="fields-h">Fields</h2>
                  {reviewer && dirty && (
                    <button className="btn primary" disabled={busy} onClick={saveEdits}>
                      Save corrections
                    </button>
                  )}
                </header>
                <div className="fields">
                  {doc.fields.map((f) => (
                    <FieldRow
                      key={f.name}
                      f={f}
                      editable={reviewer && !READ_ONLY.has(f.name) && doc.allowed_decisions.length > 0}
                      value={edits[f.name]}
                      selected={selected === f.name}
                      onFocus={() => setSelected(f.name)}
                      onChange={(v) => {
                        const original = shownValue(f);
                        setEdits((e) => {
                          const next = { ...e };
                          if (v === original) delete next[f.name];
                          else next[f.name] = v;
                          return next;
                        });
                      }}
                    />
                  ))}
                </div>
              </section>
            )}

            {doc.line_items.length > 0 && <LineItems items={doc.line_items} />}

            {(doc.decisions.length > 0 || doc.field_edits.length > 0) && (
              <section className="section" aria-labelledby="hist-h">
                <header>
                  <h2 id="hist-h">History</h2>
                </header>
                <ul className="history">
                  {[
                    ...doc.decisions.map((d) => ({
                      at: d.created_at,
                      text: `${d.user_email} ${d.decision === "escalated" ? "escalated" : d.decision} it${d.comment ? `: “${d.comment}”` : ""}`,
                    })),
                    ...doc.field_edits.map((e) => ({
                      at: e.created_at,
                      text: `${e.user_email} changed ${fieldLabel(e.field)} from “${e.old_value ?? ""}” to “${e.new_value}”`,
                    })),
                  ]
                    .sort((a, b) => b.at.localeCompare(a.at))
                    .map((h, i) => (
                      <li key={i}>
                        {h.text} <span className="muted">{ago(h.at)}</span>
                      </li>
                    ))}
                </ul>
              </section>
            )}

            <details className="section" style={{ padding: "10px 16px" }}>
              <summary className="small">Processing details</summary>
              <StageTimeline stages={doc.stages} />
              <ul className="small muted" style={{ paddingLeft: 18 }}>
                {doc.messages.map((m, i) => (
                  <li key={i}>{m}</li>
                ))}
              </ul>
            </details>
          </div>
        </div>
      )}

      {doc.allowed_decisions.length > 0 && (
        <div className="decision-bar" role="region" aria-label="Decision">
          <label className="visually-hidden" htmlFor="comment">
            Comment
          </label>
          <textarea
            id="comment"
            ref={commentRef}
            placeholder={doc.status === "escalated" ? "Comment for the record" : "Comment (needed to reject or escalate)"}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
          {DECISIONS.filter((d) => doc.allowed_decisions.includes(d.id)).map((d) => (
            <button key={d.id} className={`btn ${d.cls}`} disabled={busy} onClick={() => decide(d.id)}>
              {d.label} <kbd>{d.key.toUpperCase()}</kbd>
            </button>
          ))}
          {nextId && (
            <label className="small auto-next">
              <input type="checkbox" checked={autoNext} onChange={(e) => setAutoNext(e.target.checked)} /> then open the next
            </label>
          )}
        </div>
      )}
    </>
  );
}

function Processing({ doc }: { doc: DocDetail }) {
  const pct = Math.round(doc.progress * 100);
  return (
    <div className="section" style={{ padding: 20, maxWidth: 520 }} aria-live="polite">
      <h2>{doc.status === "queued" ? "Waiting for a worker" : STAGE_LABEL[doc.stage ?? ""] ?? "Processing"}</h2>
      <div className="progress" style={{ width: "100%", margin: "12px 0 8px" }} role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <span style={{ width: `${pct}%` }} />
      </div>
      <p className="muted small">
        {doc.kind === "video" ? "Videos take up to a minute: the page is rebuilt in 3D first." : "Photos usually take a few seconds."}
        {doc.attempts > 1 && ` Attempt ${doc.attempts}.`}
      </p>
    </div>
  );
}

function Canvas({
  doc,
  view,
  selected,
  lit,
  onSelect,
  heat,
  setHeat,
  zoomRef,
}: {
  doc: DocDetail;
  view: string | null;
  selected: string | null;
  lit: Set<string>;
  onSelect: (name: string) => void;
  heat: number;
  setHeat: (v: number) => void;
  zoomRef: RefObject<ZoomHandle>;
}) {
  const { src, size } = useApiImage(doc.id, view);
  const hasOverlay = view === "page" && doc.images.includes("overlay");
  const overlay = useApiImage(doc.id, hasOverlay && heat > 0 ? "overlay" : null);
  if (!view) return <div className="canvas empty">No image was produced for this document.</div>;
  if (!src || !size) return <div className="canvas skel-block" style={{ aspectRatio: "2.2 / 1" }} aria-label="Loading image" />;
  const boxes = view === "page" ? doc.fields.filter((f) => f.bbox) : [];
  return (
    <>
      <div className="canvas">
        <ZoomCanvas ref={zoomRef} label={`${VIEWS[view] ?? view} of ${doc.filename}. Scroll to zoom, drag to move.`}>
          <svg viewBox={`0 0 ${size.w} ${size.h}`} role="img" aria-label={`${VIEWS[view]} of ${doc.filename}`}>
            <image href={src} width={size.w} height={size.h} />
            {hasOverlay && heat > 0 && overlay.src && (
              <image href={overlay.src} width={size.w} height={size.h} opacity={heat} preserveAspectRatio="none" style={{ pointerEvents: "none" }} />
            )}
            {boxes.map((f) => {
              const [x0, y0, x1, y1] = f.bbox!;
              const cls = ["box", confidenceOf(f.confidence), lit.has(f.name) && "lit", selected === f.name && "sel"]
                .filter(Boolean)
                .join(" ");
              return (
                <rect key={f.name} className={cls} x={x0} y={y0} width={x1 - x0} height={y1 - y0} rx={4} onClick={() => onSelect(f.name)}>
                  <title>
                    {fieldLabel(f.name)}: {f.corrected ?? f.normalized ?? f.value ?? "unread"} ({Math.round(f.confidence * 100)}%)
                  </title>
                </rect>
              );
            })}
          </svg>
        </ZoomCanvas>
      </div>
      <div className="legend">
        {boxes.length > 0 && (
          <>
            <span>
              <i style={{ borderColor: "var(--teal)" }} />
              Read with confidence
            </span>
            <span>
              <i style={{ borderColor: "var(--medium-line)", borderStyle: "dashed" }} />
              Low confidence: check it
            </span>
            <span>
              <i style={{ borderColor: "var(--high)", background: "var(--high-wash)" }} />
              Selected flag
            </span>
          </>
        )}
        {hasOverlay && (
          <label className="heat-slider">
            Tamper heatmap
            <input type="range" min={0} max={1} step={0.05} value={heat} onChange={(e) => setHeat(Number(e.target.value))} aria-valuetext={`${Math.round(heat * 100)}%`} />
            <kbd>H</kbd>
          </label>
        )}
      </div>
      {(view === "heatmap" || (hasOverlay && heat > 0)) && (
        <p className="legend">Red marks areas whose compression traces differ from the rest of the page: possible edits.</p>
      )}
    </>
  );
}

function StageTimeline({ stages }: { stages: DocDetail["stages"] }) {
  const total = stages.reduce((a, s) => a + s.ms, 0) || 1;
  const shown = stages.filter((s) => s.ms / total > 0.01 || !s.ok);
  return (
    <div className="timeline" style={{ marginTop: 10 }}>
      <div className="tl-bar" role="img" aria-label={`Processing took ${(total / 1000).toFixed(1)} seconds`}>
        {shown.map((s) => (
          <span key={s.stage} style={{ flexGrow: s.ms }} className={s.ok ? "" : "bad"} title={`${s.stage}: ${(s.ms / 1000).toFixed(2)} s`} />
        ))}
      </div>
      <div className="stages">
        {stages.map((s) => (
          <span key={s.stage} title={s.note} style={{ color: s.ok ? undefined : "var(--high)" }}>
            {STAGE_LABEL[s.stage] ?? s.stage} {s.ms < 1000 ? `${Math.round(s.ms)} ms` : `${(s.ms / 1000).toFixed(1)} s`}
          </span>
        ))}
      </div>
    </div>
  );
}

function ReviewSkeleton() {
  return (
    <>
      <div className="page-head">
        <div>
          <span className="skel" style={{ width: 90, height: 12 }} />
          <span className="skel" style={{ width: 260, height: 26, marginTop: 8 }} />
        </div>
      </div>
      <div className="review">
        <div className="canvas skel-block" style={{ aspectRatio: "2.2 / 1" }} />
        <div className="panel">
          <div className="section skel-block" style={{ height: 90 }} />
          <div className="section skel-block" style={{ height: 160 }} />
          <div className="section skel-block" style={{ height: 260 }} />
        </div>
      </div>
    </>
  );
}

function CheckRow({ c, active, onClick }: { c: CheckView; active: boolean; onClick: () => void }) {
  const low = c.confidence < 0.5;
  return (
    <li>
      <button aria-pressed={active} onClick={onClick}>
        <strong>{checkLabel(c.id)}</strong>
        <span className="pts small">{c.contribution != null ? `+${Math.round(c.contribution)}` : ""}</span>
        <span className="reason">
          {c.reason}
          {low && " (low confidence: probably a reading error)"}
        </span>
      </button>
    </li>
  );
}

function FieldRow({
  f,
  editable,
  value,
  selected,
  onFocus,
  onChange,
}: {
  f: FieldView;
  editable: boolean;
  value: string | undefined;
  selected: boolean;
  onFocus: () => void;
  onChange: (v: string) => void;
}) {
  const shown = value ?? shownValue(f);
  const pct = Math.round(f.confidence * 100);
  const money = MONEY_FIELDS.has(f.name);
  return (
    <div className={`frow ${selected ? "sel" : ""}`}>
      <label htmlFor={`f-${f.name}`}>{fieldLabel(f.name)}</label>
      <input
        id={`f-${f.name}`}
        type="text"
        className={value !== undefined || f.corrected ? "edited" : ""}
        value={shown}
        readOnly={!editable}
        onFocus={onFocus}
        onChange={(e) => onChange(e.target.value)}
        style={money ? { textAlign: "right" } : undefined}
      />
      <span className="read">
        {f.corrected
          ? `Corrected; read as “${f.value ?? ""}”`
          : f.value == null || f.value === ""
            ? "Not read"
            : money
              ? `₹ ${rupees(f.normalized ?? f.value)}`
              : f.name === "signature"
                ? "Signature found"
                : `Read as “${f.value}”`}
        {f.value ? ` · ${pct}%` : ""}
        {confidenceOf(f.confidence) === "low" && f.value ? " · check this" : ""}
      </span>
    </div>
  );
}

function LineItems({ items }: { items: Record<string, unknown>[] }) {
  const cols = ["desc", "hsn", "qty", "rate", "amount"];
  const head: Record<string, string> = { desc: "Description", hsn: "HSN", qty: "Qty", rate: "Rate", amount: "Amount" };
  return (
    <section className="section" aria-labelledby="items-h">
      <header>
        <h2 id="items-h">Line items</h2>
        <span className="small muted">{items.length}</span>
      </header>
      <div className="body" style={{ overflowX: "auto" }}>
        <table className="items">
          <thead>
            <tr>
              {cols.map((c) => (
                <th key={c} className={["qty", "rate", "amount"].includes(c) ? "num" : ""}>
                  {head[c]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.map((it, i) => {
              const n = (it.normalized ?? {}) as Record<string, string | null>;
              return (
                <tr key={i}>
                  {cols.map((c) => (
                    <td key={c} className={["qty", "rate", "amount"].includes(c) ? "num" : ""}>
                      {c === "rate" || c === "amount" ? rupees(n[c] ?? null) ?? "–" : c === "desc" ? String(it.desc ?? "–") : n[c] ?? "–"}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
