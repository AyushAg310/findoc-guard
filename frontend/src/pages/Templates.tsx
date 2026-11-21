import { useEffect, useRef, useState, type FormEvent, type PointerEvent as RPointerEvent } from "react";
import { api, type ChequeTemplate, type TemplateField } from "../api";
import { useAuth } from "../auth";
import { fieldLabel } from "../format";
import { Dialog, useToast } from "../ui";

/** The cheque banks the system recognises, and (for admins) adding a bank from a blank leaf. */
export default function Templates() {
  const { user } = useAuth();
  const toast = useToast();
  const [items, setItems] = useState<ChequeTemplate[] | null>(null);
  const [editing, setEditing] = useState(false);
  const [preview, setPreview] = useState<ChequeTemplate | null>(null);
  const load = () => api.templates().then(setItems).catch((e) => toast.show(e.message, "error"));
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const custom = items?.filter((t) => !t.builtin) ?? [];
  const builtin = items?.filter((t) => t.builtin) ?? [];
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Cheque banks</h1>
          <p className="muted small">
            A cheque is read by matching it to its bank's blank leaf. {items ? `${items.length} banks are known.` : ""}
          </p>
        </div>
        {user?.role === "admin" && (
          <button className="btn primary" onClick={() => setEditing(true)}>
            Add a bank
          </button>
        )}
      </div>

      {custom.length > 0 && (
        <>
          <h2 className="sub">Your banks</h2>
          <div className="bank-grid">
            {custom.map((t) => (
              <BankCard key={t.id} t={t} onOpen={() => setPreview(t)} />
            ))}
          </div>
        </>
      )}
      <h2 className="sub">Built-in banks (fictional, for synthetic data)</h2>
      <div className="bank-grid">
        {items === null
          ? Array.from({ length: 6 }, (_, i) => <div key={i} className="bank-card skel-block" />)
          : builtin.map((t) => <BankCard key={t.id} t={t} onOpen={() => setPreview(t)} />)}
      </div>

      {preview && (
        <Dialog title={`${preview.name} (${preview.id})`} onClose={() => setPreview(null)} wide>
          <TemplateView t={preview} />
          <p className="small muted" style={{ marginTop: 8 }}>
            The account number and MICR areas are covered in this image.
          </p>
          {!preview.builtin && user?.role === "admin" && (
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 12 }}>
              <button
                className="btn reject"
                onClick={async () => {
                  try {
                    await api.deleteTemplate(preview.id);
                    toast.show(`Deleted ${preview.name}`);
                    setPreview(null);
                    load();
                  } catch (e) {
                    toast.show(e instanceof Error ? e.message : "Delete failed", "error");
                  }
                }}
              >
                Delete this bank
              </button>
            </div>
          )}
        </Dialog>
      )}
      {editing && (
        <Editor
          onClose={() => setEditing(false)}
          onSaved={(t) => {
            setEditing(false);
            toast.show(`Added ${t.name}: cheques from this bank are recognised from the next upload`);
            load();
          }}
        />
      )}
    </>
  );
}

function useTemplateImage(id: string) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    let url: string | null = null;
    let live = true;
    api
      .templateImage(id)
      .then((u) => (live ? setSrc((url = u)) : URL.revokeObjectURL(u)))
      .catch(() => {});
    return () => {
      live = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [id]);
  return src;
}

function BankCard({ t, onOpen }: { t: ChequeTemplate; onOpen: () => void }) {
  const src = useTemplateImage(t.id);
  return (
    <button className="bank-card" onClick={onOpen}>
      <span className="thumb" style={{ aspectRatio: `${t.size[0]} / ${t.size[1]}` }}>
        {src ? <img src={src} alt="" /> : <span className="skel" style={{ width: "100%", height: "100%" }} />}
      </span>
      <span className="bank-name">{t.name}</span>
      <span className="small muted">
        {t.id} · {t.fields.length} fields
      </span>
    </button>
  );
}

const KIND_CLASS: Record<string, string> = { handwritten: "k-hand", printed: "k-print", signature: "k-sign" };

function TemplateView({ t }: { t: ChequeTemplate }) {
  const src = useTemplateImage(t.id);
  const [w, h] = t.size;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="tpl">
      {src && <image href={src} width={w} height={h} />}
      {t.fields.map((f) => (
        <g key={f.name} className={KIND_CLASS[f.kind ?? ""] ?? "k-print"}>
          <rect x={f.box[0] * w} y={f.box[1] * h} width={(f.box[2] - f.box[0]) * w} height={(f.box[3] - f.box[1]) * h} rx={4} />
          <text x={f.box[0] * w + 6} y={f.box[1] * h + 20}>
            {fieldLabel(f.name)}
          </text>
        </g>
      ))}
    </svg>
  );
}

// ------------------------------------------------------------------ editor

type Drag = { name: string; mode: "move" | "resize"; start: [number, number]; box: TemplateField["box"] };

function Editor({ onClose, onSaved }: { onClose: () => void; onSaved: (t: ChequeTemplate) => void }) {
  const toast = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [src, setSrc] = useState<string | null>(null);
  const [size, setSize] = useState<[number, number]>([1624, 736]);
  const [fields, setFields] = useState<TemplateField[]>([]);
  const [standard, setStandard] = useState<TemplateField[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<Drag | null>(null);

  useEffect(() => {
    api
      .standardLayout()
      .then((f) => {
        setStandard(f);
        setFields(f);
      })
      .catch((e) => toast.show(e.message, "error"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function choose(f: File | undefined) {
    if (!f) return;
    const url = URL.createObjectURL(f);
    const img = new Image();
    img.onload = () => {
      setSize([img.naturalWidth, img.naturalHeight]);
      setSrc(url);
      setFile(f);
    };
    img.src = url;
  }

  const point = (e: RPointerEvent): [number, number] => {
    const m = svg.current!.getScreenCTM()!.inverse();
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m);
    return [p.x / size[0], p.y / size[1]];
  };

  function onDown(e: RPointerEvent, name: string, mode: Drag["mode"]) {
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    const f = fields.find((x) => x.name === name)!;
    drag.current = { name, mode, start: point(e), box: [...f.box] as TemplateField["box"] };
    setSelected(name);
  }
  function onMove(e: RPointerEvent) {
    const d = drag.current;
    if (!d) return;
    const [px, py] = point(e);
    const dx = px - d.start[0],
      dy = py - d.start[1];
    const [x0, y0, x1, y1] = d.box;
    const next: TemplateField["box"] =
      d.mode === "move"
        ? [x0 + clampD(dx, -x0, 1 - x1), y0 + clampD(dy, -y0, 1 - y1), x1 + clampD(dx, -x0, 1 - x1), y1 + clampD(dy, -y0, 1 - y1)]
        : [x0, y0, Math.min(1, Math.max(x0 + 0.02, x1 + dx)), Math.min(1, Math.max(y0 + 0.02, y1 + dy))];
    setFields((fs) => fs.map((f) => (f.name === d.name ? { ...f, box: next } : f)));
  }
  function nudge(dx: number, dy: number) {
    if (!selected) return;
    setFields((fs) =>
      fs.map((f) =>
        f.name === selected
          ? { ...f, box: [f.box[0] + dx, f.box[1] + dy, f.box[2] + dx, f.box[3] + dy] as TemplateField["box"] }
          : f,
      ),
    );
  }

  async function save(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    try {
      onSaved(await api.createTemplate(name, code, fields, file));
    } catch (err) {
      toast.show(err instanceof Error ? err.message : "Saving failed", "error");
    } finally {
      setBusy(false);
    }
  }

  const [w, h] = size;
  return (
    <Dialog title="Add a bank" onClose={onClose} wide>
      {!src ? (
        <label className="drop">
          <strong>Choose a photo or scan of a blank cheque leaf</strong>
          <span className="muted small">
            Crop it to the leaf, straight and flat. Use a leaf that isn't filled in. The account number and MICR areas are
            covered whenever the image is shown.
          </span>
          <input type="file" accept="image/*" className="visually-hidden" onChange={(e) => choose(e.target.files?.[0])} />
        </label>
      ) : (
        <form onSubmit={save} className="editor">
          <div className="editor-canvas">
            <svg
              ref={svg}
              viewBox={`0 0 ${w} ${h}`}
              className="tpl editing"
              onPointerMove={onMove}
              onPointerUp={() => (drag.current = null)}
              onPointerDown={() => setSelected(null)}
            >
              <image href={src} width={w} height={h} />
              {fields.map((f) => {
                const [x0, y0, x1, y1] = f.box;
                const sel = selected === f.name;
                return (
                  <g key={f.name} className={`${KIND_CLASS[f.kind ?? ""] ?? "k-print"} ${sel ? "sel" : ""}`}>
                    <rect
                      x={x0 * w}
                      y={y0 * h}
                      width={(x1 - x0) * w}
                      height={(y1 - y0) * h}
                      rx={4}
                      onPointerDown={(e) => onDown(e, f.name, "move")}
                    />
                    <text x={x0 * w + 6} y={y0 * h + Math.max(14, h * 0.025)} style={{ fontSize: Math.max(12, h * 0.022) }}>
                      {fieldLabel(f.name)}
                    </text>
                    <rect
                      className="handle"
                      x={x1 * w - h * 0.018}
                      y={y1 * h - h * 0.018}
                      width={h * 0.036}
                      height={h * 0.036}
                      onPointerDown={(e) => onDown(e, f.name, "resize")}
                    />
                  </g>
                );
              })}
            </svg>
            <p className="small muted">
              Drag a box to move it, drag its corner to resize. With a box selected, the arrow keys nudge it.
            </p>
          </div>
          <div
            className="editor-side"
            onKeyDown={(e) => {
              const step = e.shiftKey ? 0.01 : 0.002;
              const m: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
              if (selected && m[e.key] && (e.target as HTMLElement).tagName !== "INPUT") {
                e.preventDefault();
                nudge(...m[e.key]);
              }
            }}
          >
            <label className="field">
              Bank name
              <input type="text" required minLength={2} maxLength={60} value={name} onChange={(e) => setName(e.target.value)} />
            </label>
            <label className="field">
              Code (4 letters, like the start of its IFSC)
              <input
                type="text"
                required
                pattern="[A-Za-z]{4}"
                maxLength={4}
                value={code}
                onChange={(e) => setCode(e.target.value.toUpperCase())}
              />
            </label>
            <div className="field-list" role="listbox" aria-label="Fields">
              {fields.map((f) => (
                <button
                  type="button"
                  key={f.name}
                  role="option"
                  aria-selected={selected === f.name}
                  className={KIND_CLASS[f.kind ?? ""] ?? "k-print"}
                  onClick={() => setSelected(f.name)}
                >
                  <i /> {fieldLabel(f.name)}
                </button>
              ))}
            </div>
            <button type="button" className="btn" onClick={() => setFields(standard)}>
              Reset to the standard layout
            </button>
            <button className="btn primary" disabled={busy} type="submit">
              {busy ? "Saving…" : "Save bank"}
            </button>
          </div>
        </form>
      )}
    </Dialog>
  );
}

const clampD = (d: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, d));
