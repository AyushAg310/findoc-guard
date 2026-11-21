// App-wide UI services: theme, toasts, the command palette and the shortcuts dialog.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { api, type DocSummary } from "./api";
import { isReviewer, useAuth } from "./auth";
import { RiskBadge, StatusBadge } from "./components/bits";

// ------------------------------------------------------------------ theme

export type Theme = "light" | "dark";
const THEME_KEY = "findoc.theme";

function storedTheme(): Theme {
  try {
    const t = localStorage.getItem(THEME_KEY);
    if (t === "light" || t === "dark") return t;
  } catch {
    /* storage blocked */
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(storedTheme);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* storage blocked */
    }
  }, [theme]);
  return [theme, () => setTheme((t) => (t === "dark" ? "light" : "dark"))];
}

// ------------------------------------------------------------------ toasts

interface Toast {
  id: number;
  kind: "ok" | "error" | "info";
  text: string;
  action?: { label: string; run: () => void };
}
interface Toaster {
  show: (text: string, kind?: Toast["kind"], action?: Toast["action"]) => void;
}
const ToastContext = createContext<Toaster>({ show: () => {} });
export const useToast = () => useContext(ToastContext);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const next = useRef(1);
  const show = useCallback<Toaster["show"]>((text, kind = "ok", action) => {
    const id = next.current++;
    setToasts((t) => [...t.slice(-3), { id, kind, text, action }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === "error" ? 8000 : 4500);
  }, []);
  const value = useMemo(() => ({ show }), [show]);
  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`}>
            <span>{t.text}</span>
            {t.action && (
              <button
                className="textlink"
                onClick={() => {
                  t.action!.run();
                  setToasts((all) => all.filter((x) => x.id !== t.id));
                }}
              >
                {t.action.label}
              </button>
            )}
            <button className="x" aria-label="Dismiss" onClick={() => setToasts((all) => all.filter((x) => x.id !== t.id))}>
              ×
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

// ------------------------------------------------------------------ dialog

export function Dialog({ title, onClose, children, wide }: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    ref.current?.querySelector<HTMLElement>("input, button, [tabindex]")?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      prev?.focus?.();
    };
  }, [onClose]);
  return (
    <div className="scrim" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`dialog ${wide ? "wide" : ""}`} role="dialog" aria-modal="true" aria-label={title} ref={ref}>
        <header>
          <h2>{title}</h2>
          <button className="x" aria-label="Close" onClick={onClose}>
            ×
          </button>
        </header>
        {children}
      </div>
    </div>
  );
}

export const SHORTCUTS: [string, string][] = [
  ["Ctrl K", "Search documents and jump anywhere"],
  ["?", "Show these shortcuts"],
  ["J / K", "Next / previous document (review)"],
  ["A", "Approve"],
  ["R", "Reject (needs a comment)"],
  ["E", "Escalate (needs a comment)"],
  ["+ / − / 0", "Zoom in / out / fit the page"],
  ["H", "Show or hide the tamper heatmap"],
  ["G then Q / U / O", "Go to the queue / upload / overview"],
];

export function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  return (
    <Dialog title="Keyboard shortcuts" onClose={onClose}>
      <table className="keys">
        <tbody>
          {SHORTCUTS.map(([k, what]) => (
            <tr key={k}>
              <td>
                {k.split(" ").map((p, i) => (p === "/" || p === "then" ? <span key={i}> {p} </span> : <kbd key={i}>{p}</kbd>))}
              </td>
              <td>{what}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Dialog>
  );
}

// ------------------------------------------------------------------ command palette

interface Command {
  id: string;
  label: string;
  hint?: string;
  run: () => void;
  doc?: DocSummary;
}

export function CommandPalette({ onClose, toggleTheme }: { onClose: () => void; toggleTheme: () => void }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [q, setQ] = useState("");
  const [docs, setDocs] = useState<DocSummary[]>([]);
  const [active, setActive] = useState(0);
  const go = useCallback(
    (to: string) => {
      navigate(to);
      onClose();
    },
    [navigate, onClose],
  );

  useEffect(() => {
    const t = setTimeout(() => {
      const p = new URLSearchParams({ limit: "8" });
      if (q.trim()) p.set("q", q.trim());
      api.documents(p).then((r) => setDocs(r.items)).catch(() => setDocs([]));
    }, 150);
    return () => clearTimeout(t);
  }, [q]);

  const commands = useMemo<Command[]>(() => {
    const nav: Command[] = [
      ...(isReviewer(user)
        ? [
            { id: "queue", label: "Go to the review queue", hint: "G Q", run: () => go("/") },
            { id: "overview", label: "Go to the overview", hint: "G O", run: () => go("/overview") },
            { id: "templates", label: "Go to cheque banks", run: () => go("/templates") },
          ]
        : []),
      { id: "upload", label: "Upload documents", hint: "G U", run: () => go("/upload") },
      ...(user?.role === "admin" ? [{ id: "admin", label: "Users and audit", run: () => go("/admin") }] : []),
      { id: "theme", label: "Switch light / dark theme", run: () => (toggleTheme(), onClose()) },
      { id: "out", label: "Sign out", run: logout },
    ];
    const needle = q.trim().toLowerCase();
    const filtered = needle ? nav.filter((c) => c.label.toLowerCase().includes(needle)) : nav;
    const docCmds = docs.map((d) => ({ id: d.id, label: d.filename, doc: d, run: () => go(`/documents/${d.id}`) }));
    return [...docCmds, ...filtered];
  }, [docs, q, user, go, logout, toggleTheme, onClose]);

  useEffect(() => setActive(0), [q, docs.length]);

  return (
    <div className="scrim top" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Command palette">
        <input
          autoFocus
          type="search"
          placeholder="Search documents by file name, or type a command…"
          value={q}
          role="combobox"
          aria-expanded="true"
          aria-controls="palette-list"
          aria-activedescendant={commands[active] ? `cmd-${commands[active].id}` : undefined}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") onClose();
            else if (e.key === "ArrowDown") {
              e.preventDefault();
              setActive((a) => Math.min(commands.length - 1, a + 1));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setActive((a) => Math.max(0, a - 1));
            } else if (e.key === "Enter") commands[active]?.run();
          }}
        />
        <ul id="palette-list" role="listbox">
          {commands.map((c, i) => (
            <li
              key={c.id}
              id={`cmd-${c.id}`}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => {
                e.preventDefault();
                c.run();
              }}
            >
              {c.doc ? (
                <>
                  <span className="file">{c.label}</span>
                  <span className="meta">
                    <StatusBadge status={c.doc.status} /> <RiskBadge band={c.doc.risk_band} score={c.doc.risk_score} />
                  </span>
                </>
              ) : (
                <>
                  <span>{c.label}</span>
                  {c.hint && <kbd>{c.hint}</kbd>}
                </>
              )}
            </li>
          ))}
          {commands.length === 0 && <li className="muted">Nothing matches “{q}”.</li>}
        </ul>
        <footer className="small muted">
          <kbd>↑</kbd> <kbd>↓</kbd> to move · <kbd>Enter</kbd> to open · <kbd>Esc</kbd> to close
        </footer>
      </div>
    </div>
  );
}

/** Global keys: Ctrl/Cmd+K palette, ? shortcuts, G-then-letter navigation. */
export function useGlobalKeys(openPalette: () => void, openHelp: () => void) {
  const navigate = useNavigate();
  useEffect(() => {
    let g = 0;
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement;
      const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName) || t.isContentEditable;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        openPalette();
        return;
      }
      if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "?") {
        e.preventDefault();
        openHelp();
      } else if (e.key.toLowerCase() === "g") g = Date.now();
      else if (Date.now() - g < 1200) {
        const to = { q: "/", u: "/upload", o: "/overview" }[e.key.toLowerCase()];
        if (to) navigate(to);
        g = 0;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate, openPalette, openHelp]);
}

/** Ids of the queue the reviewer came from, so J/K and "approve and next" can move on. */
export const reviewList = {
  set(ids: string[]) {
    try {
      sessionStorage.setItem("findoc.queue", JSON.stringify(ids));
    } catch {
      /* ignore */
    }
  },
  get(): string[] {
    try {
      return JSON.parse(sessionStorage.getItem("findoc.queue") ?? "[]");
    } catch {
      return [];
    }
  },
};
