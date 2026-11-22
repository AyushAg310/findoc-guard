import { useEffect, useState } from "react";
import { api } from "../api";
import { STATUS_LABEL } from "../format";

// Risk and status always show an icon and a word, never colour alone (WCAG 1.4.1).

const icons = {
  low: <path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" strokeWidth="2" />,
  medium: <path d="M8 2l7 12H1z M8 6v4 M8 11.5v1" fill="none" stroke="currentColor" strokeWidth="1.6" />,
  high: (
    <g fill="none" stroke="currentColor" strokeWidth="1.8">
      <circle cx="8" cy="8" r="6.5" />
      <path d="M8 4.5v4.5 M8 10.8v1" />
    </g>
  ),
};

export function RiskBadge({ band, score }: { band: string | null; score?: number | null }) {
  if (!band) return <span className="muted small">–</span>;
  const b = band as keyof typeof icons;
  return (
    <span className={`badge risk ${b}`}>
      <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
        {icons[b]}
      </svg>
      {score != null ? `${score} · ` : ""}
      {band.charAt(0).toUpperCase() + band.slice(1)}
    </span>
  );
}

export function RiskBar({ band, score }: { band: string | null; score: number | null }) {
  if (score == null || !band) return <span className="muted small">–</span>;
  return (
    <span>
      <span className="riskbar" aria-hidden="true">
        <span className={band} style={{ width: `${Math.max(4, score)}%` }} />
      </span>
      <RiskBadge band={band} score={score} />
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  return <span className={`badge status ${status}`}>{STATUS_LABEL[status] ?? status}</span>;
}

/** An image from the API (needs the auth header, so it's fetched as a blob). */
export function useApiImage(docId: string, variant: string | null) {
  const [src, setSrc] = useState<string | null>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  useEffect(() => {
    if (!variant) return;
    let url: string | null = null;
    let cancelled = false;
    setSrc(null);
    setSize(null);
    api
      .image(docId, variant)
      .then((u) => {
        url = u;
        if (cancelled) return URL.revokeObjectURL(u);
        const img = new Image();
        img.onload = () => {
          if (!cancelled) {
            setSize({ w: img.naturalWidth, h: img.naturalHeight });
            setSrc(u);
          }
        };
        img.src = u;
      })
      .catch(() => setSrc(null));
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [docId, variant]);
  return { src, size };
}

export function Logo() {
  return (
    <svg width="26" height="18" viewBox="0 0 26 18" aria-hidden="true">
      <rect x="0.75" y="0.75" width="24.5" height="16.5" rx="2" fill="none" stroke="var(--rail-accent)" strokeWidth="1.5" />
      <path d="M4 5h11 M4 8.5h7" stroke="var(--rail-ink)" strokeWidth="1.5" />
      <path d="M4 13h2 M7.5 13h1 M10 13h3 M14.5 13h1 M17 13h2" stroke="var(--rail-accent)" strokeWidth="2" />
    </svg>
  );
}
