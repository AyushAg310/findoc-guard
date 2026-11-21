// Small hand-made SVG charts (no chart library): grouped bars over days, a histogram and
// horizontal bars. Each has a text alternative for screen readers and a hover readout.
import { useState } from "react";

export function DailyBars({ data }: { data: { date: string; received: number; decided: number; flagged: number }[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 640,
    H = 200,
    pad = { l: 28, r: 8, t: 12, b: 26 };
  const max = Math.max(1, ...data.flatMap((d) => [d.received, d.decided]));
  const nice = niceMax(max);
  const bw = (W - pad.l - pad.r) / data.length;
  const y = (v: number) => pad.t + (H - pad.t - pad.b) * (1 - v / nice);
  const label = (iso: string) => new Date(iso + "T00:00:00").toLocaleDateString("en-IN", { day: "numeric", month: "short" });
  const h = hover != null ? data[hover] : null;
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Documents per day for the last ${data.length} days`}>
        {[0, nice / 2, nice].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={pad.l - 6} y={y(v) + 4} className="tick" textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {data.map((d, i) => {
          const x = pad.l + i * bw;
          const w = Math.max(3, bw / 2 - 3);
          return (
            <g key={d.date} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={x} y={pad.t} width={bw} height={H - pad.t - pad.b} className={hover === i ? "hoverband" : "hit"} />
              <rect x={x + bw / 2 - w - 1} y={y(d.received)} width={w} height={y(0) - y(d.received)} rx={2} className="s1" />
              <rect x={x + bw / 2 + 1} y={y(d.decided)} width={w} height={y(0) - y(d.decided)} rx={2} className="s2" />
              {i % 2 === data.length % 2 && (
                <text x={x + bw / 2} y={H - 8} className="tick" textAnchor="middle">
                  {label(d.date)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption>
        <span>
          <i className="sw s1" /> Received
        </span>
        <span>
          <i className="sw s2" /> Decided
        </span>
        <span className="readout">
          {h ? `${label(h.date)}: ${h.received} received, ${h.decided} decided, ${h.flagged} flagged` : "Hover a day for its numbers"}
        </span>
      </figcaption>
    </figure>
  );
}

export function RiskHistogram({ counts }: { counts: number[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 320,
    H = 160,
    pad = { l: 8, r: 8, t: 10, b: 24 };
  const max = Math.max(1, ...counts);
  const bw = (W - pad.l - pad.r) / counts.length;
  const band = (i: number) => (i >= 7 ? "high" : i >= 3 ? "medium" : "low");
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Risk scores: ${counts.map((c, i) => `${i * 10}-${i * 10 + 9}: ${c}`).join(", ")}`}>
        {counts.map((c, i) => {
          const h = ((H - pad.t - pad.b) * c) / max;
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={pad.l + i * bw} y={pad.t} width={bw} height={H - pad.t - pad.b} className={hover === i ? "hoverband" : "hit"} />
              <rect x={pad.l + i * bw + 2} y={H - pad.b - h} width={bw - 4} height={h} rx={2} className={`risk-${band(i)}`} />
            </g>
          );
        })}
        <text x={pad.l} y={H - 6} className="tick">
          0
        </text>
        <text x={pad.l + 3 * bw} y={H - 6} className="tick" textAnchor="middle">
          30
        </text>
        <text x={pad.l + 7 * bw} y={H - 6} className="tick" textAnchor="middle">
          70
        </text>
        <text x={W - pad.r} y={H - 6} className="tick" textAnchor="end">
          100
        </text>
      </svg>
      <figcaption>
        <span className="readout">
          {hover != null ? `Score ${hover * 10}–${hover * 10 + 9}: ${counts[hover]} document${counts[hover] === 1 ? "" : "s"}` : "Bands: low < 30 ≤ medium < 70 ≤ high"}
        </span>
      </figcaption>
    </figure>
  );
}

export function HBars({ items, label }: { items: [string, number][]; label: (k: string) => string }) {
  const max = Math.max(1, ...items.map(([, v]) => v));
  if (!items.length) return <p className="muted small">No check has fired yet.</p>;
  return (
    <ul className="hbars">
      {items.map(([k, v]) => (
        <li key={k}>
          <span className="name">{label(k)}</span>
          <span className="bar" aria-hidden="true">
            <span style={{ width: `${(100 * v) / max}%` }} />
          </span>
          <span className="val">{v}</span>
        </li>
      ))}
    </ul>
  );
}

function niceMax(v: number): number {
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 5, 10]) if (v <= m * p) return m * p;
  return 10 * p;
}

/** A ring split into parts (decisions made), with the total in the middle and a legend. */
export function Donut({ parts, centre }: { parts: { label: string; value: number; cls: string }[]; centre: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const total = parts.reduce((a, p) => a + p.value, 0);
  const r = 46,
    len = 2 * Math.PI * r;
  let offset = 0;
  return (
    <figure className="chart donut">
      <svg viewBox="0 0 120 120" role="img" aria-label={parts.map((p) => `${p.label}: ${p.value}`).join(", ")}>
        <circle cx="60" cy="60" r={r} className="d-track" />
        {total > 0 &&
          parts.map((p, i) => {
            const share = p.value / total;
            const el = (
              <circle
                key={p.label}
                cx="60"
                cy="60"
                r={r}
                className={`d-seg ${p.cls} ${hover === i ? "on" : ""}`}
                strokeDasharray={`${Math.max(0, share * len - (share < 1 ? 2 : 0))} ${len}`}
                strokeDashoffset={-offset * len}
                transform="rotate(-90 60 60)"
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
              />
            );
            offset += share;
            return el;
          })}
        <text x="60" y="58" textAnchor="middle" className="d-num">
          {hover != null ? parts[hover].value : centre}
        </text>
        <text x="60" y="74" textAnchor="middle" className="tick">
          {hover != null ? parts[hover].label.toLowerCase() : "decided"}
        </text>
      </svg>
      <figcaption>
        {parts.map((p, i) => (
          <span key={p.label} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
            <i className={`sw ${p.cls}`} /> {p.label} {p.value}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
