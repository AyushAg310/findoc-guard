// Result-first pieces (v0.9.2): the score gauge, what the score is made of, where a document is
// in the pipeline, and a before/after slider between the capture and the flattened page.
import { useEffect, useState } from "react";
import { PIPELINE_STEPS, checkLabel, stepIndex } from "../format";
import { useApiImage } from "./bits";

const reduceMotion = () => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** Counts from 0 up to `value` once, in about half a second (instantly with reduced motion). */
function useCountUp(value: number, ms = 600): number {
  const [shown, setShown] = useState(reduceMotion() ? value : 0);
  useEffect(() => {
    if (reduceMotion()) return setShown(value);
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / ms);
      setShown(Math.round(value * (1 - (1 - k) ** 3)));
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, ms]);
  return shown;
}

/** A half-circle gauge from 0 to 100 with the 30 and 70 band edges marked. */
export function ScoreGauge({ score, band, size = 132 }: { score: number; band: string; size?: number }) {
  const shown = useCountUp(score);
  const r = 52;
  const len = Math.PI * r;
  const at = (v: number) => {
    const a = Math.PI * (1 - v / 100);
    return [62 + r * Math.cos(a), 62 - r * Math.sin(a)];
  };
  return (
    <svg className={`gauge ${band}`} width={size} height={size * 0.6} viewBox="0 0 124 74" role="img" aria-label={`Risk score ${score} of 100, ${band}`}>
      <path d="M10 62 A52 52 0 0 1 114 62" className="track" />
      <path d="M10 62 A52 52 0 0 1 114 62" className="value" strokeDasharray={len} strokeDashoffset={len * (1 - shown / 100)} />
      {[30, 70].map((v) => {
        const [x, y] = at(v);
        const [x2, y2] = [62 + (x - 62) * 0.78, 62 + (y - 62) * 0.78];
        return <line key={v} x1={x} y1={y} x2={x2} y2={y2} className="edge" />;
      })}
      <text x="62" y="60" textAnchor="middle" className="num">
        {shown}
      </text>
    </svg>
  );
}

/** A small ring for cards and table rows. */
export function ScoreRing({ score, band }: { score: number; band: string }) {
  const r = 16;
  const len = 2 * Math.PI * r;
  return (
    <svg className={`ring ${band}`} width="44" height="44" viewBox="0 0 44 44" role="img" aria-label={`Risk ${score}, ${band}`}>
      <circle cx="22" cy="22" r={r} className="track" />
      <circle cx="22" cy="22" r={r} className="value" strokeDasharray={len} strokeDashoffset={len * (1 - Math.min(100, score) / 100)} transform="rotate(-90 22 22)" />
      <text x="22" y="26.5" textAnchor="middle">
        {score}
      </text>
    </svg>
  );
}

export interface Part {
  id: string;
  points: number;
}

const SEGMENT = ["seg1", "seg2", "seg3", "seg4"];

/** What the score is made of: one segment per check that fired, sized by its points.
 * Clicking a segment (or its legend row) selects that check. */
export function RiskBreakdown({
  parts,
  score,
  forcedBy,
  active,
  onPick,
}: {
  parts: Part[];
  score: number;
  forcedBy?: string | null;
  active: string | null;
  onPick: (id: string) => void;
}) {
  const sum = parts.reduce((a, p) => a + p.points, 0);
  if (!parts.length) return null;
  // contributions can add up past 100 (the score is capped): scale the bar to the score
  const scale = sum > 0 ? Math.min(score, 100) / Math.max(sum, Math.min(score, 100)) : 0;
  return (
    <div className="breakdown">
      <div className="bd-bar" role="group" aria-label="Score breakdown">
        {parts.map((p, i) => (
          <button
            key={p.id}
            className={`${SEGMENT[i % SEGMENT.length]} ${active === p.id ? "on" : ""}`}
            style={{ width: `${Math.max(2, p.points * scale)}%` }}
            aria-pressed={active === p.id}
            title={`${checkLabel(p.id)}: +${Math.round(p.points)}`}
            onClick={() => onPick(p.id)}
          >
            <span className="visually-hidden">
              {checkLabel(p.id)}, {Math.round(p.points)} points
            </span>
          </button>
        ))}
        <span className="bd-edge" style={{ left: "30%" }} aria-hidden="true" />
        <span className="bd-edge" style={{ left: "70%" }} aria-hidden="true" />
      </div>
      <div className="bd-scale small muted" aria-hidden="true">
        <span>0</span>
        <span style={{ left: "30%" }}>30</span>
        <span style={{ left: "70%" }}>70</span>
        <span style={{ right: 0 }}>100</span>
      </div>
      {forcedBy && sum < 70 && <p className="small muted">Forced to high by {checkLabel(forcedBy)}, whatever the total.</p>}
    </div>
  );
}

/** Six steps of the pipeline, lit up as a document moves through them. */
export function PipelineSteps({ status, stage, compact = false }: { status: string; stage: string | null; compact?: boolean }) {
  const at = stepIndex(status, stage);
  return (
    <ol className={`steps ${compact ? "compact" : ""}`} aria-label="Processing steps">
      {PIPELINE_STEPS.map((s, i) => {
        const state = i < at ? "done" : i === at ? "active" : "todo";
        return (
          <li key={s.label} className={state} aria-current={state === "active" ? "step" : undefined}>
            <span className="dot" aria-hidden="true">
              {state === "done" ? "✓" : ""}
            </span>
            <span className="lbl">{s.label}</span>
          </li>
        );
      })}
    </ol>
  );
}

/** Drag (or use the arrow keys) to compare the capture with the flattened page. */
export function BeforeAfter({ docId }: { docId: string }) {
  const before = useApiImage(docId, "original");
  const after = useApiImage(docId, "flat");
  const [pos, setPos] = useState(50);
  if (!before.src || !after.src || !after.size) return <div className="canvas skel-block" style={{ aspectRatio: "2.2 / 1" }} aria-label="Loading images" />;
  return (
    <div className="compare canvas" style={{ aspectRatio: `${after.size.w} / ${after.size.h}` }}>
      <img src={after.src} alt="Flattened page" draggable={false} />
      <img src={before.src} alt="Original capture" draggable={false} className="top" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
      <span className="handle" style={{ left: `${pos}%` }} aria-hidden="true">
        <span>‹ ›</span>
      </span>
      <span className="tag l">Capture</span>
      <span className="tag r">Flattened</span>
      <input
        type="range"
        min={0}
        max={100}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        aria-label="Compare the capture with the flattened page"
        aria-valuetext={`${pos}% capture`}
      />
    </div>
  );
}
