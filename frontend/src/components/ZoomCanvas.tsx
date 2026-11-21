import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState, type ReactNode } from "react";

export interface ZoomHandle {
  zoomBy: (factor: number) => void;
  fit: () => void;
}

/** A pan-and-zoom viewport: scroll (or pinch) to zoom around the pointer, drag to pan.
 * Clicks still reach the content when the pointer didn't move. */
export const ZoomCanvas = forwardRef<ZoomHandle, { children: ReactNode; label: string }>(function ZoomCanvas(
  { children, label },
  ref,
) {
  const box = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ s: 1, x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; vx: number; vy: number; moved: boolean } | null>(null);

  const clamp = (v: { s: number; x: number; y: number }) => {
    const el = box.current;
    if (!el) return v;
    const s = Math.min(8, Math.max(1, v.s));
    const w = el.clientWidth,
      h = el.clientHeight;
    // keep the page covering the viewport: no panning past its edges
    return { s, x: Math.min(0, Math.max(w - w * s, v.x)), y: Math.min(0, Math.max(h - h * s, v.y)) };
  };

  const zoomAt = useCallback((factor: number, cx?: number, cy?: number) => {
    setView((v) => {
      const el = box.current;
      const px = cx ?? (el ? el.clientWidth / 2 : 0);
      const py = cy ?? (el ? el.clientHeight / 2 : 0);
      const s = Math.min(8, Math.max(1, v.s * factor));
      const k = s / v.s;
      return clamp({ s, x: px - (px - v.x) * k, y: py - (py - v.y) * k });
    });
  }, []);

  // React's onWheel is passive, so preventDefault (stop the page scrolling) needs a native listener
  const scale = useRef(1);
  scale.current = view.s;
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && scale.current === 1 && e.deltaY > 0) return; // let the page scroll
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  useImperativeHandle(ref, () => ({ zoomBy: (f) => zoomAt(f), fit: () => setView({ s: 1, x: 0, y: 0 }) }), [zoomAt]);

  return (
    <div className="zoom-wrap">
      <div
        ref={box}
        className={`zoom ${view.s > 1 ? "zoomed" : ""}`}
        aria-label={label}
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false };
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d) return;
          const dx = e.clientX - d.x,
            dy = e.clientY - d.y;
          if (!d.moved && Math.hypot(dx, dy) < 4) return;
          if (!d.moved) (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
          d.moved = true;
          setView((v) => clamp({ ...v, x: d.vx + dx, y: d.vy + dy }));
        }}
        onPointerUp={() => setTimeout(() => (drag.current = null))}
        onClickCapture={(e) => {
          if (drag.current?.moved) e.stopPropagation(); // a drag isn't a click on a box
        }}
        onDoubleClick={(e) => {
          const r = box.current!.getBoundingClientRect();
          zoomAt(view.s > 2.5 ? 1 / view.s : 2, e.clientX - r.left, e.clientY - r.top);
        }}
      >
        <div className="zoom-inner" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }}>
          {children}
        </div>
      </div>
      <div className="zoom-tools" role="toolbar" aria-label="Zoom">
        <button className="icon" onClick={() => zoomAt(1 / 1.4)} aria-label="Zoom out" title="Zoom out (−)" disabled={view.s <= 1}>
          −
        </button>
        <span className="small muted" aria-live="polite">
          {Math.round(view.s * 100)}%
        </span>
        <button className="icon" onClick={() => zoomAt(1.4)} aria-label="Zoom in" title="Zoom in (+)">
          +
        </button>
        <button className="icon wide" onClick={() => setView({ s: 1, x: 0, y: 0 })} title="Fit (0)" disabled={view.s === 1}>
          Fit
        </button>
      </div>
    </div>
  );
});
