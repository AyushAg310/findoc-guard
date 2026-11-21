import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type DocSummary, type Stats } from "../api";
import { DailyBars, HBars, RiskHistogram } from "../components/charts";
import { RiskBadge, StatusBadge } from "../components/bits";
import { ago, checkLabel } from "../format";

/** Operations overview (SRS S6): volumes, what's flagged and why, how fast. */
export default function Overview() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [urgent, setUrgent] = useState<DocSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = () => {
      api.stats().then(setStats).catch((e) => setError(e.message));
      const p = new URLSearchParams({ limit: "5" });
      ["needs_review", "escalated"].forEach((s) => p.append("status", s));
      p.append("band", "high");
      api.documents(p).then((r) => setUrgent(r.items)).catch(() => {});
    };
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, []);

  if (error) return <p className="error-box">{error}</p>;
  if (!stats) return <OverviewSkeleton />;
  const s = stats.by_status;
  const decided = (s.approved ?? 0) + (s.rejected ?? 0);
  const flaggedShare = stats.total ? Math.round((100 * ((stats.by_band.medium ?? 0) + (stats.by_band.high ?? 0))) / stats.total) : 0;
  const weekReceived = stats.daily.slice(-7).reduce((a, d) => a + d.received, 0);

  return (
    <>
      <div className="page-head">
        <h1>Overview</h1>
        <span className="muted small">Updates every 15 seconds</span>
      </div>

      <div className="counts">
        <div>
          <strong>{stats.open_reviews}</strong>
          <span>
            <Link to="/">waiting for a decision</Link>
          </span>
        </div>
        <div>
          <strong>{weekReceived}</strong>
          <span>received in the last 7 days</span>
        </div>
        <div>
          <strong>{flaggedShare}%</strong>
          <span>flagged medium or high</span>
        </div>
        <div>
          <strong>{stats.median_seconds != null ? `${stats.median_seconds} s` : "–"}</strong>
          <span>median processing time</span>
        </div>
        <div>
          <strong>{decided}</strong>
          <span>decided of {stats.total}</span>
        </div>
      </div>

      <div className="ov-grid">
        <section className="section span2">
          <header>
            <h2>Documents per day</h2>
            <span className="small muted">last 14 days</span>
          </header>
          <div className="body">
            <DailyBars data={stats.daily} />
          </div>
        </section>
        <section className="section">
          <header>
            <h2>Risk scores</h2>
          </header>
          <div className="body">
            <RiskHistogram counts={stats.risk_histogram} />
          </div>
        </section>
        <section className="section">
          <header>
            <h2>Why documents are flagged</h2>
            <span className="small muted">checks that fired</span>
          </header>
          <div className="body">
            <HBars items={Object.entries(stats.by_check)} label={checkLabel} />
          </div>
        </section>
        <section className="section span2">
          <header>
            <h2>High risk, waiting</h2>
            <Link className="small" to="/?view=To+review&band=high">
              See all
            </Link>
          </header>
          <ul className="urgent">
            {urgent.map((d) => (
              <li key={d.id}>
                <Link to={`/documents/${d.id}`} className="file">
                  {d.filename}
                </Link>
                <StatusBadge status={d.status} />
                <RiskBadge band={d.risk_band} score={d.risk_score} />
                <span className="muted small">{ago(d.created_at)}</span>
              </li>
            ))}
            {urgent.length === 0 && <li className="muted">No high-risk document is waiting.</li>}
          </ul>
        </section>
      </div>
    </>
  );
}

function OverviewSkeleton() {
  return (
    <>
      <div className="page-head">
        <h1>Overview</h1>
      </div>
      <div className="counts">
        {Array.from({ length: 5 }, (_, i) => (
          <div key={i}>
            <span className="skel" style={{ width: 48, height: 32 }} />
            <span className="skel" style={{ width: 110, height: 12, marginTop: 6 }} />
          </div>
        ))}
      </div>
      <div className="ov-grid">
        <div className="section span2 skel-block" />
        <div className="section skel-block" />
        <div className="section skel-block" />
      </div>
    </>
  );
}
