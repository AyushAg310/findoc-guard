import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type DocSummary, type Stats } from "../api";
import { DailyBars, Donut, HBars, RiskHistogram } from "../components/charts";
import { RiskBadge } from "../components/bits";
import { ago, checkLabel, compactRupees } from "../format";

/** Operations overview (SRS S6): volumes, what's flagged and why, how fast. */
export default function Overview() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [urgent, setUrgent] = useState<DocSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = () => {
      api.stats().then(setStats).catch((e) => setError(e.message));
      const p = new URLSearchParams({ limit: "100" });
      ["needs_review", "escalated"].forEach((s) => p.append("status", s));
      ["high", "medium"].forEach((b) => p.append("band", b));
      api
        .documents(p)
        .then((r) => setUrgent([...r.items].sort((a, b) => (b.risk_score ?? 0) - (a.risk_score ?? 0)).slice(0, 4)))
        .catch(() => {});
    };
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, []);

  if (error) return <p className="error-box">{error}</p>;
  if (!stats) return <OverviewSkeleton />;
  const s = stats.by_status;
  const decided = (s.approved ?? 0) + (s.rejected ?? 0) + (s.escalated ?? 0);
  const flaggedShare = stats.total ? Math.round((100 * ((stats.by_band.medium ?? 0) + (stats.by_band.high ?? 0))) / stats.total) : 0;
  const weekReceived = stats.daily.slice(-7).reduce((a, d) => a + d.received, 0);

  return (
    <>
      <div className="page-head">
        <h1>Overview</h1>
        <span className="muted small">Updates every 15 seconds</span>
      </div>

      <h2 className="ov-h">Outcomes</h2>
      <div className="kpis">
        <div className="kpi high">
          <span className="k-label">Stopped by reviewers</span>
          <strong>{compactRupees(stats.value_rejected)}</strong>
          <span className="k-note">
            {s.rejected ?? 0} document{s.rejected === 1 ? "" : "s"} rejected
          </span>
        </div>
        <div className="kpi medium">
          <span className="k-label">Sent for a closer look</span>
          <strong>{compactRupees(stats.value_flagged)}</strong>
          <span className="k-note">on documents scored medium or high</span>
        </div>
        <div className="kpi teal">
          <span className="k-label">Reviewers agreed with the score</span>
          <strong>{stats.agreement != null ? `${Math.round(stats.agreement * 100)}%` : "–"}</strong>
          <span className="k-note">
            {stats.agreement_n ? `of ${stats.agreement_n} decisions: flagged ones stopped, clean ones approved` : "no decisions yet"}
          </span>
        </div>
        <div className="kpi">
          <span className="k-label">Time to a decision</span>
          <strong>{stats.median_decision_minutes != null ? minutes(stats.median_decision_minutes) : "–"}</strong>
          <span className="k-note">median, from processed to decided</span>
        </div>
      </div>

      <h2 className="ov-h">Volume</h2>
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
            <h2>Decisions</h2>
            <span className="small muted">all time</span>
          </header>
          <div className="body">
            <Donut
              centre={String(decided)}
              parts={[
                { label: "Approved", value: s.approved ?? 0, cls: "d-low" },
                { label: "Escalated", value: s.escalated ?? 0, cls: "d-medium" },
                { label: "Rejected", value: s.rejected ?? 0, cls: "d-high" },
              ]}
            />
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
        <section className="section">
          <header>
            <h2>Most urgent, waiting</h2>
            <Link className="small" to="/?view=To+review">
              See all
            </Link>
          </header>
          <ul className="urgent">
            {urgent.map((d) => (
              <li key={d.id}>
                <span className="u-name">
                  <Link to={`/documents/${d.id}`} className="file">
                    {d.filename}
                  </Link>
                  <span className="small muted">
                    {d.top_flags[0] ? `${checkLabel(d.top_flags[0].id)} · ` : ""}
                    {ago(d.created_at)}
                  </span>
                </span>
                <RiskBadge band={d.risk_band} score={d.risk_score} />
              </li>
            ))}
            {urgent.length === 0 && <li className="muted">Nothing flagged is waiting. The queue is clear.</li>}
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

function minutes(m: number): string {
  if (m < 1) return "under 1 min";
  if (m < 90) return `${Math.round(m)} min`;
  return `${(m / 60).toFixed(1)} h`;
}
