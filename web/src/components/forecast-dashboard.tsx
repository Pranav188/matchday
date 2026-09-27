"use client";

import { useEffect, useState } from "react";

type Fixture = {
  id: string;
  matchweek: number;
  kickoff: string;
  date: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  finished: boolean;
};
type Schedule = { season: string; updated_at: string; fixtures: Fixture[] };
type Forecast = {
  fixture_id: string;
  predicted_code: "H" | "D" | "A";
  probabilities: { code: string; label: string; probability: number }[];
  data_through: string;
  explanation: {
    rules: { feature: string; label: string; value: number; threshold: number; operator: string; imputed: boolean }[];
    leaf_matches: number;
  };
  evaluation: {
    season: string;
    matches: number;
    accuracy: number;
    baseline_accuracy: number;
    macro_f1: number;
    confusion_matrix: number[][];
    class_order: string[];
  } | null;
};

const outcomeNames: Record<string, string> = { H: "Home", D: "Draw", A: "Away" };
const numberFormat = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 2 });

function ModelEvidence({ forecast }: { forecast: Forecast }) {
  const [open, setOpen] = useState(false);
  const { explanation, evaluation } = forecast;
  const contentId = `evidence-${forecast.fixture_id}`;
  return <div className="model-evidence">
    <button className="evidence-toggle" aria-expanded={open} aria-controls={contentId} onClick={() => setOpen((current) => !current)}>
      Model evidence<span className="disclosure-chevron" aria-hidden="true" />
    </button>
    <div className={`disclosure${open ? " disclosure-open" : ""}`} id={contentId} inert={!open} aria-hidden={!open}>
    <div className="disclosure-inner"><div className="evidence-grid">
      <section className="tree-evidence" aria-label="Prediction decision path">
        <h4>How the tree reached this prediction</h4>
        <p className="evidence-caption">Decision Tree · results through {forecast.data_through}</p>
        <ol className="decision-rules">
          {explanation.rules.map((rule, index) => <li key={`${rule.feature}-${index}`}>
            <span>{rule.label}{rule.imputed ? " (median filled)" : ""}</span>
            <strong>{numberFormat.format(rule.value)} {rule.operator} {numberFormat.format(rule.threshold)}</strong>
          </li>)}
        </ol>
        <p className="evidence-caption">{explanation.leaf_matches.toLocaleString("en-GB")} training matches reached this leaf. Probabilities use their class-weighted outcomes.</p>
      </section>
      {evaluation && <section className="test-evidence" aria-label="Historical model evaluation">
        <h4>Tested on {evaluation.season.replace("-", "/")}</h4>
        <p className="evidence-caption">{evaluation.matches} held-out matches · before deployment refit</p>
        <p className="test-scores"><strong>{(evaluation.accuracy * 100).toFixed(1)}% accuracy</strong><span>Always-home baseline {(evaluation.baseline_accuracy * 100).toFixed(1)}%</span></p>
        <table className="confusion-matrix">
          <caption>Confusion matrix · match counts</caption>
          <thead><tr><th scope="col">Actual ↓<br />Predicted →</th>{evaluation.class_order.map((code) => <th scope="col" key={code}>{outcomeNames[code]}</th>)}</tr></thead>
          <tbody>{evaluation.confusion_matrix.map((row, rowIndex) => <tr key={evaluation.class_order[rowIndex]}>
            <th scope="row">{outcomeNames[evaluation.class_order[rowIndex]]}</th>
            {row.map((count, columnIndex) => <td className={rowIndex === columnIndex ? "matrix-correct" : ""} key={columnIndex}>{count}</td>)}
          </tr>)}</tbody>
        </table>
        <p className="evidence-caption">Highlighted cells are correct predictions. This measures past performance; it does not validate this fixture’s outcome.</p>
      </section>}
    </div></div></div>
  </div>;
}

const clubNames: Record<string, string> = {
  "Man City": "Manchester City",
  "Man United": "Manchester United",
  "Nott'm Forest": "Nottingham Forest",
  Tottenham: "Tottenham Hotspur",
};
const clubName = (name: string) => clubNames[name] ?? name;
const dateFormat = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/London", weekday: "long", day: "numeric", month: "long",
});
const timeFormat = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/London", hour: "2-digit", minute: "2-digit", hour12: false,
});

export function ForecastDashboard() {
  const [schedule, setSchedule] = useState<Schedule | null>(null);
  const [week, setWeek] = useState(6);
  const [team, setTeam] = useState("");
  const [loadError, setLoadError] = useState("");
  const [retry, setRetry] = useState(0);
  const [forecasts, setForecasts] = useState<Record<string, Forecast>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [pending, setPending] = useState<Record<string, boolean>>({});
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const controller = new AbortController();
    setLoadError("");
    async function load() {
      try {
        const response = await fetch("/api/fixtures", { signal: controller.signal });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error ?? "Could not load fixtures.");
        const data = payload as Schedule;
        setSchedule(data);
        const next = data.fixtures.find((fixture) => !fixture.finished && new Date(fixture.kickoff).getTime() > Date.now());
        setWeek(next?.matchweek ?? 38);
      } catch (error) {
        if (!controller.signal.aborted) setLoadError(error instanceof Error ? error.message : "Could not load fixtures.");
      }
    }
    void load();
    return () => controller.abort();
  }, [retry]);

  async function predict(fixture: Fixture) {
    if (forecasts[fixture.id]) {
      setExpanded((current) => ({ ...current, [fixture.id]: !current[fixture.id] }));
      return;
    }
    setPending((current) => ({ ...current, [fixture.id]: true }));
    setErrors((current) => ({ ...current, [fixture.id]: "" }));
    try {
      const response = await fetch("/api/forecast", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fixture_id: fixture.id }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "Prediction unavailable. Try again.");
      setForecasts((current) => ({ ...current, [fixture.id]: payload as Forecast }));
      setExpanded((current) => ({ ...current, [fixture.id]: true }));
    } catch (error) {
      setErrors((current) => ({ ...current, [fixture.id]: error instanceof Error ? error.message : "Prediction unavailable. Try again." }));
    } finally {
      setPending((current) => ({ ...current, [fixture.id]: false }));
    }
  }

  const teams = schedule ? Array.from(new Set(schedule.fixtures.flatMap((fixture) => [fixture.home_team, fixture.away_team]))).sort() : [];
  const fixtures = schedule?.fixtures.filter((fixture) => fixture.matchweek === week && (!team || fixture.home_team === team || fixture.away_team === team)) ?? [];
  const groups = new Map<string, Fixture[]>();
  for (const fixture of fixtures) {
    const day = dateFormat.format(new Date(fixture.kickoff));
    groups.set(day, [...(groups.get(day) ?? []), fixture]);
  }

  return (
    <main className="season-page">
      <header className="masthead">
        <a className="wordmark" href="/" aria-label="Matchday home">matchday<span aria-hidden="true">.</span></a>
        <span className="masthead-note">Match predictions</span>
      </header>
      <section aria-labelledby="season-heading" className="fixtures-section">
        <div className="season-heading">
          <div><p className="competition">Premier League</p><h1 id="season-heading">2026<span className="season-slash">/</span>27</h1></div>
        </div>
        <div className="fixture-toolbar">
          <div className="week-navigation">
            <button className="week-arrow" aria-label="Previous matchweek" disabled={week === 1 || !schedule} onClick={() => setWeek((current) => current - 1)}>‹</button>
            <label className="week-select"><span className="sr-only">Matchweek</span>
              <select value={week} disabled={!schedule} onChange={(event) => setWeek(Number(event.target.value))}>
                {Array.from({ length: 38 }, (_, index) => <option key={index + 1} value={index + 1}>Matchweek {index + 1}</option>)}
              </select>
            </label>
            <button className="week-arrow" aria-label="Next matchweek" disabled={week === 38 || !schedule} onClick={() => setWeek((current) => current + 1)}>›</button>
          </div>
          <label className="team-filter"><span className="sr-only">Filter by club</span>
            <select value={team} disabled={!schedule} onChange={(event) => setTeam(event.target.value)}>
              <option value="">All clubs</option>
              {teams.map((club) => <option key={club} value={club}>{clubName(club)}</option>)}
            </select>
          </label>
        </div>
        {loadError ? <div className="empty-state" role="alert"><p>{loadError}</p><button className="predict-button" onClick={() => setRetry((value) => value + 1)}>Try again</button></div>
          : !schedule ? <p className="empty-state" role="status">Loading fixtures…</p>
          : fixtures.length === 0 ? <p className="empty-state">No match for this club in matchweek {week}.</p>
          : <div className="fixture-list">
            {Array.from(groups, ([day, matches]) => <section className="date-group" key={day} aria-label={day}>
              <h2>{day}</h2>
              <ul>
                {matches.map((fixture) => {
                  const forecast = forecasts[fixture.id];
                  const isExpanded = !!expanded[fixture.id];
                  const started = new Date(fixture.kickoff).getTime() <= Date.now();
                  return <li className={`fixture${isExpanded ? " fixture-open" : ""}`} key={fixture.id}>
                    <div className="fixture-row">
                      <time className="kickoff" dateTime={fixture.kickoff}>{timeFormat.format(new Date(fixture.kickoff))}</time>
                      <div className="matchup">
                        <span className="home-club">{clubName(fixture.home_team)}</span>
                        <span className={`match-divider${fixture.finished ? " score" : ""}`}>{fixture.finished ? `${fixture.home_score} : ${fixture.away_score}` : <span aria-label="versus">v</span>}</span>
                        <span className="away-club">{clubName(fixture.away_team)}</span>
                      </div>
                      {fixture.finished ? <span className="match-status">Full time</span> : started ? <span className="match-status">Started</span> :
                        <button className="predict-button" disabled={pending[fixture.id]} aria-busy={!!pending[fixture.id]} aria-label={`${pending[fixture.id] ? "Loading prediction" : forecast ? "View prediction" : "Predict"}: ${clubName(fixture.home_team)} versus ${clubName(fixture.away_team)}`} aria-expanded={isExpanded} aria-controls={`forecast-${fixture.id}`} onClick={() => void predict(fixture)}>
                          {pending[fixture.id] ? "Loading" : forecast ? "Prediction" : "Predict"}
                          {pending[fixture.id] ? <span className="prediction-spinner" aria-hidden="true" /> : forecast ? <span className="disclosure-chevron" aria-hidden="true" /> : null}
                        </button>}
                    </div>
                    {errors[fixture.id] && <p className="forecast-error" role="alert">{errors[fixture.id]}</p>}
                    {forecast && <div className={`disclosure${isExpanded ? " disclosure-open" : ""}`} id={`forecast-${fixture.id}`} inert={!isExpanded} aria-hidden={!isExpanded}>
                    <div className="disclosure-inner"><div className="forecast">
                      <div className="forecast-heading"><h3>{forecast.predicted_code === "D" ? "Draw" : `${clubName(forecast.predicted_code === "H" ? fixture.home_team : fixture.away_team)} to win`}</h3></div>
                      <div className="outcomes">
                        {forecast.probabilities.map((outcome) => <div className={`outcome${forecast.predicted_code === outcome.code ? " outcome-leading" : ""}`} key={outcome.code}>
                          <div className="outcome-label"><span>{outcome.label}</span><strong>{Math.round(outcome.probability * 100)}<span>%</span></strong></div>
                          <div className="probability-track"><div style={{ width: `${outcome.probability * 100}%` }} /></div>
                        </div>)}
                      </div>
                      <p className="forecast-note">Estimated probabilities, not a guaranteed result.</p>
                      <ModelEvidence forecast={forecast} />
                    </div></div></div>}
                  </li>;
                })}
              </ul>
            </section>)}
          </div>}
        <footer className="schedule-footer"><span>Kickoff times in UK time. Fixtures may change.</span>{schedule && <a href="https://fixturedownload.com/results/epl-2026" target="_blank" rel="noreferrer">Fixture source ↗</a>}</footer>
      </section>
    </main>
  );
}
