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
  home_team: string;
  away_team: string;
  date: string;
  fixture_id: string;
  predicted_code: "H" | "D" | "A";
  probabilities: { code: string; label: string; probability: number }[];
  data_through: string;
  model: string;
  model_version: string;
  predicted_at: string;
  factors: Record<string, number | null>;
  learned_context: string[];
  availability: {
    status: string;
    observed_at?: string;
    players: { player: string; team: string; status: string; news: string }[];
  };
  goals: {
    expected_home_goals: number;
    expected_away_goals: number;
    scorelines: { home: number; away: number; probability: number }[];
  };

  explanation: {
    rules: {
      feature: string;
      label: string;
      value: number;
      threshold: number;
      operator: string;
      imputed: boolean;
    }[];
    leaf_matches: number | null;
    method: "tree" | "linear" | "global";
    calibrated?: boolean;
    rival?: string;
    contributions?: { label: string; value: number }[];
  };
  evaluation: {
    season: string;
    matches: number;
    accuracy: number;
    baseline_accuracy: number;
    macro_f1: number;
    log_loss: number;
    brier_score: number;
    protocol: string;
    confusion_matrix: number[][];
    class_order: string[];
  } | null;
};

const outcomeNames: Record<string, string> = {
  H: "Home",
  D: "Draw",
  A: "Away",
};
const numberFormat = new Intl.NumberFormat("en-GB", {
  maximumFractionDigits: 2,
});

function ModelEvidence({ forecast }: { forecast: Forecast }) {
  const [open, setOpen] = useState(false);
  const { explanation, evaluation } = forecast;
  const contentId = `evidence-${forecast.fixture_id}`;
  return (
    <div className="model-evidence">
      <button
        className="evidence-toggle"
        aria-expanded={open}
        aria-controls={contentId}
        onClick={() => setOpen((current) => !current)}
      >
        Model evidence
        <span className="disclosure-chevron" aria-hidden="true" />
      </button>
      <div
        className={`disclosure${open ? " disclosure-open" : ""}`}
        id={contentId}
        inert={!open}
        aria-hidden={!open}
      >
        <div className="disclosure-inner">
          <div className="evidence-grid">
            <section
              className="tree-evidence"
              aria-label="Prediction decision path"
            >
              <h4>
                {explanation.method === "tree"
                  ? "Decision path"
                  : explanation.method === "linear"
                    ? "Factors behind this prediction"
                    : "Model feature importance"}
              </h4>
              <p className="evidence-caption">
                {forecast.model.replaceAll("_", " ")} · results through{" "}
                {forecast.data_through}
              </p>
              {explanation.method === "tree" ? (
                <ol className="decision-rules">
                  {explanation.rules.map((rule, index) => (
                    <li key={`${rule.feature}-${index}`}>
                      <span>
                        {rule.label}
                        {rule.imputed ? " (median filled)" : ""}
                      </span>
                      <strong>
                        {numberFormat.format(rule.value)} {rule.operator}{" "}
                        {numberFormat.format(rule.threshold)}
                      </strong>
                    </li>
                  ))}
                </ol>
              ) : (
                <ul className="contribution-list">
                  {explanation.contributions?.map((item) => (
                    <li key={item.label}>
                      <span>{item.label}</span>
                      <strong>
                        {item.value > 0 ? "+" : ""}
                        {numberFormat.format(item.value)}
                      </strong>
                    </li>
                  ))}
                </ul>
              )}
              <p className="evidence-caption">
                {explanation.method === "tree"
                  ? `${explanation.leaf_matches?.toLocaleString("en-GB")} training matches reached this leaf. ${explanation.calibrated ? "Displayed probabilities are calibrated on a later season." : "Probabilities use class-weighted outcomes in this leaf."}`
                  : explanation.method === "linear"
                    ? `Contributions to log odds versus ${outcomeNames[explanation.rival ?? "D"].toLowerCase()}. Positive values favour the prediction; intercept and other features also contribute.`
                    : "These are global importances, not explanations of this individual match."}
              </p>
              <p className="evidence-caption">
                Model {forecast.model_version.slice(0, 8)} · saved{" "}
                {new Date(forecast.predicted_at).toLocaleString("en-GB")}
              </p>
            </section>
            {evaluation && (
              <section
                className="test-evidence"
                aria-label="Historical model evaluation"
              >
                <h4>Tested on {evaluation.season.replace("-", "/")}</h4>
                <p className="evidence-caption">
                  {evaluation.matches} matches · retrospective benchmark
                </p>
                <p className="test-scores">
                  <strong>
                    {(evaluation.accuracy * 100).toFixed(1)}% accuracy
                  </strong>
                  <span>
                    Always-home baseline{" "}
                    {(evaluation.baseline_accuracy * 100).toFixed(1)}%
                  </span>
                </p>
                <p className="evidence-caption">
                  Macro F1 {evaluation.macro_f1.toFixed(3)} · Log loss{" "}
                  {evaluation.log_loss.toFixed(3)} · Brier score{" "}
                  {evaluation.brier_score.toFixed(3)}
                </p>
                <table className="confusion-matrix">
                  <caption>Confusion matrix · match counts</caption>
                  <thead>
                    <tr>
                      <th scope="col">
                        Actual ↓<br />
                        Predicted →
                      </th>
                      {evaluation.class_order.map((code) => (
                        <th scope="col" key={code}>
                          {outcomeNames[code]}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {evaluation.confusion_matrix.map((row, rowIndex) => (
                      <tr key={evaluation.class_order[rowIndex]}>
                        <th scope="row">
                          {outcomeNames[evaluation.class_order[rowIndex]]}
                        </th>
                        {row.map((count, columnIndex) => (
                          <td
                            className={
                              rowIndex === columnIndex ? "matrix-correct" : ""
                            }
                            key={columnIndex}
                          >
                            {count}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="evidence-caption">
                  Highlighted cells are correct predictions. This season was
                  previously inspected; saved future forecasts provide the
                  prospective test.
                </p>
              </section>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const clubNames: Record<string, string> = {
  "Man City": "Manchester City",
  "Man United": "Manchester United",
  "Nott'm Forest": "Nottingham Forest",
  Tottenham: "Tottenham Hotspur",
};
const clubName = (name: string) => clubNames[name] ?? name;
const dateFormat = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/London",
  weekday: "long",
  day: "numeric",
  month: "long",
});
const timeFormat = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/London",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
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
  const [view, setView] = useState<"fixtures" | "history">("fixtures");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const controller = new AbortController();
    setLoadError("");
    async function load() {
      try {
        const response = await fetch("/api/fixtures", {
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok)
          throw new Error(payload.error ?? "Could not load fixtures.");
        const data = payload as Schedule;
        setSchedule(data);
        const next = data.fixtures.find(
          (fixture) =>
            !fixture.finished &&
            new Date(fixture.kickoff).getTime() > Date.now(),
        );
        setWeek(next?.matchweek ?? 38);
      } catch (error) {
        if (!controller.signal.aborted)
          setLoadError(
            error instanceof Error ? error.message : "Could not load fixtures.",
          );
      }
    }
    void load();
    return () => controller.abort();
  }, [retry]);

  async function predict(fixture: Fixture) {
    if (forecasts[fixture.id]) {
      setExpanded((current) => ({
        ...current,
        [fixture.id]: !current[fixture.id],
      }));
      return;
    }
    setPending((current) => ({ ...current, [fixture.id]: true }));
    setErrors((current) => ({ ...current, [fixture.id]: "" }));
    try {
      const response = await fetch("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fixture_id: fixture.id }),
      });
      const payload = await response.json();
      if (!response.ok)
        throw new Error(payload.error ?? "Prediction unavailable. Try again.");
      setForecasts((current) => ({
        ...current,
        [fixture.id]: payload as Forecast,
      }));
      setExpanded((current) => ({ ...current, [fixture.id]: true }));
    } catch (error) {
      setErrors((current) => ({
        ...current,
        [fixture.id]:
          error instanceof Error
            ? error.message
            : "Prediction unavailable. Try again.",
      }));
    } finally {
      setPending((current) => ({ ...current, [fixture.id]: false }));
    }
  }

  const teams = schedule
    ? Array.from(
        new Set(
          schedule.fixtures.flatMap((fixture) => [
            fixture.home_team,
            fixture.away_team,
          ]),
        ),
      ).sort()
    : [];
  const fixtures =
    schedule?.fixtures.filter(
      (fixture) =>
        fixture.matchweek === week &&
        (!team || fixture.home_team === team || fixture.away_team === team),
    ) ?? [];
  const groups = new Map<string, Fixture[]>();
  for (const fixture of fixtures) {
    const day = dateFormat.format(new Date(fixture.kickoff));
    groups.set(day, [...(groups.get(day) ?? []), fixture]);
  }

  return (
    <main className="season-page">
      <header className="masthead">
        <a className="wordmark" href="/" aria-label="Matchday home">
          matchday<span aria-hidden="true">.</span>
        </a>
        <span className="masthead-note">Match predictions</span>
      </header>
      <section aria-labelledby="season-heading" className="fixtures-section">
        <div className="season-heading">
          <div>
            <p className="competition">Premier League</p>
            <h1 id="season-heading">
              2026<span className="season-slash">/</span>27
            </h1>
          </div>
        </div>
        <nav className="view-tabs" aria-label="Matchday views">
          <button
            aria-pressed={view === "fixtures"}
            onClick={() => setView("fixtures")}
          >
            Fixtures
          </button>
          <button
            aria-pressed={view === "history"}
            onClick={() => setView("history")}
          >
            Past predictions
          </button>
        </nav>
        {view === "history" ? (
          <PredictionHistory />
        ) : (
          <>
            <div className="fixture-toolbar">
              <div className="week-navigation">
                <button
                  className="week-arrow"
                  aria-label="Previous matchweek"
                  disabled={week === 1 || !schedule}
                  onClick={() => setWeek((current) => current - 1)}
                >
                  ‹
                </button>
                <label className="week-select">
                  <span className="sr-only">Matchweek</span>
                  <select
                    value={week}
                    disabled={!schedule}
                    onChange={(event) => setWeek(Number(event.target.value))}
                  >
                    {Array.from({ length: 38 }, (_, index) => (
                      <option key={index + 1} value={index + 1}>
                        Matchweek {index + 1}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className="week-arrow"
                  aria-label="Next matchweek"
                  disabled={week === 38 || !schedule}
                  onClick={() => setWeek((current) => current + 1)}
                >
                  ›
                </button>
              </div>
              <label className="team-filter">
                <span className="sr-only">Filter by club</span>
                <select
                  value={team}
                  disabled={!schedule}
                  onChange={(event) => setTeam(event.target.value)}
                >
                  <option value="">All clubs</option>
                  {teams.map((club) => (
                    <option key={club} value={club}>
                      {clubName(club)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {loadError ? (
              <div className="empty-state" role="alert">
                <p>{loadError}</p>
                <button
                  className="predict-button"
                  onClick={() => setRetry((value) => value + 1)}
                >
                  Try again
                </button>
              </div>
            ) : !schedule ? (
              <p className="empty-state" role="status">
                Loading fixtures…
              </p>
            ) : fixtures.length === 0 ? (
              <p className="empty-state">
                No match for this club in matchweek {week}.
              </p>
            ) : (
              <div className="fixture-list">
                {Array.from(groups, ([day, matches]) => (
                  <section className="date-group" key={day} aria-label={day}>
                    <h2>{day}</h2>
                    <ul>
                      {matches.map((fixture) => {
                        const forecast = forecasts[fixture.id];
                        const isExpanded = !!expanded[fixture.id];
                        const started =
                          new Date(fixture.kickoff).getTime() <= Date.now();
                        return (
                          <li
                            className={`fixture${isExpanded ? " fixture-open" : ""}`}
                            key={fixture.id}
                          >
                            <div className="fixture-row">
                              <time
                                className="kickoff"
                                dateTime={fixture.kickoff}
                              >
                                {timeFormat.format(new Date(fixture.kickoff))}
                              </time>
                              <div className="matchup">
                                <span className="home-club">
                                  {clubName(fixture.home_team)}
                                </span>
                                <span
                                  className={`match-divider${fixture.finished ? " score" : ""}`}
                                >
                                  {fixture.finished ? (
                                    `${fixture.home_score} : ${fixture.away_score}`
                                  ) : (
                                    <span aria-label="versus">v</span>
                                  )}
                                </span>
                                <span className="away-club">
                                  {clubName(fixture.away_team)}
                                </span>
                              </div>
                              {fixture.finished ? (
                                <span className="match-status">Full time</span>
                              ) : started ? (
                                <span className="match-status">Started</span>
                              ) : (
                                <button
                                  className="predict-button"
                                  disabled={pending[fixture.id]}
                                  aria-busy={!!pending[fixture.id]}
                                  aria-label={`${pending[fixture.id] ? "Loading prediction" : forecast ? "View prediction" : "Predict"}: ${clubName(fixture.home_team)} versus ${clubName(fixture.away_team)}`}
                                  aria-expanded={isExpanded}
                                  aria-controls={`forecast-${fixture.id}`}
                                  onClick={() => void predict(fixture)}
                                >
                                  {pending[fixture.id]
                                    ? "Loading"
                                    : forecast
                                      ? "Prediction"
                                      : "Predict"}
                                  {pending[fixture.id] ? (
                                    <span
                                      className="prediction-spinner"
                                      aria-hidden="true"
                                    />
                                  ) : forecast ? (
                                    <span
                                      className="disclosure-chevron"
                                      aria-hidden="true"
                                    />
                                  ) : null}
                                </button>
                              )}
                            </div>
                            {errors[fixture.id] && (
                              <p className="forecast-error" role="alert">
                                {errors[fixture.id]}
                              </p>
                            )}
                            {forecast && (
                              <div
                                className={`disclosure${isExpanded ? " disclosure-open" : ""}`}
                                id={`forecast-${fixture.id}`}
                                inert={!isExpanded}
                                aria-hidden={!isExpanded}
                              >
                                <div className="disclosure-inner">
                                  <div className="forecast">
                                    <div className="forecast-heading">
                                      <h3>
                                        {forecast.predicted_code === "D"
                                          ? "Draw"
                                          : `${clubName(forecast.predicted_code === "H" ? fixture.home_team : fixture.away_team)} to win`}
                                      </h3>
                                    </div>
                                    <div className="outcomes">
                                      {forecast.probabilities.map((outcome) => (
                                        <div
                                          className={`outcome${forecast.predicted_code === outcome.code ? " outcome-leading" : ""}`}
                                          key={outcome.code}
                                        >
                                          <div className="outcome-label">
                                            <span>{outcome.label}</span>
                                            <strong>
                                              {Math.round(
                                                outcome.probability * 100,
                                              )}
                                              <span>%</span>
                                            </strong>
                                          </div>
                                          <div className="probability-track">
                                            <div
                                              style={{
                                                width: `${outcome.probability * 100}%`,
                                              }}
                                            />
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                    <p className="forecast-note">
                                      Estimated probabilities, not a guaranteed
                                      result.
                                    </p>
                                    <div className="goal-summary">
                                      <span>Expected goals</span>
                                      <strong>
                                        {forecast.goals.expected_home_goals.toFixed(
                                          1,
                                        )}{" "}
                                        –{" "}
                                        {forecast.goals.expected_away_goals.toFixed(
                                          1,
                                        )}
                                      </strong>
                                      <p>
                                        Likely scores:{" "}
                                        {forecast.goals.scorelines
                                          .map(
                                            (score) =>
                                              `${score.home}–${score.away} (${Math.round(score.probability * 100)}%)`,
                                          )
                                          .join(", ")}
                                      </p>
                                      <small>
                                        Separate Poisson model; scorelines are
                                        estimates.
                                      </small>
                                    </div>
                                    <MatchFactors forecast={forecast} />
                                    <ModelEvidence forecast={forecast} />
                                  </div>
                                </div>
                              </div>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  </section>
                ))}
              </div>
            )}
          </>
        )}
        <footer className="schedule-footer">
          <span>Kickoff times in UK time. Fixtures may change.</span>
          {schedule && (
            <a
              href="https://fixturedownload.com/results/epl-2026"
              target="_blank"
              rel="noreferrer"
            >
              Fixture source ↗
            </a>
          )}
        </footer>
      </section>
    </main>
  );
}

const importedFactors = [
  ["Unavailable minutes share", "missing_minutes_share"],
  ["Absent attackers", "missing_attackers"],
  ["Absent defenders", "missing_defenders"],
  ["Absent goalkeepers", "missing_goalkeepers"],
  ["Suspended starters", "suspended_starters"],
  ["Lineup minutes share", "lineup_minutes_share"],
  ["Days with current manager", "manager_days"],
  ["All matches in 7 days", "all_matches_last7"],
  ["All matches in 14 days", "all_matches_last14"],
];

function MatchFactors({ forecast }: { forecast: Forecast }) {
  const f = forecast.factors;
  const [open, setOpen] = useState(false);
  const contentId = `context-${forecast.fixture_id}`;
  return (
    <div className="match-factors">
      <button
        className="evidence-toggle"
        aria-expanded={open}
        aria-controls={contentId}
        onClick={() => setOpen((current) => !current)}
      >
        Match context
        <span className="disclosure-chevron" aria-hidden="true" />
      </button>
      <div
        className={`disclosure${open ? " disclosure-open" : ""}`}
        id={contentId}
        inert={!open}
        aria-hidden={!open}
      >
        <div className="disclosure-inner">
          <table className="factor-table">
            <caption>Recent completed league matches</caption>
            <thead>
              <tr>
                <th>Factor</th>
                <th>Home</th>
                <th>Away</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Shots per match", "shots_last5"],
                ["Shots on target", "shots_on_target_last5"],
                ["Matches in 7 days", "matches_last7"],
                ["Matches in 14 days", "matches_last14"],
              ].map(([label, key]) => (
                <tr key={key}>
                  <th scope="row">{label}</th>
                  <td>
                    {f[`home_${key}`] == null
                      ? "Unknown"
                      : numberFormat.format(f[`home_${key}`]!)}
                  </td>
                  <td>
                    {f[`away_${key}`] == null
                      ? "Unknown"
                      : numberFormat.format(f[`away_${key}`]!)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {importedFactors.some(
            ([, key]) => f[`home_${key}`] != null || f[`away_${key}`] != null,
          ) && (
            <table className="factor-table">
              <caption>Imported pre-match reports</caption>
              <thead>
                <tr>
                  <th scope="col">Factor</th>
                  <th scope="col">Home</th>
                  <th scope="col">Away</th>
                </tr>
              </thead>
              <tbody>
                {importedFactors
                  .filter(
                    ([, key]) =>
                      f[`home_${key}`] != null || f[`away_${key}`] != null,
                  )
                  .map(([label, key]) => (
                    <tr key={key}>
                      <th scope="row">{label}</th>
                      {["home", "away"].map((side) => (
                        <td key={side}>
                          {f[`${side}_${key}`] == null
                            ? "Unknown"
                            : key.endsWith("share")
                              ? `${(f[`${side}_${key}`]! * 100).toFixed(0)}%`
                              : numberFormat.format(f[`${side}_${key}`]!)}
                        </td>
                      ))}
                    </tr>
                  ))}
              </tbody>
            </table>
          )}
          <p className="evidence-caption">
            {forecast.learned_context.length
              ? "Imported availability, lineup, manager or all-competition features are included where covered by training data."
              : "Injuries, lineups, manager changes and cup workload have not been learned from historical data yet."}
          </p>
          {forecast.availability.status === "context_only" ? (
            <>
              <h4>Reported availability</h4>
              <p className="evidence-caption">
                Public snapshot,{" "}
                {forecast.availability.observed_at?.slice(0, 10)}. Context only;
                it does not change this forecast.
              </p>
              {forecast.availability.players.length ? (
                <ul className="availability-list">
                  {forecast.availability.players.map((player) => (
                    <li key={`${player.team}-${player.player}`}>
                      <strong>{player.player}</strong>
                      <span>
                        {player.team} ·{" "}
                        {player.status === "s"
                          ? "Suspended"
                          : player.status === "i"
                            ? "Injured"
                            : "Doubtful"}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="evidence-caption">
                  No flagged players in this snapshot. Coverage is not a medical
                  confirmation.
                </p>
              )}
            </>
          ) : (
            <p className="evidence-caption">
              Player availability is not current for this fixture.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function PredictionHistory() {
  type Archived = Forecast & {
    archive_id: number;
    actual_code: string | null;
    home_score: number | null;
    away_score: number | null;
  };
  const [data, setData] = useState<{
    forecasts: Archived[];
    total: number;
    settled: number;
    metrics: { accuracy: number; log_loss: number } | null;
  } | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch("/api/history", {
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok)
          throw new Error(payload.error ?? "History unavailable.");
        setData(payload);
      } catch (e) {
        if (!controller.signal.aborted)
          setError(e instanceof Error ? e.message : "History unavailable.");
      }
    }
    void load();
    return () => controller.abort();
  }, []);
  if (error)
    return (
      <p className="empty-state" role="alert">
        {error}
      </p>
    );
  if (!data)
    return (
      <p className="empty-state" role="status">
        Loading predictions…
      </p>
    );
  if (!data.total)
    return (
      <p className="empty-state">
        Predict an upcoming fixture to start your history. Forecasts are saved
        before kickoff.
      </p>
    );
  return (
    <section className="prediction-history" aria-label="Saved predictions">
      <p className="history-summary">
        {data.total} saved fixtures · {data.settled} completed
        {data.metrics
          ? ` · ${(data.metrics.accuracy * 100).toFixed(1)}% accuracy · log loss ${data.metrics.log_loss.toFixed(3)}`
          : ""}
      </p>
      <p className="evidence-caption">
        Each fixture is scored using its earliest saved forecast.
      </p>
      <ul>
        {data.forecasts.map((forecast) => (
          <li className="history-row" key={forecast.archive_id}>
            <div>
              <strong>
                {clubName(forecast.home_team)} v {clubName(forecast.away_team)}
              </strong>
              <span>
                {forecast.date} · saved {forecast.predicted_at.slice(0, 10)} ·
                model {forecast.model_version.slice(0, 8)}
              </span>
            </div>
            <div>
              <span>
                Predicted {outcomeNames[forecast.predicted_code].toLowerCase()}
              </span>
              <strong>
                {forecast.actual_code
                  ? `${forecast.home_score}–${forecast.away_score} · ${forecast.actual_code === forecast.predicted_code ? "Correct" : "Missed"}`
                  : "Awaiting result"}
              </strong>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
