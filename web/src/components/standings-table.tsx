"use client";

import { useEffect, useState } from "react";
import styles from "@/app/standings/standings.module.css";

type Projection = {
  completed_matches: number;
  remaining_matches: number;
  data_through: string;
  model_version: string;
  rows: {
    position: number;
    team: string;
    played: number;
    remaining: number;
    current_points: number;
    projected_points: number;
  }[];
};
const clubNames: Record<string, string> = {
  "Man City": "Manchester City",
  "Man United": "Manchester United",
  "Nott'm Forest": "Nottingham Forest",
  Tottenham: "Tottenham Hotspur",
};
const pointsFormat = new Intl.NumberFormat("en-GB", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

export function StandingsTable() {
  const [projection, setProjection] = useState<Projection | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      setError("");
      try {
        const response = await fetch("/api/standings", {
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok)
          throw new Error(
            payload.error ?? "The table is unavailable. Try again.",
          );
        setProjection(payload as Projection);
      } catch (error) {
        if (!controller.signal.aborted)
          setError(
            error instanceof Error
              ? error.message
              : "The table is unavailable. Try again.",
          );
      }
    }
    void load();
    return () => controller.abort();
  }, [retry]);

  if (error)
    return (
      <div className="empty-state" role="alert">
        <p>{error}</p>
        <button
          className="predict-button"
          onClick={() => setRetry((current) => current + 1)}
        >
          Try again
        </button>
      </div>
    );
  if (!projection)
    return (
      <p className="empty-state" role="status">
        Calculating the final table…
      </p>
    );
  return (
    <>
      <div className={styles.tableContainer}>
        <table className={styles.table}>
          <caption className="sr-only">
            Projected Premier League 2026/27 final standings, ordered by
            expected points
          </caption>
          <thead>
            <tr>
              <th scope="col">
                <span className="sr-only">Position</span>
              </th>
              <th scope="col">Club</th>
              <th scope="col">Left</th>
              <th scope="col">
                Now<span className="sr-only">, current points</span>
              </th>
              <th scope="col">
                Final<span className="sr-only">, projected points</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {projection.rows.map((row) => (
              <tr key={row.team}>
                <td className={styles.position}>{row.position}</td>
                <th scope="row">{clubNames[row.team] ?? row.team}</th>
                <td>{row.remaining}</td>
                <td>{row.current_points}</td>
                <td className={styles.points}>
                  {pointsFormat.format(row.projected_points)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className={styles.notes}>
        <p>
          {projection.completed_matches} matches completed.{" "}
          {projection.remaining_matches} remaining.
        </p>
        <p>
          Final points are averages, so decimals are expected. Equal totals
          share a position. Team form stays fixed until new results arrive.
        </p>
        <p className={styles.provenance}>
          Form through {projection.data_through}. Model{" "}
          {projection.model_version.slice(0, 8)}.
        </p>
      </div>
    </>
  );
}
