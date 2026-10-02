import type { Metadata } from "next";
import { MatchdayHeader } from "@/components/matchday-header";
import { StandingsTable } from "@/components/standings-table";
import styles from "./standings.module.css";

export const metadata: Metadata = {
  title: "Final Premier League table | Matchday",
  description:
    "Projected Premier League 2026/27 standings from completed results and remaining match probabilities.",
};

export default function StandingsPage() {
  return (
    <main className="season-page">
      <MatchdayHeader active="standings" />
      <section className={styles.section} aria-labelledby="standings-heading">
        <div className="season-heading">
          <div>
            <p className="competition">Premier League</p>
            <h1>
              2026<span className="season-slash">/</span>27
            </h1>
          </div>
        </div>
        <h2 id="standings-heading" className={styles.heading}>
          Projected final table
        </h2>
        <p className={styles.intro}>
          Completed results, plus expected points from the remaining fixtures.
        </p>
        <StandingsTable />
      </section>
    </main>
  );
}
