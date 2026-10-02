import Link from "next/link";
import styles from "./matchday-header.module.css";

export function MatchdayHeader({
  active,
}: {
  active: "fixtures" | "standings";
}) {
  return (
    <header className="masthead">
      <Link className="wordmark" href="/" aria-label="Matchday home">
        matchday<span aria-hidden="true">.</span>
      </Link>
      <nav className={styles.navigation} aria-label="Main navigation">
        <Link
          href="/"
          aria-current={active === "fixtures" ? "page" : undefined}
        >
          Fixtures
        </Link>
        <Link
          href="/standings"
          aria-current={active === "standings" ? "page" : undefined}
        >
          Final table
        </Link>
      </nav>
    </header>
  );
}
