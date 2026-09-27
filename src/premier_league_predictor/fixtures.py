"""Stored season schedule, independent of the completed-result training data."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

SEASON = "2026-27"
SOURCE_URL = "https://fixturedownload.com/feed/json/epl-2026"
TEAM_ALIASES = {"Man Utd": "Man United", "Spurs": "Tottenham"}


def normalize_schedule(rows: list[dict]) -> list[dict]:
    fixtures = []
    for row in rows:
        kickoff = datetime.fromisoformat(row["DateUtc"].replace("Z", "+00:00"))
        home = TEAM_ALIASES.get(row["HomeTeam"], row["HomeTeam"])
        away = TEAM_ALIASES.get(row["AwayTeam"], row["AwayTeam"])
        scores = [row["HomeTeamScore"], row["AwayTeamScore"]]
        if (scores[0] is None) != (scores[1] is None):
            raise ValueError("A fixture has an incomplete score.")
        if any(score is not None and (type(score) is not int or score < 0) for score in scores):
            raise ValueError("Invalid fixture score.")
        fixtures.append({
            "id": f"{SEASON}-{int(row['MatchNumber'])}",
            "matchweek": int(row["RoundNumber"]),
            "kickoff": kickoff.isoformat(),
            "date": kickoff.date().isoformat(),
            "home_team": home,
            "away_team": away,
            "venue": row["Location"],
            "home_score": scores[0],
            "away_score": scores[1],
            "finished": scores[0] is not None,
        })
    validate_schedule(fixtures)
    return sorted(fixtures, key=lambda fixture: (fixture["kickoff"], fixture["id"]))


def validate_schedule(fixtures: list[dict]) -> None:
    """Reject partial downloads and broken double round-robin schedules."""
    pairs = {(f["home_team"], f["away_team"]) for f in fixtures}
    teams = Counter(team for f in fixtures for team in (f["home_team"], f["away_team"]))
    weeks = Counter(f["matchweek"] for f in fixtures)
    if (len(fixtures) != 380 or len({f["id"] for f in fixtures}) != 380
            or len(pairs) != 380 or len(teams) != 20
            or set(teams.values()) != {38} or set(weeks) != set(range(1, 39))
            or set(weeks.values()) != {10} or any(home == away for home, away in pairs)):
        raise ValueError("Expected all 380 unique fixtures for 20 clubs and 38 matchweeks.")


def load_schedule(path: str | Path) -> dict:
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    if snapshot["season"] != SEASON:
        raise ValueError("The fixture snapshot must cover 2026/27.")
    validate_schedule(snapshot["fixtures"])
    return snapshot


def refresh_schedule(path: str | Path) -> Path:
    request = Request(SOURCE_URL, headers={"User-Agent": "PremierLeagueCourseProject/1.0"})
    with urlopen(request, timeout=30) as response:
        fixtures = normalize_schedule(json.load(response))
    snapshot = {
        "season": SEASON,
        "source": SOURCE_URL,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "fixtures": fixtures,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
