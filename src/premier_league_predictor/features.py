"""Build match-history features using only results available before each date."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "home_points_last5",
    "away_points_last5",
    "home_goals_for_last5",
    "away_goals_for_last5",
    "home_goals_against_last5",
    "away_goals_against_last5",
    "home_home_points_last5",
    "away_away_points_last5",
    "home_rest_days",
    "away_rest_days",
    "head_to_head_count",
    "elo_difference",
]
MATCH_COLUMNS = {
    "date",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "target",
}
INITIAL_ELO = 1500.0
ELO_K = 20.0
HOME_ADVANTAGE = 60.0


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(np.mean(items)) if items else float("nan")


def _points(target: str) -> tuple[int, int]:
    if target == "H":
        return 3, 0
    if target == "A":
        return 0, 3
    if target == "D":
        return 1, 1
    raise ValueError(f"Unknown match result: {target}")


def _validate_matches(matches: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(MATCH_COLUMNS - set(matches.columns))
    if missing:
        raise ValueError(f"Missing canonical match columns: {', '.join(missing)}")

    ordered = matches.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise").dt.normalize()
    ordered["target"] = ordered["target"].astype(str).str.upper()
    if not ordered["target"].isin({"H", "D", "A"}).all():
        raise ValueError("target must contain only H, D, or A")
    duplicate = ordered.duplicated(["date", "home_team", "away_team"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate fixtures must be resolved before feature generation")
    return ordered.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)


def build_pre_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Create one feature row per match, updating state only after each date batch."""
    ordered = _validate_matches(matches)
    all_history: dict[str, deque[tuple[int, int, int]]] = defaultdict(lambda: deque(maxlen=5))
    home_history: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=5))
    away_history: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=5))
    last_played: dict[str, pd.Timestamp] = {}
    head_to_head: dict[tuple[str, str], int] = defaultdict(int)
    ratings: dict[str, float] = defaultdict(lambda: INITIAL_ELO)
    feature_rows: list[dict[str, object]] = []

    for match_date, date_matches in ordered.groupby("date", sort=True):
        date_updates: list[tuple[pd.Series, int, int, float, float]] = []
        rating_deltas: dict[str, float] = defaultdict(float)

        for _, match in date_matches.iterrows():
            home = str(match["home_team"])
            away = str(match["away_team"])
            home_form = list(all_history[home])
            away_form = list(all_history[away])
            home_venue_form = list(home_history[home])
            away_venue_form = list(away_history[away])
            pair = tuple(sorted((home, away)))

            feature_rows.append(
                {
                    "date": match_date,
                    "season": match["season"],
                    "home_team": home,
                    "away_team": away,
                    "target": str(match["target"]),
                    "home_points_last5": sum(item[0] for item in home_form) if home_form else np.nan,
                    "away_points_last5": sum(item[0] for item in away_form) if away_form else np.nan,
                    "home_goals_for_last5": _mean(item[1] for item in home_form),
                    "away_goals_for_last5": _mean(item[1] for item in away_form),
                    "home_goals_against_last5": _mean(item[2] for item in home_form),
                    "away_goals_against_last5": _mean(item[2] for item in away_form),
                    "home_home_points_last5": sum(home_venue_form) if home_venue_form else np.nan,
                    "away_away_points_last5": sum(away_venue_form) if away_venue_form else np.nan,
                    "home_rest_days": (match_date - last_played[home]).days if home in last_played else np.nan,
                    "away_rest_days": (match_date - last_played[away]).days if away in last_played else np.nan,
                    "head_to_head_count": head_to_head[pair],
                    "elo_difference": ratings[home] - ratings[away],
                }
            )

            home_points, away_points = _points(str(match["target"]))
            home_rating = ratings[home]
            away_rating = ratings[away]
            expected_home = 1.0 / (
                1.0 + 10.0 ** ((away_rating - (home_rating + HOME_ADVANTAGE)) / 400.0)
            )
            actual_home = 1.0 if match["target"] == "H" else 0.5 if match["target"] == "D" else 0.0
            home_delta = ELO_K * (actual_home - expected_home)
            away_delta = ELO_K * ((1.0 - actual_home) - (1.0 - expected_home))
            date_updates.append((match, home_points, away_points, home_delta, away_delta))
            rating_deltas[home] += home_delta
            rating_deltas[away] += away_delta

        for match, home_points, away_points, _, _ in date_updates:
            home = str(match["home_team"])
            away = str(match["away_team"])
            home_goals = int(match["home_goals"])
            away_goals = int(match["away_goals"])
            all_history[home].append((home_points, home_goals, away_goals))
            all_history[away].append((away_points, away_goals, home_goals))
            home_history[home].append(home_points)
            away_history[away].append(away_points)
            last_played[home] = match_date
            last_played[away] = match_date
            head_to_head[tuple(sorted((home, away)))] += 1

        for team, delta in rating_deltas.items():
            ratings[team] += delta

    return pd.DataFrame.from_records(feature_rows)


def build_fixture_features(
    matches: pd.DataFrame,
    fixture_date: str | pd.Timestamp,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    """Build one upcoming fixture row from completed matches before its date."""
    if not home_team.strip() or not away_team.strip():
        raise ValueError("Home and away team names are required")
    if home_team.strip().casefold() == away_team.strip().casefold():
        raise ValueError("Home and away teams must be different")

    match_date = pd.to_datetime(fixture_date, errors="raise").normalize()
    historical = matches.copy()
    historical["date"] = pd.to_datetime(historical["date"], errors="raise").dt.normalize()
    historical = historical.loc[historical["date"] < match_date]
    start_year = match_date.year if match_date.month >= 7 else match_date.year - 1
    placeholder = pd.DataFrame.from_records(
        [
            {
                "date": match_date,
                "season": f"{start_year}-{(start_year + 1) % 100:02d}",
                "home_team": home_team.strip(),
                "away_team": away_team.strip(),
                "home_goals": 0,
                "away_goals": 0,
                "target": "D",
            }
        ]
    )
    features = build_pre_match_features(pd.concat([historical, placeholder], ignore_index=True))
    return features.iloc[[-1]][FEATURE_COLUMNS].reset_index(drop=True)
