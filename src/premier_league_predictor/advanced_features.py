"""Reusable history state for richer features and constant-size fixture inference."""

from collections import defaultdict, deque
import numpy as np
import pandas as pd
from premier_league_predictor.features import (
    FEATURE_COLUMNS,
    INITIAL_ELO,
    ELO_K,
    HOME_ADVANTAGE,
    _validate_matches,
)
from premier_league_predictor.context import CONTEXT_COLUMNS, context_features

STAT_COLUMNS = [
    f"{side}_{stat}_last5"
    for side in ("home", "away")
    for stat in ("shots", "shots_on_target", "shots_allowed", "shots_on_target_allowed")
]
EXTRA_COLUMNS = (
    STAT_COLUMNS
    + [f"{side}_matches_last{days}" for side in ("home", "away") for days in (7, 14)]
    + [f"{side}_points_last{window}" for side in ("home", "away") for window in (3, 10)]
)
CONTEXT_FEATURES = [
    f"{side}_{column}" for side in ("home", "away") for column in CONTEXT_COLUMNS
]
ADVANCED_COLUMNS = FEATURE_COLUMNS + EXTRA_COLUMNS


def mean(values):
    known = [v for v in values if pd.notna(v)]
    return float(np.mean(known)) if known else np.nan


def recent_ten():
    return deque(maxlen=10)


def recent_five():
    return deque(maxlen=5)


def initial_elo():
    return INITIAL_ELO


class HistoryState:
    def __init__(self):
        self.form = defaultdict(recent_ten)
        self.venue = defaultdict(recent_five)
        self.dates = defaultdict(list)
        self.meetings = defaultdict(int)
        self.ratings = defaultdict(initial_elo)

    def features(self, day, home, away):
        day = pd.Timestamp(day).normalize()
        result = {}
        for side, team, venue in [("home", home, "home"), ("away", away, "away")]:
            history = list(self.form[team])
            recent = history[-5:]
            result[f"{side}_points_last5"] = (
                sum(r[0] for r in recent) if recent else np.nan
            )
            result[f"{side}_goals_for_last5"] = mean(r[1] for r in recent)
            result[f"{side}_goals_against_last5"] = mean(r[2] for r in recent)
            venue_points = self.venue[(team, venue)]
            result[f"{side}_{venue}_points_last5"] = (
                sum(venue_points) if venue_points else np.nan
            )
            dates = self.dates[team]
            result[f"{side}_rest_days"] = (day - dates[-1]).days if dates else np.nan
            for days in (7, 14):
                result[f"{side}_matches_last{days}"] = sum(
                    day - pd.Timedelta(days=days) <= date < day for date in dates[-20:]
                )
            for window in (3, 10):
                result[f"{side}_points_last{window}"] = (
                    sum(r[0] for r in history[-window:]) if history else np.nan
                )
            for index, stat in enumerate(
                (
                    "shots",
                    "shots_on_target",
                    "shots_allowed",
                    "shots_on_target_allowed",
                ),
                start=3,
            ):
                result[f"{side}_{stat}_last5"] = mean(r[index] for r in recent)
        result["head_to_head_count"] = self.meetings[tuple(sorted((home, away)))]
        result["elo_difference"] = self.ratings[home] - self.ratings[away]
        return result

    def update_day(self, rows):
        deltas = defaultdict(float)
        for row in rows:
            home, away = row.home_team, row.away_team
            expected = 1 / (
                1
                + 10
                ** ((self.ratings[away] - self.ratings[home] - HOME_ADVANTAGE) / 400)
            )
            actual = 1 if row.target == "H" else 0.5 if row.target == "D" else 0
            delta = ELO_K * (actual - expected)
            deltas[home] += delta
            deltas[away] -= delta
            hp, ap = (
                (3, 0) if row.target == "H" else (1, 1) if row.target == "D" else (0, 3)
            )
            hs, ash = getattr(row, "home_shots", np.nan), getattr(
                row, "away_shots", np.nan
            )
            hst, ast = getattr(row, "home_shots_on_target", np.nan), getattr(
                row, "away_shots_on_target", np.nan
            )
            self.form[home].append(
                (hp, row.home_goals, row.away_goals, hs, hst, ash, ast)
            )
            self.form[away].append(
                (ap, row.away_goals, row.home_goals, ash, ast, hs, hst)
            )
            self.venue[(home, "home")].append(hp)
            self.venue[(away, "away")].append(ap)
            self.dates[home].append(row.date)
            self.dates[away].append(row.date)
            self.meetings[tuple(sorted((home, away)))] += 1
        for team, delta in deltas.items():
            self.ratings[team] += delta


def build_advanced_features(matches, context):
    state = HistoryState()
    records = []
    for day, group in _validate_matches(matches).groupby("date", sort=True):
        rows = list(group.itertuples(index=False))
        for row in rows:
            records.append(
                {
                    "date": day,
                    "season": row.season,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "target": row.target,
                    "home_goals": row.home_goals,
                    "away_goals": row.away_goals,
                    **state.features(day, row.home_team, row.away_team),
                    **context_features(context, day, row.home_team, row.away_team),
                }
            )
        state.update_day(rows)
    return pd.DataFrame(records), state
