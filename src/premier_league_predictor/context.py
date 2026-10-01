"""Timestamped, provider-independent pre-match context with explicit missingness."""

from pathlib import Path
import pandas as pd
import numpy as np

CONTEXT_COLUMNS = [
    "missing_minutes_share",
    "missing_attackers",
    "missing_defenders",
    "missing_goalkeepers",
    "suspended_starters",
    "lineup_minutes_share",
    "manager_days",
    "all_matches_last7",
    "all_matches_last14",
]
KEYS = ["fixture_date", "home_team", "away_team", "team", "observed_at"]


def load_context(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=KEYS + CONTEXT_COLUMNS)
    frame = pd.read_csv(path)
    missing = set(KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"Context CSV is missing {sorted(missing)}")
    frame["fixture_date"] = pd.to_datetime(
        frame["fixture_date"], format="%Y-%m-%d", errors="raise"
    ).dt.normalize()
    from datetime import datetime

    if any(
        datetime.fromisoformat(str(value).replace("Z", "+00:00")).tzinfo is None
        for value in frame["observed_at"]
    ):
        raise ValueError("observed_at must include a timezone")
    frame["observed_at"] = pd.to_datetime(
        frame["observed_at"], utc=True, errors="raise"
    )
    if frame[KEYS].isna().any().any():
        raise ValueError("Context keys and timestamps cannot be missing")
    for column in CONTEXT_COLUMNS:
        values = (
            pd.to_numeric(frame[column], errors="raise")
            if column in frame
            else pd.Series(np.nan, index=frame.index)
        )
        if not np.isfinite(values.dropna()).all() or (values.dropna() < 0).any():
            raise ValueError(f"Invalid context values in {column}")
        if column.endswith("share") and (values.dropna() > 1).any():
            raise ValueError(f"{column} must be between 0 and 1")
        if not column.endswith("share") and (values.dropna() % 1 != 0).any():
            raise ValueError(f"{column} must contain integer counts")
        frame[column] = values
    if (frame.all_matches_last7 > frame.all_matches_last14).any():
        raise ValueError("7-day workload cannot exceed 14-day workload")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate context snapshots")
    if not ((frame.team == frame.home_team) | (frame.team == frame.away_team)).all():
        raise ValueError("Context team must belong to the fixture")
    return frame.sort_values("observed_at")


def context_features(frame, fixture_date, home, away, cutoff=None):
    # Date-only historical results use a conservative midnight cutoff.
    day = pd.Timestamp(fixture_date).normalize()
    cutoff = (
        pd.Timestamp(cutoff or day).tz_localize("UTC")
        if pd.Timestamp(cutoff or day).tzinfo is None
        else pd.Timestamp(cutoff or day).tz_convert("UTC")
    )
    eligible = (
        frame[
            (frame.fixture_date == day)
            & (frame.home_team == home)
            & (frame.away_team == away)
            & (frame.observed_at < cutoff)
        ]
        if len(frame)
        else frame
    )
    values = {}
    for side, team in [("home", home), ("away", away)]:
        records = eligible[eligible.team == team]
        row = records.iloc[-1] if len(records) else None
        for column in CONTEXT_COLUMNS:
            values[f"{side}_{column}"] = (
                float(row[column]) if row is not None else np.nan
            )
    return values
