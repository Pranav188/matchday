"""Load, validate, and audit Football-Data.co.uk season files."""

from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_COLUMNS = {
    "date": "date",
    "hometeam": "home_team",
    "awayteam": "away_team",
    "fthg": "home_goals",
    "ftag": "away_goals",
    "ftr": "target",
}
CANONICAL_COLUMNS = [
    "date",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "target",
]
CLASS_ORDER = ("H", "D", "A")
SEASON_FILENAME_PATTERN = re.compile(r"^(\d{4})_E0$", flags=re.IGNORECASE)


def season_from_filename(path: str | Path) -> str:
    """Extract a season label from filenames such as ``2526_E0.csv``."""
    stem = Path(path).stem
    match = SEASON_FILENAME_PATTERN.fullmatch(stem)
    if not match:
        raise ValueError(f"Expected a season filename like 2526_E0.csv, got: {Path(path).name}")

    code = match.group(1)
    start_suffix = int(code[:2])
    start_year = 1900 + start_suffix if start_suffix >= 90 else 2000 + start_suffix
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _read_csv(path: Path) -> pd.DataFrame:
    usecols = lambda name: str(name).replace("\ufeff", "").strip().lower() in SOURCE_COLUMNS
    try:
        return pd.read_csv(path, encoding="utf-8-sig", usecols=usecols)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", usecols=usecols)


def canonicalize_season(frame: pd.DataFrame, source: str | Path) -> pd.DataFrame:
    """Select the result-only source fields and validate labels against full-time goals."""
    season = season_from_filename(source)
    original_columns = {
        str(column).replace("\ufeff", "").strip().lower(): column
        for column in frame.columns
    }
    missing = sorted(set(SOURCE_COLUMNS) - set(original_columns))
    if missing:
        raise ValueError(f"{Path(source).name} is missing required columns: {', '.join(missing)}")

    selected = pd.DataFrame(
        {canonical: frame[original_columns[source_name]] for source_name, canonical in SOURCE_COLUMNS.items()}
    )
    blank_rows = selected.isna().all(axis=1)
    ignored_blank_rows = int(blank_rows.sum())
    selected = selected.loc[~blank_rows].copy()
    selected["date"] = pd.to_datetime(
        selected["date"].astype("string").str.strip(),
        dayfirst=True,
        errors="coerce",
        format="mixed",
    )
    selected["home_team"] = selected["home_team"].astype("string").str.strip()
    selected["away_team"] = selected["away_team"].astype("string").str.strip()
    selected["home_goals"] = pd.to_numeric(selected["home_goals"], errors="coerce")
    selected["away_goals"] = pd.to_numeric(selected["away_goals"], errors="coerce")
    selected["target"] = selected["target"].astype("string").str.strip().str.upper()
    selected["target"] = selected["target"].replace("", pd.NA)
    selected["season"] = season

    unplayed = selected[["home_goals", "away_goals", "target"]].isna().all(axis=1)
    invalid_unplayed = unplayed & (
        selected["date"].isna()
        | selected["home_team"].isna()
        | selected["away_team"].isna()
        | selected["home_team"].eq("")
        | selected["away_team"].eq("")
    )
    if invalid_unplayed.any():
        raise ValueError(
            f"{Path(source).name} has {int(invalid_unplayed.sum())} scheduled fixtures "
            "with missing dates or team names"
        )
    ignored_unplayed_fixtures = int(unplayed.sum())
    selected = selected.loc[~unplayed].copy()

    invalid = (
        selected["date"].isna()
        | selected["home_team"].isna()
        | selected["away_team"].isna()
        | selected["home_team"].eq("")
        | selected["away_team"].eq("")
        | selected["home_goals"].isna()
        | selected["away_goals"].isna()
        | ~selected["target"].isin(CLASS_ORDER)
    )
    if invalid.any():
        raise ValueError(
            f"{Path(source).name} has {int(invalid.sum())} rows with missing or invalid required values"
        )

    expected_target = np.select(
        [selected["home_goals"] > selected["away_goals"], selected["home_goals"] == selected["away_goals"]],
        ["H", "D"],
        default="A",
    )
    inconsistent = selected["target"].to_numpy() != expected_target
    if inconsistent.any():
        raise ValueError(
            f"{Path(source).name} has {int(inconsistent.sum())} FTR values inconsistent with FTHG/FTAG"
        )

    selected["home_goals"] = selected["home_goals"].astype(int)
    selected["away_goals"] = selected["away_goals"].astype(int)
    canonical = selected[CANONICAL_COLUMNS].sort_values("date", kind="stable").reset_index(drop=True)
    canonical.attrs["ignored_blank_rows"] = ignored_blank_rows
    canonical.attrs["ignored_unplayed_fixtures"] = ignored_unplayed_fixtures
    return canonical


def load_matches(data_dir: str | Path) -> pd.DataFrame:
    """Load all downloaded ``*_E0.csv`` files in chronological order."""
    directory = Path(data_dir)
    paths = sorted(directory.glob("*_E0.csv"))
    if not paths:
        raise FileNotFoundError(f"No season files matching *_E0.csv found in {directory}")

    seasons = [canonicalize_season(_read_csv(path), path) for path in paths]
    ignored_blank_rows = sum(int(season.attrs.get("ignored_blank_rows", 0)) for season in seasons)
    ignored_unplayed_fixtures = sum(
        int(season.attrs.get("ignored_unplayed_fixtures", 0)) for season in seasons
    )
    matches = pd.concat(seasons, ignore_index=True).sort_values(
        ["date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)
    matches.attrs["ignored_blank_rows"] = ignored_blank_rows
    matches.attrs["ignored_unplayed_fixtures"] = ignored_unplayed_fixtures
    return matches


def audit_matches(matches: pd.DataFrame) -> dict[str, Any]:
    """Return a compact, JSON-serializable dataset audit."""
    counts = matches["target"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    season_counts = matches.groupby("season", sort=True).size()
    duplicate_rows = matches.duplicated(["date", "home_team", "away_team"], keep=False)
    return {
        "rows": int(len(matches)),
        "date_start": matches["date"].min().date().isoformat(),
        "date_end": matches["date"].max().date().isoformat(),
        "seasons": {str(season): int(count) for season, count in season_counts.items()},
        "duplicate_fixture_rows": int(duplicate_rows.sum()),
        "ignored_blank_rows": int(matches.attrs.get("ignored_blank_rows", 0)),
        "ignored_unplayed_fixtures": int(matches.attrs.get("ignored_unplayed_fixtures", 0)),
        "missing_required_values": 0,
        "target_counts": {label: int(counts[label]) for label in CLASS_ORDER},
        "target_proportions": {
            label: float(counts[label] / len(matches)) for label in CLASS_ORDER
        },
        "teams": int(pd.concat([matches["home_team"], matches["away_team"]]).nunique()),
        "source_columns_used": list(SOURCE_COLUMNS),
    }


def write_audit(audit: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return destination


def source_manifest(data_dir: str | Path) -> list[dict[str, Any]]:
    """Record source filenames, byte counts, and SHA-256 hashes for reproducibility."""
    directory = Path(data_dir)
    metadata_path = directory / ".download_manifest.json"
    retrieval_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    manifest = []
    for path in sorted(directory.glob("*_E0.csv")):
        try:
            headers = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
        except UnicodeDecodeError:
            headers = pd.read_csv(path, nrows=0, encoding="cp1252").columns
        actual_headers = {
            str(column).replace("\ufeff", "").strip().lower(): str(column).replace("\ufeff", "").strip()
            for column in headers
        }
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        retrieval = retrieval_metadata.get(path.name)
        if retrieval is None:
            retrieval = {
                "retrieved_at_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(timespec="seconds"),
                "timestamp_source": "inferred_from_file_mtime",
            }
        manifest.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                **retrieval,
                "required_headers": {
                    source_name: actual_headers[source_name] for source_name in SOURCE_COLUMNS
                },
            }
        )
    return manifest
