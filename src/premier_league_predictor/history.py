"""Immutable pre-kickoff predictions and result reconciliation in SQLite."""

from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import sqlite3
import numpy as np


class ForecastStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS forecasts (id INTEGER PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL, fixture_id TEXT NOT NULL, predicted_at TEXT NOT NULL, kickoff TEXT NOT NULL, model_version TEXT NOT NULL, payload TEXT NOT NULL, actual_code TEXT, home_score INTEGER, away_score INTEGER)"
            )

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, forecast, fixture, features):
        from datetime import datetime

        if datetime.fromisoformat(forecast["predicted_at"]) >= datetime.fromisoformat(
            fixture["kickoff"]
        ):
            raise ValueError(
                "Only predictions saved before kickoff can enter the archive"
            )
        encoded = json.dumps(features, sort_keys=True, allow_nan=False)
        fingerprint = hashlib.sha256(
            f"{fixture['id']}:{forecast['model_version']}:{encoded}".encode()
        ).hexdigest()
        with self.connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO forecasts(fingerprint,fixture_id,predicted_at,kickoff,model_version,payload) VALUES (?,?,?,?,?,?)",
                (
                    fingerprint,
                    fixture["id"],
                    forecast["predicted_at"],
                    fixture["kickoff"],
                    forecast["model_version"],
                    json.dumps({**forecast, "features": features}, allow_nan=False),
                ),
            )
            record = connection.execute(
                "SELECT id,payload FROM forecasts WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
        return {**json.loads(record[1]), "archive_id": record[0]}

    def reconcile(self, fixtures):
        with self.connection() as connection:
            for fixture in fixtures:
                if not fixture["finished"]:
                    continue
                home, away = fixture["home_score"], fixture["away_score"]
                result = "H" if home > away else "D" if home == away else "A"
                connection.execute(
                    "UPDATE forecasts SET actual_code=?,home_score=?,away_score=? WHERE fixture_id=?",
                    (result, home, away, fixture["id"]),
                )

    def list(self):
        with self.connection() as connection:
            # Score each fixture once, using its earliest saved forecast, even after retraining.
            records = connection.execute(
                "SELECT id,payload,actual_code,home_score,away_score FROM forecasts WHERE id IN (SELECT MIN(id) FROM forecasts GROUP BY fixture_id) ORDER BY predicted_at DESC"
            ).fetchall()
        items = [
            {
                **json.loads(row[1]),
                "archive_id": row[0],
                "actual_code": row[2],
                "home_score": row[3],
                "away_score": row[4],
            }
            for row in records
        ]
        settled = [row for row in items if row["actual_code"] is not None]
        metrics = None
        if settled:
            from premier_league_predictor.deployment import probability_metrics, LABELS

            probabilities = np.array(
                [[p["probability"] for p in row["probabilities"]] for row in settled]
            )
            metrics = probability_metrics(
                np.array([LABELS[row["actual_code"]] for row in settled]), probabilities
            )
        return {
            "forecasts": items[:200],
            "total": len(items),
            "settled": len(settled),
            "metrics": metrics,
            "policy": "Earliest pre-kickoff forecast per fixture",
        }
