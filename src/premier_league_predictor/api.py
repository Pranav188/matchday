"""Small JSON API for the interactive Premier League forecast interface."""

from __future__ import annotations

import json
import os
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from premier_league_predictor.data import audit_matches, load_matches
from premier_league_predictor.features import FEATURE_COLUMNS, build_fixture_features, build_pre_match_features
from premier_league_predictor.fixtures import load_schedule
from premier_league_predictor.explanations import explain_tree


RESULT_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}
RESULT_CODES = ("H", "D", "A")
RESULT_IDS = {label: result_id for result_id, label in enumerate(RESULT_CODES)}
CURRENT_TARGET_SEASON = "2026-27"
MAX_REQUEST_BYTES = 8_192


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_evaluation(reports_dir: Path) -> dict[str, Any] | None:
    path = reports_dir / "final_test_metrics.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    baselines = report.get("baselines", {})
    baseline = baselines.get("majority_class", {})
    return {
        "selected_model": report.get("selected_model"),
        "season": report.get("final_test_season", "2025-26"),
        "matches": report.get("final_test_matches", 0),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "baseline_accuracy": baseline.get("accuracy"),
        "baseline_macro_f1": baseline.get("macro_f1"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "class_order": metrics.get("class_order", list(RESULT_CODES)),
    }


def _decision_tree_parameters(reports_dir: Path) -> dict[str, Any]:
    """Reuse the tuned tree settings recorded by the validation run."""
    defaults: dict[str, Any] = {
        "max_depth": 3,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
    }
    path = reports_dir / "best_parameters.json"
    if not path.exists():
        return defaults

    saved = json.loads(path.read_text(encoding="utf-8")).get("decision_tree", {})
    allowed = {"max_depth", "min_samples_leaf", "class_weight"}
    for name in allowed:
        key = f"classifier__{name}"
        if key in saved:
            defaults[name] = saved[key]
    return defaults


class PredictorService:
    """Train the deployment estimator on available results and forecast fixtures."""

    def __init__(self, data_dir: str | Path, reports_dir: str | Path, fixtures_path: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.schedule = load_schedule(fixtures_path or _project_root() / "data/fixtures/2026-27.json")
        self.fixture_lookup = {fixture["id"]: fixture for fixture in self.schedule["fixtures"]}
        self.matches = load_matches(self.data_dir)
        duplicate_rows = self.matches.duplicated(
            ["date", "home_team", "away_team"], keep=False
        )
        if duplicate_rows.any():
            raise ValueError("Duplicate fixtures exist in the local data; resolve them before serving forecasts")

        self.evaluation = _read_evaluation(Path(reports_dir))
        if self.evaluation and self.evaluation["selected_model"] != "decision_tree":
            raise ValueError(
                "The current evaluation selected a different model; update the prediction service before serving forecasts"
            )

        feature_table = build_pre_match_features(self.matches)
        target = feature_table["target"].map(RESULT_IDS).to_numpy(dtype=int)
        classifier = DecisionTreeClassifier(
            criterion="gini",
            random_state=42,
            **_decision_tree_parameters(Path(reports_dir)),
        )
        self.estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("classifier", classifier),
            ]
        )
        self.estimator.fit(feature_table[FEATURE_COLUMNS], target)

        audit = audit_matches(self.matches)
        most_recent_season = str(
            self.matches.loc[self.matches["date"].idxmax(), "season"]
        )
        latest_teams = self.matches.loc[
            self.matches["season"] == most_recent_season, ["home_team", "away_team"]
        ]
        self.teams = sorted(
            set(latest_teams["home_team"].astype(str))
            | set(latest_teams["away_team"].astype(str)),
            key=str.casefold,
        )
        self.team_lookup = {team.casefold(): team for team in self.teams}
        self.metadata = {
            "status": "ready",
            "model": "Decision Tree",
            "target_season": CURRENT_TARGET_SEASON,
            "latest_season": most_recent_season,
            "current_season_loaded": most_recent_season == CURRENT_TARGET_SEASON,
            "history_matches": audit["rows"],
            "history_start": audit["date_start"],
            "data_through": audit["date_end"],
            "training_seasons": len(audit["seasons"]),
            "teams": self.teams,
            "evaluation": self.evaluation,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        fixture_id = payload.get("fixture_id")
        if fixture_id is not None:
            if not isinstance(fixture_id, str) or fixture_id not in self.fixture_lookup:
                raise ValueError("Choose a fixture from the 2026/27 schedule.")
            fixture = self.fixture_lookup[fixture_id]
            if fixture["finished"]:
                raise ValueError("This match has already finished.")
            if pd.Timestamp(fixture["kickoff"]) <= pd.Timestamp.now(tz="UTC"):
                raise ValueError("This match has already started. Refresh the fixture schedule.")
            payload = {"home_team": fixture["home_team"], "away_team": fixture["away_team"], "date": fixture["date"]}
        home_team = payload.get("home_team")
        away_team = payload.get("away_team")
        fixture_date = payload.get("date")
        if not isinstance(home_team, str) or not isinstance(away_team, str):
            raise ValueError("Choose both a home team and an away team.")
        home_team = home_team.strip()
        away_team = away_team.strip()
        if not home_team or not away_team:
            raise ValueError("Choose both a home team and an away team.")
        if len(home_team) > 80 or len(away_team) > 80:
            raise ValueError("Team names must be 80 characters or fewer.")
        if home_team.casefold() == away_team.casefold():
            raise ValueError("Choose two different teams.")
        if not isinstance(fixture_date, str):
            raise ValueError("Choose the fixture date.")
        try:
            parsed_date = pd.Timestamp(date.fromisoformat(fixture_date))
        except ValueError as error:
            raise ValueError("Enter a valid fixture date.") from error

        latest_result_date = pd.to_datetime(self.matches["date"]).max().normalize()
        if parsed_date.normalize() <= latest_result_date:
            raise ValueError(
                f"Choose a date after the latest result in this dataset ({latest_result_date.date().isoformat()})."
            )

        canonical_home = self.team_lookup.get(home_team.casefold(), home_team)
        canonical_away = self.team_lookup.get(away_team.casefold(), away_team)
        features = build_fixture_features(
            self.matches, parsed_date, canonical_home, canonical_away
        )
        raw_probabilities = self.estimator.predict_proba(features)[0]
        probabilities = np.zeros(len(RESULT_CODES), dtype=float)
        for source_index, class_id in enumerate(self.estimator.classes_):
            probabilities[int(class_id)] = raw_probabilities[source_index]
        predicted_index = int(probabilities.argmax())
        outcomes = [
            {
                "code": code,
                "label": RESULT_LABELS[code],
                "probability": float(probabilities[index]),
            }
            for index, code in enumerate(RESULT_CODES)
        ]
        return {
            "fixture_id": fixture_id,
            "date": parsed_date.date().isoformat(),
            "home_team": canonical_home,
            "away_team": canonical_away,
            "predicted_code": RESULT_CODES[predicted_index],
            "predicted_label": RESULT_LABELS[RESULT_CODES[predicted_index]],
            "probabilities": outcomes,
            "model": self.metadata["model"],
            "data_through": self.metadata["data_through"],
            "latest_season": self.metadata["latest_season"],
            "explanation": explain_tree(self.estimator, features),
            "evaluation": self.evaluation,
        }


class PredictorHandler(BaseHTTPRequestHandler):
    """Expose health metadata and predictions as small JSON responses."""

    service: PredictorService

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/fixtures":
            self._send_json(200, self.service.schedule)
            return
        if urlparse(self.path).path == "/health":
            self._send_json(200, self.service.metadata)
            return
        self._send_json(404, {"error": "Route not found."})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/predict":
            self._send_json(404, {"error": "Route not found."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                self._send_json(413, {"error": "The forecast request is empty or too large."})
                return
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("The forecast request must be a JSON object.")
            self._send_json(200, self.service.predict(payload))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Send a valid JSON forecast request."})
        except ValueError as error:
            self._send_json(422, {"error": str(error)})

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format_string % args}")


def main() -> None:
    root = _project_root()
    data_dir = Path(os.environ.get("PREDICTOR_DATA_DIR", root / "data" / "raw"))
    reports_dir = Path(os.environ.get("PREDICTOR_REPORTS_DIR", root / "reports"))
    host = os.environ.get("PREDICTOR_HOST", "127.0.0.1")
    port = int(os.environ.get("PREDICTOR_PORT", "8000"))
    service = PredictorService(data_dir, reports_dir)
    handler = type("BoundPredictorHandler", (PredictorHandler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    print(
        f"Decision Tree ready at http://{host}:{port}; "
        f"history through {service.metadata['data_through']} "
        f"({service.metadata['history_matches']:,} completed matches)."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping predictor API.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
