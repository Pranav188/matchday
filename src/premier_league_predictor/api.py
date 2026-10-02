"""Small JSON API for the interactive Premier League forecast interface."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from premier_league_predictor.fixtures import load_schedule

RESULT_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}
RESULT_CODES = ("H", "D", "A")
RESULT_IDS = {label: result_id for result_id, label in enumerate(RESULT_CODES)}
CURRENT_TARGET_SEASON = "2026-27"
MAX_REQUEST_BYTES = 8_192


def _project_root() -> Path:
    return Path(
        os.environ.get("MATCHDAY_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()


class PredictorService:
    """Serve a versioned artifact; never train inside a request or at startup."""

    def __init__(
        self,
        data_dir,
        reports_dir,
        fixtures_path=None,
        model_root=None,
        history_path=None,
    ):
        from premier_league_predictor.deployment import load_deployment
        from premier_league_predictor.context import load_context
        from premier_league_predictor.history import ForecastStore

        root = Path(model_root or _project_root())
        import threading

        self.lock = threading.RLock()
        self.root = root
        self.schedule = load_schedule(
            fixtures_path or root / "data/fixtures/2026-27.json"
        )
        self.fixture_lookup = {
            fixture["id"]: fixture for fixture in self.schedule["fixtures"]
        }
        self.bundle = load_deployment(root)
        self.estimator = self.bundle["estimator"]
        self.state = self.bundle["state"]
        self.context = load_context(root / "data/context/fixtures.csv")
        self.store = ForecastStore(
            history_path or root / "data/history/forecasts.sqlite"
        )
        self.store.reconcile(self.schedule["fixtures"])
        self.evaluation = self.bundle["evaluation"]
        meta = self.bundle["metadata"]
        self.pointer_version = self.bundle["metadata"]["version"]
        self.context_mtime = (
            (root / "data/context/fixtures.csv").stat().st_mtime_ns
            if (root / "data/context/fixtures.csv").exists()
            else None
        )
        self.schedule_mtime = (
            (
                Path(fixtures_path)
                if fixtures_path
                else root / "data/fixtures/2026-27.json"
            )
            .stat()
            .st_mtime_ns
        )
        self.schedule_path = (
            Path(fixtures_path)
            if fixtures_path
            else root / "data/fixtures/2026-27.json"
        )
        self.metadata = {
            "status": "ready",
            "model": meta["kind"],
            "model_version": meta["version"],
            "target_season": CURRENT_TARGET_SEASON,
            "data_through": meta["data_through"],
            "history_matches": meta["history_matches"],
            "learned_context": meta["learned_context"],
            "evaluation": self.evaluation,
        }

    def refresh_if_changed(self):
        from premier_league_predictor.deployment import load_deployment
        from premier_league_predictor.context import load_context

        pointer = json.loads((self.root / "models/current.json").read_text())
        if pointer["version"] != self.pointer_version:
            bundle = load_deployment(self.root)
            self.bundle = bundle
            self.estimator = bundle["estimator"]
            self.state = bundle["state"]
            self.evaluation = bundle["evaluation"]
            meta = bundle["metadata"]
            self.metadata.update(
                model=meta["kind"],
                model_version=meta["version"],
                data_through=meta["data_through"],
                history_matches=meta["history_matches"],
                learned_context=meta["learned_context"],
                evaluation=self.evaluation,
            )
            self.pointer_version = meta["version"]
        path = self.root / "data/context/fixtures.csv"
        mtime = path.stat().st_mtime_ns if path.exists() else None
        if mtime != self.context_mtime:
            self.context = load_context(path)
            self.context_mtime = mtime
        mtime = self.schedule_path.stat().st_mtime_ns
        if mtime != self.schedule_mtime:
            schedule = load_schedule(self.schedule_path)
            self.schedule = schedule
            self.fixture_lookup = {f["id"]: f for f in schedule["fixtures"]}
            self.schedule_mtime = mtime
            self.store.reconcile(schedule["fixtures"])

    def standings(self):
        """Calculate a season projection without saving individual match forecasts."""
        from datetime import datetime, timezone
        from premier_league_predictor.context import context_features
        from premier_league_predictor.evaluation import classifier_probabilities
        from premier_league_predictor.standings import project_standings

        with self.lock:
            self.refresh_if_changed()
            now = datetime.now(timezone.utc)
            fixtures = self.schedule["fixtures"]
            remaining = [fixture for fixture in fixtures if not fixture["finished"]]
            columns = self.bundle["metadata"]["feature_columns"]
            records = []
            for fixture in remaining:
                if datetime.fromisoformat(fixture["kickoff"]) <= now:
                    raise ValueError(
                        "Fixtures are awaiting results. Try again after the next schedule update."
                    )
                if pd.Timestamp(fixture["date"]) <= pd.Timestamp(
                    self.metadata["data_through"]
                ):
                    raise ValueError(
                        "The fixture schedule needs an update before the final table can be projected."
                    )
                values = self.state.features(
                    fixture["date"], fixture["home_team"], fixture["away_team"]
                )
                values.update(
                    context_features(
                        self.context,
                        fixture["date"],
                        fixture["home_team"],
                        fixture["away_team"],
                        now.isoformat(),
                    )
                )
                records.append({column: values[column] for column in columns})
            probabilities = {}
            if records:
                chances = classifier_probabilities(
                    self.estimator, pd.DataFrame(records)
                )
                probabilities = {
                    fixture["id"]: row for fixture, row in zip(remaining, chances)
                }
            return {
                "season": self.schedule["season"],
                "rows": project_standings(fixtures, probabilities),
                "completed_matches": len(fixtures) - len(remaining),
                "remaining_matches": len(remaining),
                "model_version": self.metadata["model_version"],
                "data_through": self.metadata["data_through"],
                "generated_at": now.isoformat(),
            }

    def predict(self, payload):
        with self.lock:
            self.refresh_if_changed()
            return self._predict(payload)

    def _predict(self, payload):
        from datetime import datetime, timezone
        from premier_league_predictor.context import context_features
        from premier_league_predictor.deployment import score_distribution
        from premier_league_predictor.availability import availability_for
        from premier_league_predictor.explanations import explain_deployment

        fixture_id = payload.get("fixture_id")
        if not isinstance(fixture_id, str) or fixture_id not in self.fixture_lookup:
            raise ValueError("Choose a fixture from the 2026/27 schedule.")
        fixture = self.fixture_lookup[fixture_id]
        now = datetime.now(timezone.utc)
        kickoff = datetime.fromisoformat(fixture["kickoff"])
        if fixture["finished"]:
            raise ValueError("This match has already finished.")
        if kickoff <= now:
            raise ValueError(
                "This match has already started. Refresh the fixture schedule."
            )
        if pd.Timestamp(fixture["date"]) <= pd.Timestamp(self.metadata["data_through"]):
            raise ValueError(
                "A forecast must be after the model's last completed result date."
            )
        values = self.state.features(
            fixture["date"], fixture["home_team"], fixture["away_team"]
        )
        values.update(
            context_features(
                self.context,
                fixture["date"],
                fixture["home_team"],
                fixture["away_team"],
                now.isoformat(),
            )
        )
        columns = self.bundle["metadata"]["feature_columns"]
        features = pd.DataFrame([{column: values[column] for column in columns}])
        from premier_league_predictor.evaluation import classifier_probabilities

        probabilities = classifier_probabilities(self.estimator, features)[0]
        chosen = int(probabilities.argmax())
        goal_means = [
            float(self.bundle["goal_models"][side].predict(features)[0])
            for side in ("home", "away")
        ]
        observed = {
            key: float(value) if pd.notna(value) else None
            for key, value in values.items()
        }
        forecast = {
            "fixture_id": fixture_id,
            "date": fixture["date"],
            "kickoff": fixture["kickoff"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "predicted_code": RESULT_CODES[chosen],
            "predicted_label": RESULT_LABELS[RESULT_CODES[chosen]],
            "probabilities": [
                {
                    "code": code,
                    "label": RESULT_LABELS[code],
                    "probability": float(probabilities[index]),
                }
                for index, code in enumerate(RESULT_CODES)
            ],
            "model": self.metadata["model"],
            "model_version": self.metadata["model_version"],
            "data_through": self.metadata["data_through"],
            "explanation": explain_deployment(self.bundle, features, chosen),
            "evaluation": self.evaluation,
            "goals": score_distribution(*goal_means),
            "factors": observed,
            "learned_context": self.metadata["learned_context"],
            "availability": availability_for(
                self.root / "data/context/availability.json",
                fixture["home_team"],
                fixture["away_team"],
                kickoff,
                now,
            ),
        }
        forecast["predicted_at"] = datetime.now(timezone.utc).isoformat()
        return self.store.save(forecast, fixture, observed)


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
        with self.service.lock:
            self.service.refresh_if_changed()
        if urlparse(self.path).path == "/standings":
            try:
                self._send_json(200, self.service.standings())
            except ValueError as error:
                self._send_json(503, {"error": str(error)})
            return
        if urlparse(self.path).path == "/history":
            self._send_json(200, self.service.store.list())
            return
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
                self._send_json(
                    413, {"error": "The forecast request is empty or too large."}
                )
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
    service = PredictorService(
        data_dir, reports_dir, history_path=os.environ.get("PREDICTOR_HISTORY_PATH")
    )
    handler = type("BoundPredictorHandler", (PredictorHandler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    print(
        f"{service.metadata['model']} ready at http://{host}:{port}; "
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
