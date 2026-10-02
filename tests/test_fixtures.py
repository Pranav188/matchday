import copy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from premier_league_predictor.api import PredictorService
from premier_league_predictor.fixtures import (
    load_schedule,
    normalize_schedule,
    validate_schedule,
)

SNAPSHOT = Path(__file__).resolve().parents[1] / "data/fixtures/2026-27.json"


def test_snapshot_contains_the_complete_double_round_robin():
    fixtures = load_schedule(SNAPSHOT)["fixtures"]
    assert len(fixtures) == 380
    pairs = {(f["home_team"], f["away_team"]) for f in fixtures}
    assert all((away, home) in pairs for home, away in pairs)
    assert "Man Utd" not in {f["home_team"] for f in fixtures}
    assert "Spurs" not in {f["home_team"] for f in fixtures}


def test_incomplete_or_duplicate_schedule_is_rejected():
    fixtures = load_schedule(SNAPSHOT)["fixtures"]
    with pytest.raises(ValueError, match="380"):
        validate_schedule(fixtures[:-1])
    broken = copy.deepcopy(fixtures)
    broken[-1] = broken[0]
    with pytest.raises(ValueError, match="380"):
        validate_schedule(broken)


def test_partial_score_is_rejected():
    with pytest.raises(ValueError, match="incomplete score"):
        normalize_schedule(
            [
                {
                    "DateUtc": "2026-08-21 19:00:00Z",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Coventry",
                    "HomeTeamScore": 1,
                    "AwayTeamScore": None,
                }
            ]
        )


@pytest.fixture
def service(monkeypatch):
    service = PredictorService.__new__(PredictorService)
    service.fixture_lookup = {
        "scheduled": {
            "date": "2099-01-01",
            "kickoff": "2099-01-01T15:00:00+00:00",
            "home_team": "Arsenal",
            "away_team": "Man United",
            "finished": False,
        }
    }
    import threading

    service.lock = threading.RLock()
    service.refresh_if_changed = Mock()
    service.root = Path("/nonexistent")
    service.state = Mock()
    service.state.features.return_value = {"elo_difference": 100.0}
    from premier_league_predictor.context import load_context

    service.context = load_context(Path("/nonexistent/context.csv"))
    service.bundle = {
        "metadata": {"feature_columns": ["elo_difference"]},
        "goal_models": {"home": Mock(), "away": Mock()},
    }
    for model in service.bundle["goal_models"].values():
        model.predict.return_value = [1.0]
    service.estimator = Mock(classes_=np.array([0, 1, 2]))
    service.estimator.predict_proba.return_value = np.array([[0.5, 0.3, 0.2]])
    service.evaluation = None
    service.metadata = {
        "model": "tree",
        "model_version": "test",
        "data_through": "2026-09-20",
        "learned_context": [],
    }
    service.store = Mock()
    service.store.save.side_effect = lambda forecast, fixture, features: forecast
    monkeypatch.setattr(
        "premier_league_predictor.explanations.explain_deployment",
        Mock(return_value={"rules": [], "leaf_matches": 1}),
    )
    return service


def test_fixture_id_controls_teams_and_date(service):
    result = service.predict(
        {
            "fixture_id": "scheduled",
            "date": "1900-01-01",
            "home_team": "Chelsea",
            "away_team": "Liverpool",
        }
    )
    assert result["date"] == "2099-01-01"
    assert result["home_team"] == "Arsenal"
    assert result["away_team"] == "Man United"
    assert result["fixture_id"] == "scheduled"
    assert sum(
        item["probability"] for item in result["probabilities"]
    ) == pytest.approx(1)


@pytest.mark.parametrize("fixture_id", ["unknown", [], 4])
def test_unknown_fixture_is_rejected(service, fixture_id):
    with pytest.raises(ValueError, match="schedule"):
        service.predict({"fixture_id": fixture_id})


def test_completed_fixture_cannot_be_predicted(service):
    service.fixture_lookup["scheduled"]["finished"] = True
    with pytest.raises(ValueError, match="finished"):
        service.predict({"fixture_id": "scheduled"})


def test_started_fixture_cannot_be_predicted(service):
    service.fixture_lookup["scheduled"]["kickoff"] = "2020-01-01T15:00:00+00:00"
    with pytest.raises(ValueError, match="started"):
        service.predict({"fixture_id": "scheduled"})


def test_service_reloads_new_atomic_model_pointer(service, tmp_path, monkeypatch):
    import json

    service.root = tmp_path
    (tmp_path / "models").mkdir()
    (tmp_path / "models/current.json").write_text(json.dumps({"version": "v2"}))
    service.pointer_version = "v1"
    service.context_mtime = None
    service.schedule_path = SNAPSHOT
    service.schedule_mtime = SNAPSHOT.stat().st_mtime_ns
    metadata = {
        "version": "v2",
        "kind": "logistic",
        "data_through": "2026-09-20",
        "history_matches": 100,
        "learned_context": [],
    }
    bundle = {
        "metadata": metadata,
        "estimator": Mock(),
        "state": Mock(),
        "evaluation": {"accuracy": 0.5},
    }
    monkeypatch.setattr(
        "premier_league_predictor.deployment.load_deployment", lambda root: bundle
    )
    PredictorService.refresh_if_changed(service)
    assert service.estimator is bundle["estimator"]
    assert service.state is bundle["state"]
    assert service.metadata["model_version"] == service.pointer_version == "v2"


def test_standings_predicts_remaining_matches_in_one_batch_without_archiving(service):
    service.schedule = {
        "season": "2026-27",
        "fixtures": [
            {
                "id": "complete",
                "home_team": "Arsenal",
                "away_team": "Man United",
                "finished": True,
                "home_score": 2,
                "away_score": 0,
            },
            {"id": "scheduled", **service.fixture_lookup["scheduled"]},
        ],
    }
    result = service.standings()
    service.estimator.predict_proba.assert_called_once()
    service.store.save.assert_not_called()
    service.state.update_day.assert_not_called()
    assert result["completed_matches"] == result["remaining_matches"] == 1
    assert result["rows"][0]["current_points"] == 3
    assert result["rows"][0]["projected_points"] == pytest.approx(4.8)
    assert result["model_version"] == "test"


def test_standings_waits_for_missing_results_instead_of_predicting_started_games(
    service,
):
    fixture = {
        "id": "scheduled",
        **service.fixture_lookup["scheduled"],
        "kickoff": "2020-01-01T15:00:00+00:00",
    }
    service.schedule = {"season": "2026-27", "fixtures": [fixture]}
    with pytest.raises(ValueError, match="awaiting results"):
        service.standings()
    service.estimator.predict_proba.assert_not_called()
    service.store.save.assert_not_called()
