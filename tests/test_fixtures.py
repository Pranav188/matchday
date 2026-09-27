import copy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from premier_league_predictor.api import PredictorService
from premier_league_predictor.fixtures import load_schedule, normalize_schedule, validate_schedule


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
        normalize_schedule([{
            "DateUtc": "2026-08-21 19:00:00Z", "HomeTeam": "Arsenal",
            "AwayTeam": "Coventry", "HomeTeamScore": 1, "AwayTeamScore": None,
        }])


@pytest.fixture
def service(monkeypatch):
    service = PredictorService.__new__(PredictorService)
    service.fixture_lookup = {"scheduled": {
        "date": "2099-01-01", "kickoff": "2099-01-01T15:00:00+00:00",
        "home_team": "Arsenal", "away_team": "Man United", "finished": False,
    }}
    service.matches = pd.DataFrame({"date": [pd.Timestamp("2026-09-20")]})
    service.team_lookup = {"arsenal": "Arsenal", "man united": "Man United"}
    service.estimator = Mock(classes_=np.array([0, 1, 2]))
    service.estimator.predict_proba.return_value = np.array([[.5, .3, .2]])
    service.evaluation = None
    monkeypatch.setattr("premier_league_predictor.api.explain_tree", Mock(return_value={"rules": [], "leaf_matches": 1}))
    service.metadata = {"model": "Decision Tree", "data_through": "2026-09-20", "latest_season": "2026-27"}
    monkeypatch.setattr("premier_league_predictor.api.build_fixture_features", Mock(return_value=pd.DataFrame()))
    return service


def test_fixture_id_controls_teams_and_date(service):
    result = service.predict({"fixture_id": "scheduled", "date": "1900-01-01", "home_team": "Chelsea", "away_team": "Liverpool"})
    assert result["date"] == "2099-01-01"
    assert result["home_team"] == "Arsenal"
    assert result["away_team"] == "Man United"
    assert result["fixture_id"] == "scheduled"
    assert sum(item["probability"] for item in result["probabilities"]) == pytest.approx(1)


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
