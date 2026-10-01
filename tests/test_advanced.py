import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from premier_league_predictor.advanced_features import build_advanced_features
from premier_league_predictor.features import build_pre_match_features, FEATURE_COLUMNS
from premier_league_predictor.context import load_context, context_features
from premier_league_predictor.deployment import (
    score_distribution,
    probability_metrics,
    restrict_window,
)
from premier_league_predictor.history import ForecastStore


def matches():
    return pd.DataFrame(
        [
            ("2024-08-01", "2024-25", "Arsenal", "Chelsea", 2, 0, "H", 12, 4, 5, 1),
            ("2024-08-08", "2024-25", "Arsenal", "Chelsea", 0, 1, "A", 8, 9, 2, 3),
            ("2024-08-15", "2024-25", "Chelsea", "Arsenal", 1, 1, "D", 10, 10, 3, 3),
        ],
        columns=[
            "date",
            "season",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "target",
            "home_shots",
            "away_shots",
            "home_shots_on_target",
            "away_shots_on_target",
        ],
    )


def test_extended_features_preserve_legacy_and_prior_only_statistics(tmp_path):
    data = matches()
    context = load_context(tmp_path / "missing.csv")
    advanced, state = build_advanced_features(data, context)
    legacy = build_pre_match_features(data)
    assert_frame_equal(
        advanced[FEATURE_COLUMNS], legacy[FEATURE_COLUMNS], check_dtype=False
    )
    assert advanced.loc[1, "home_shots_last5"] == 12
    assert advanced.loc[1, "home_matches_last7"] == 1
    changed = data.copy()
    changed.loc[1, "home_shots"] = 999
    updated, _ = build_advanced_features(changed, context)
    assert advanced.loc[1, "home_shots_last5"] == updated.loc[1, "home_shots_last5"]
    assert updated.loc[2, "away_shots_last5"] != advanced.loc[2, "away_shots_last5"]
    assert state.features("2024-08-22", "Arsenal", "Chelsea")["home_matches_last7"] == 1


def test_context_observed_after_prediction_is_excluded(tmp_path):
    path = tmp_path / "context.csv"
    pd.DataFrame(
        [
            {
                "fixture_date": "2024-08-08",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "team": "Arsenal",
                "observed_at": "2024-08-08T12:00:00Z",
                "missing_minutes_share": 0.3,
            }
        ]
    ).to_csv(path, index=False)
    frame = load_context(path)
    assert np.isnan(
        context_features(frame, "2024-08-08", "Arsenal", "Chelsea")[
            "home_missing_minutes_share"
        ]
    )
    assert (
        context_features(
            frame, "2024-08-08", "Arsenal", "Chelsea", "2024-08-08T13:00:00Z"
        )["home_missing_minutes_share"]
        == 0.3
    )


def test_score_distribution_is_normalized_and_home_away_orientation():
    result = score_distribution(3, 0.5)
    assert sum(result["outcome_probabilities"]) == pytest.approx(1)
    assert result["outcome_probabilities"][0] > result["outcome_probabilities"][2]
    assert len(result["scorelines"]) == 3


def test_archive_is_immutable_and_counts_fixture_once(tmp_path):
    store = ForecastStore(tmp_path / "history.sqlite")
    fixture = {"id": "one", "kickoff": "2026-08-01T15:00:00+00:00", "finished": False}
    forecast = {
        "fixture_id": "one",
        "predicted_at": "2026-08-01T12:00:00+00:00",
        "model_version": "v1",
        "probabilities": [
            {"probability": 0.6},
            {"probability": 0.2},
            {"probability": 0.2},
        ],
    }
    first = store.save(forecast, fixture, {"form": 3})
    duplicate = store.save(
        {**forecast, "predicted_at": "2026-08-01T13:00:00+00:00"}, fixture, {"form": 3}
    )
    assert first["predicted_at"] == duplicate["predicted_at"]
    store.save({**forecast, "model_version": "v2"}, fixture, {"form": 4})
    store.reconcile([{**fixture, "finished": True, "home_score": 2, "away_score": 0}])
    history = store.list()
    assert history["total"] == history["settled"] == 1
    assert history["metrics"]["accuracy"] == 1
    with pytest.raises(ValueError, match="before kickoff"):
        store.save({**forecast, "predicted_at": fixture["kickoff"]}, fixture, {})


def test_training_window_never_contains_validation_dates():
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2010-01-01", "2020-01-01", "2025-01-01"])}
    )
    selected = restrict_window(frame, pd.Timestamp("2025-01-01"), 5)
    assert selected.date.tolist() == [pd.Timestamp("2020-01-01")]


def test_statistics_loader_retains_optional_shots_without_using_current_match(tmp_path):
    from premier_league_predictor.data import canonicalize_season

    frame = pd.DataFrame(
        {
            "Date": ["1/8/24"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [0],
            "FTR": ["H"],
            "HS": [12],
            "HST": [5],
        }
    )
    result = canonicalize_season(frame, "2425_E0.csv", include_statistics=True)
    assert result.home_shots.iloc[0] == 12
    assert np.isnan(result.away_shots.iloc[0])


def test_context_rejects_naive_timestamps_and_invalid_shares(tmp_path):
    path = tmp_path / "context.csv"
    row = {
        "fixture_date": "2024-08-08",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "team": "Arsenal",
        "observed_at": "2024-08-07T12:00:00",
        "missing_minutes_share": 0.3,
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="timezone"):
        load_context(path)
    row.update(observed_at="2024-08-07T12:00:00Z", missing_minutes_share=1.5)
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="between"):
        load_context(path)


def test_linear_explanation_matches_log_odds_difference(tmp_path):
    from premier_league_predictor.deployment import pipeline
    from premier_league_predictor.explanations import explain_deployment

    columns = ["home_points_last5", "away_points_last5"]
    training = pd.DataFrame(
        [[0, 10], [1, 8], [5, 5], [6, 6], [8, 1], [10, 0]], columns=columns
    )
    model = pipeline("logistic").fit(training, [2, 2, 1, 1, 0, 0])
    features = training.iloc[[-1]]
    result = explain_deployment(
        {"base_estimator": model, "metadata": {"kind": "logistic"}}, features, 0
    )
    score = result["intercept"] + sum(item["value"] for item in result["contributions"])
    probabilities = model.predict_proba(features)[0]
    rival = {"H": 0, "D": 1, "A": 2}[result["rival"]]
    assert score == pytest.approx(np.log(probabilities[0] / probabilities[rival]))


def test_context_feature_selection_ignores_future_coverage():
    from premier_league_predictor.advanced_features import CONTEXT_FEATURES
    from premier_league_predictor.deployment import eligible_context_columns

    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=300)})
    for column in CONTEXT_FEATURES:
        frame[column] = np.nan
    frame["home_manager_days"] = np.arange(300)
    future = frame.copy()
    future["date"] = future.date + pd.DateOffset(years=5)
    future["home_missing_minutes_share"] = np.linspace(0, 1, 300)
    assert eligible_context_columns(pd.concat([frame, future])) == ["home_manager_days"]


def test_history_state_is_pickle_safe_and_same_day_uses_prior_snapshot(tmp_path):
    import joblib

    frame = matches()
    frame.loc[1, "date"] = frame.loc[0, "date"]
    frame.loc[1, "away_team"] = "Liverpool"
    features, state = build_advanced_features(
        frame, load_context(tmp_path / "absent.csv")
    )
    assert pd.isna(features.loc[1, "home_shots_last5"])
    changed = frame.copy()
    changed.loc[0, "home_shots"] = 999
    updated, _ = build_advanced_features(changed, load_context(tmp_path / "absent.csv"))
    assert_frame_equal(features.iloc[:2], updated.iloc[:2])
    joblib.dump(state, tmp_path / "state.joblib")
    restored = joblib.load(tmp_path / "state.joblib")
    assert restored.features("2024-08-22", "Arsenal", "Chelsea") == state.features(
        "2024-08-22", "Arsenal", "Chelsea"
    )


def test_optional_statistics_reject_non_numeric_counts():
    from premier_league_predictor.data import canonicalize_season

    frame = pd.DataFrame(
        {
            "Date": ["1/8/24"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [0],
            "FTR": ["H"],
            "HS": ["broken"],
        }
    )
    with pytest.raises(ValueError):
        canonicalize_season(frame, "2425_E0.csv", include_statistics=True)


def test_project_root_can_be_set_for_installed_container_package(monkeypatch, tmp_path):
    from premier_league_predictor.api import _project_root

    monkeypatch.setenv("MATCHDAY_ROOT", str(tmp_path))
    assert _project_root() == tmp_path


def test_calibration_is_fitted_after_its_base_training_period():
    from premier_league_predictor.deployment import fit_candidate

    rows = 700
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=rows),
            "target": np.resize(["H", "D", "A"], rows),
            "input": np.arange(rows),
        }
    )
    base, calibrated, cutoff = fit_candidate(frame, ["input"], "calibrated_tree")
    assert cutoff == "2023-07-01"
    assert base.named_steps["classifier"].tree_.n_node_samples[0] == sum(
        frame.date < cutoff
    )
    assert calibrated.predict_proba(frame[["input"]].iloc[[-1]]).sum() == pytest.approx(
        1
    )


def test_availability_expires_and_is_not_used_for_far_future_fixtures(tmp_path):
    from datetime import timedelta
    from premier_league_predictor.availability import availability_for

    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    path = tmp_path / "availability.json"
    path.write_text(
        json.dumps(
            {
                "observed_at": now.isoformat(),
                "players": [
                    {"team": "Arsenal", "status": "i", "player": "Test player"}
                ],
            }
        )
    )
    assert (
        availability_for(path, "Arsenal", "Chelsea", now + timedelta(days=2), now)[
            "status"
        ]
        == "context_only"
    )
    assert (
        availability_for(path, "Arsenal", "Chelsea", now + timedelta(days=8), now)[
            "players"
        ]
        == []
    )
    assert (
        availability_for(
            path, "Arsenal", "Chelsea", now + timedelta(days=4), now + timedelta(days=3)
        )["players"]
        == []
    )


def test_model_load_rejects_pointer_version_mismatch(tmp_path):
    import joblib
    from premier_league_predictor.deployment import load_deployment

    models = tmp_path / "models"
    models.mkdir()
    joblib.dump({"metadata": {"version": "v1"}}, models / "saved.joblib")
    (models / "current.json").write_text(
        json.dumps({"file": "saved.joblib", "version": "v2"})
    )
    with pytest.raises(ValueError, match="version"):
        load_deployment(tmp_path)


def test_context_handles_mixed_timestamp_precision_and_timezone(tmp_path):
    path = tmp_path / "context.csv"
    row = {
        "fixture_date": "2024-08-08",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "team": "Arsenal",
        "observed_at": "2024-08-07T12:00:00Z",
        "missing_minutes_share": 0.2,
    }
    pd.DataFrame(
        [
            row,
            {
                **row,
                "observed_at": "2024-08-07T14:00:00.123456+01:00",
                "missing_minutes_share": 0.4,
            },
        ]
    ).to_csv(path, index=False)
    loaded = load_context(path)
    assert (
        context_features(loaded, "2024-08-08", "Arsenal", "Chelsea")[
            "home_missing_minutes_share"
        ]
        == 0.4
    )


def test_context_cli_validates_and_atomically_imports(tmp_path, monkeypatch):
    import sys
    from premier_league_predictor.cli import main

    source = tmp_path / "supplied.csv"
    pd.DataFrame(
        [
            {
                "fixture_date": "2024-08-08",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "team": "Arsenal",
                "observed_at": "2024-08-07T12:00:00Z",
                "manager_days": 30,
            }
        ]
    ).to_csv(source, index=False)
    monkeypatch.setattr("premier_league_predictor.api._project_root", lambda: tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["pl-predictor", "context", "--context-file", str(source)]
    )
    main()
    imported = load_context(tmp_path / "data/context/fixtures.csv")
    assert imported.manager_days.iloc[0] == 30
    assert not (tmp_path / "data/context/fixtures.tmp").exists()


@pytest.mark.parametrize(
    "extra",
    [{"missing_attackers": 1.5}, {"all_matches_last7": 3, "all_matches_last14": 2}],
)
def test_context_rejects_invalid_counts_and_workload(tmp_path, extra):
    path = tmp_path / "context.csv"
    row = {
        "fixture_date": "2024-08-08",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "team": "Arsenal",
        "observed_at": "2024-08-07T12:00:00Z",
        **extra,
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_context(path)


def test_concurrent_forecasts_are_saved_only_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = ForecastStore(tmp_path / "history.sqlite")
    fixture = {"id": "one", "kickoff": "2026-08-01T15:00:00+00:00", "finished": False}
    forecast = {
        "fixture_id": "one",
        "predicted_at": "2026-08-01T12:00:00+00:00",
        "model_version": "v1",
        "probabilities": [
            {"probability": 0.6},
            {"probability": 0.2},
            {"probability": 0.2},
        ],
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(
            pool.map(lambda _: store.save(forecast, fixture, {"form": 3}), range(12))
        )
    assert len({record["archive_id"] for record in records}) == 1
    assert store.list()["total"] == 1
