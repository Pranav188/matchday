import pandas as pd
from pandas.testing import assert_series_equal

from premier_league_predictor.features import FEATURE_COLUMNS, build_pre_match_features


def _matches(rows):
    return pd.DataFrame(
        rows,
        columns=["date", "season", "home_team", "away_team", "home_goals", "away_goals", "target"],
    )


def test_a_fixture_result_does_not_change_its_own_features():
    matches = _matches(
        [
            ("2025-08-10", "2025-26", "Arsenal", "Chelsea", 2, 0, "H"),
            ("2025-08-17", "2025-26", "Arsenal", "Leeds", 1, 0, "H"),
        ]
    )
    changed_result = matches.copy()
    changed_result.loc[1, ["home_goals", "away_goals", "target"]] = [0, 2, "A"]

    original_features = build_pre_match_features(matches).loc[1, FEATURE_COLUMNS]
    changed_features = build_pre_match_features(changed_result).loc[1, FEATURE_COLUMNS]

    assert_series_equal(original_features, changed_features)


def test_matches_on_the_same_date_use_one_shared_history_snapshot():
    matches = _matches(
        [
            ("2025-08-10", "2025-26", "Arsenal", "Chelsea", 2, 0, "H"),
            ("2025-08-10", "2025-26", "Arsenal", "Leeds", 0, 1, "A"),
        ]
    )
    changed_result = matches.copy()
    changed_result.loc[0, ["home_goals", "away_goals", "target"]] = [0, 2, "A"]

    original_features = build_pre_match_features(matches).loc[1, FEATURE_COLUMNS]
    changed_features = build_pre_match_features(changed_result).loc[1, FEATURE_COLUMNS]

    assert_series_equal(original_features, changed_features)
    assert pd.isna(original_features["home_points_last5"])


def test_only_earlier_dates_update_form_and_rest_features():
    matches = _matches(
        [
            ("2025-08-10", "2025-26", "Arsenal", "Chelsea", 2, 0, "H"),
            ("2025-08-11", "2025-26", "Arsenal", "Leeds", 1, 1, "D"),
        ]
    )

    features = build_pre_match_features(matches)

    assert features.loc[1, "home_points_last5"] == 3
    assert features.loc[1, "home_goals_for_last5"] == 2
    assert features.loc[1, "home_goals_against_last5"] == 0
    assert features.loc[1, "home_rest_days"] == 1
    assert features.loc[1, "head_to_head_count"] == 0
