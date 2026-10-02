import copy

import numpy as np
import pytest

from premier_league_predictor.standings import project_standings


def fixture(match_id, home="Arsenal", away="Chelsea", scores=None):
    return {
        "id": match_id,
        "home_team": home,
        "away_team": away,
        "finished": scores is not None,
        "home_score": scores[0] if scores else None,
        "away_score": scores[1] if scores else None,
    }


def test_projection_combines_real_points_and_correct_home_away_probabilities():
    fixtures = [
        fixture("complete", scores=(3, 1)),
        fixture("future", "Chelsea", "Arsenal"),
    ]
    original = copy.deepcopy(fixtures)
    rows = project_standings(fixtures, {"future": [0.2, 0.3, 0.5]})
    arsenal, chelsea = rows
    assert arsenal["team"] == "Arsenal"
    assert arsenal["current_points"] == 3
    assert arsenal["projected_points"] == pytest.approx(4.8)
    assert chelsea["current_points"] == 0
    assert chelsea["projected_points"] == pytest.approx(0.9)
    assert all(row["played"] == row["remaining"] == 1 for row in rows)
    assert fixtures == original
    # A remaining match awards 3 - P(draw) total expected points.
    assert sum(row["projected_points"] for row in rows) == pytest.approx(3 + 3 - 0.3)


def test_completed_season_requires_no_future_probabilities():
    rows = project_standings(
        [fixture("one", scores=(1, 1)), fixture("two", "Chelsea", "Arsenal", (2, 0))],
        {},
    )
    assert rows[0]["team"] == "Chelsea"
    assert rows[0]["projected_points"] == rows[0]["current_points"] == 4
    assert all(row["remaining"] == 0 and row["played"] == 2 for row in rows)


def test_equal_expected_points_share_position_and_stable_club_order():
    rows = project_standings([fixture("one")], {"one": [0.3, 0.4, 0.3]})
    assert [row["position"] for row in rows] == [1, 1]
    assert [row["team"] for row in rows] == ["Arsenal", "Chelsea"]


@pytest.mark.parametrize(
    "chances",
    [[0.5, 0.5], [-0.1, 0.5, 0.6], [np.nan, 0.3, 0.7], [0, 0, 0], [1.1, 0, -0.1]],
)
def test_invalid_probabilities_cannot_create_a_table(chances):
    with pytest.raises(ValueError, match="distribution"):
        project_standings([fixture("one")], {"one": chances})


def test_incomplete_result_is_rejected():
    with pytest.raises(ValueError, match="scores"):
        project_standings([fixture("one", scores=(1, None))], {})
