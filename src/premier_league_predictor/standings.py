"""Expected final points from completed results and remaining fixture probabilities."""

import numpy as np


def project_standings(fixtures, probabilities):
    """Rank expected point totals; equal totals share a position."""
    clubs = sorted(
        {fixture[side] for fixture in fixtures for side in ("home_team", "away_team")}
    )
    table = {
        club: {
            "team": club,
            "played": 0,
            "remaining": 0,
            "current_points": 0,
            "projected_points": 0.0,
        }
        for club in clubs
    }
    for fixture in fixtures:
        home, away = table[fixture["home_team"]], table[fixture["away_team"]]
        if fixture["finished"]:
            hg, ag = fixture["home_score"], fixture["away_score"]
            if any(type(score) is not int or score < 0 for score in (hg, ag)):
                raise ValueError("Completed fixtures require two valid scores.")
            home_points, away_points = (
                (3, 0) if hg > ag else (1, 1) if hg == ag else (0, 3)
            )
            for row, points in ((home, home_points), (away, away_points)):
                row["played"] += 1
                row["current_points"] += points
                row["projected_points"] += points
        else:
            chances = np.asarray(probabilities[fixture["id"]], dtype=float)
            if (
                chances.shape != (3,)
                or not np.isfinite(chances).all()
                or (chances < 0).any()
                or (chances > 1).any()
                or not np.isclose(chances.sum(), 1)
            ):
                raise ValueError(
                    "Fixture probabilities must be a valid H/D/A distribution."
                )
            home["remaining"] += 1
            away["remaining"] += 1
            home["projected_points"] += float(3 * chances[0] + chances[1])
            away["projected_points"] += float(3 * chances[2] + chances[1])
    ordered = sorted(
        table.values(),
        key=lambda row: (-round(row["projected_points"], 10), row["team"]),
    )
    previous_points, previous_position = None, None
    for index, row in enumerate(ordered, start=1):
        points = round(row["projected_points"], 10)
        row["position"] = previous_position if points == previous_points else index
        previous_points, previous_position = points, row["position"]
    return ordered
