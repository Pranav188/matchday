import pandas as pd
import pytest

from premier_league_predictor.data import canonicalize_season, season_from_filename, source_manifest


def test_season_filename_is_mapped_to_a_human_readable_label():
    assert season_from_filename("2526_E0.csv") == "2025-26"
    assert season_from_filename("9394_E0.csv") == "1993-94"


def test_canonicalizer_keeps_only_result_contract_columns():
    source = pd.DataFrame(
        {
            "Date": ["15/08/25"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [0],
            "FTR": ["H"],
            "HS": [14],
            "B365H": [1.8],
        }
    )

    result = canonicalize_season(source, "2526_E0.csv")

    assert result.columns.tolist() == [
        "date",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "target",
    ]
    assert result.loc[0, "target"] == "H"
    assert result.loc[0, "season"] == "2025-26"


def test_canonicalizer_rejects_a_result_that_disagrees_with_the_score():
    source = pd.DataFrame(
        {
            "Date": ["15/08/25"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [0],
            "FTAG": [0],
            "FTR": ["H"],
        }
    )

    with pytest.raises(ValueError, match="inconsistent"):
        canonicalize_season(source, "2526_E0.csv")


def test_canonicalizer_ignores_and_counts_trailing_blank_rows():
    source = pd.DataFrame(
        {
            "Date": ["15/08/25", None],
            "HomeTeam": ["Arsenal", None],
            "AwayTeam": ["Chelsea", None],
            "FTHG": [2, None],
            "FTAG": [0, None],
            "FTR": ["H", None],
        }
    )

    result = canonicalize_season(source, "2526_E0.csv")

    assert len(result) == 1
    assert result.attrs["ignored_blank_rows"] == 1


def test_source_manifest_records_actual_required_headers(tmp_path):
    source = tmp_path / "2526_E0.csv"
    pd.DataFrame(
        {
            "Date": ["15/08/25"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [0],
            "FTR": ["H"],
        }
    ).to_csv(source, index=False)

    manifest = source_manifest(tmp_path)

    assert manifest[0]["required_headers"] == {
        "date": "Date",
        "hometeam": "HomeTeam",
        "awayteam": "AwayTeam",
        "fthg": "FTHG",
        "ftag": "FTAG",
        "ftr": "FTR",
    }
    assert len(manifest[0]["sha256"]) == 64
    assert manifest[0]["retrieved_at_utc"]
    assert manifest[0]["timestamp_source"] == "inferred_from_file_mtime"
