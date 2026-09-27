import pandas as pd

from premier_league_predictor.modeling import _date_blocked_splits


def test_date_blocked_splits_keep_each_calendar_date_in_one_fold():
    dates = pd.Series(
        pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-05",
            ]
        )
    )

    for train_indices, validation_indices in _date_blocked_splits(dates, n_splits=2):
        train_dates = set(dates.iloc[train_indices])
        validation_dates = set(dates.iloc[validation_indices])
        assert train_dates.isdisjoint(validation_dates)
        assert dates.iloc[train_indices].max() < dates.iloc[validation_indices].min()
