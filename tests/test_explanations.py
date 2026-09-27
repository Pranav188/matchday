import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from premier_league_predictor.explanations import explain_tree
from premier_league_predictor.features import FEATURE_COLUMNS


@pytest.mark.parametrize("value", [0.0, 9.0, np.nan])
def test_explanation_follows_the_fitted_tree_after_imputation(value):
    training = pd.DataFrame(0.0, index=range(6), columns=FEATURE_COLUMNS)
    training["elo_difference"] = [0, 1, 2, 7, 8, 9]
    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", DecisionTreeClassifier(max_depth=2, random_state=42)),
    ]).fit(training, [0, 0, 0, 2, 2, 2])
    fixture = training.iloc[[0]].copy()
    fixture["elo_difference"] = value
    explanation = explain_tree(estimator, fixture)
    assert len(explanation["rules"]) == 1
    rule = explanation["rules"][0]
    assert rule["feature"] == "elo_difference"
    assert rule["imputed"] == bool(np.isnan(value))
    assert rule["operator"] == ("≤" if rule["value"] <= rule["threshold"] else ">")
    assert explanation["leaf_matches"] == 3
    expected_class = 0 if rule["operator"] == "≤" else 2
    assert estimator.predict(fixture)[0] == expected_class


def test_tree_without_splits_has_an_empty_path():
    training = pd.DataFrame(0.0, index=range(3), columns=FEATURE_COLUMNS)
    estimator = Pipeline([
        ("imputer", SimpleImputer(keep_empty_features=True)),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ]).fit(training, [0, 0, 0])
    assert explain_tree(estimator, training.iloc[[0]]) == {"rules": [], "leaf_matches": 3}
