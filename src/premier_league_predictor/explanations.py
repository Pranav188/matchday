"""Expose the exact fitted-tree route taken by one fixture."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from premier_league_predictor.features import FEATURE_COLUMNS


FEATURE_LABELS = {
    "home_points_last5": "Home points · last 5 matches",
    "away_points_last5": "Away points · last 5 matches",
    "home_goals_for_last5": "Home goals scored · last 5 average",
    "away_goals_for_last5": "Away goals scored · last 5 average",
    "home_goals_against_last5": "Home goals conceded · last 5 average",
    "away_goals_against_last5": "Away goals conceded · last 5 average",
    "home_home_points_last5": "Home points · last 5 home matches",
    "away_away_points_last5": "Away points · last 5 away matches",
    "home_rest_days": "Home rest days",
    "away_rest_days": "Away rest days",
    "head_to_head_count": "Previous meetings",
    "elo_difference": "Elo difference · home minus away",
}


def explain_tree(estimator: Pipeline, features: pd.DataFrame) -> dict:
    # Use the same preprocessing and float32 conversion as tree inference.
    transformed = np.asarray(estimator.named_steps["imputer"].transform(features), dtype=np.float32)
    classifier = estimator.named_steps["classifier"]
    tree = classifier.tree_
    visited = set(classifier.decision_path(transformed).indices)
    leaf = int(classifier.apply(transformed)[0])
    rules = []
    node = 0
    while node != leaf:
        index = int(tree.feature[node])
        feature = FEATURE_COLUMNS[index]
        goes_left = int(tree.children_left[node]) in visited
        observed = features.iloc[0][feature]
        rules.append({
            "feature": feature,
            "label": FEATURE_LABELS[feature],
            "value": float(transformed[0, index]),
            "threshold": float(tree.threshold[node]),
            "operator": "≤" if goes_left else ">",
            "imputed": bool(pd.isna(observed)),
        })
        node = int(tree.children_left[node] if goes_left else tree.children_right[node])
    return {"rules": rules, "leaf_matches": int(tree.n_node_samples[leaf])}
