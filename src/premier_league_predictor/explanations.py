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
    transformed = np.asarray(
        estimator.named_steps["imputer"].transform(features), dtype=np.float32
    )
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
        rules.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "value": float(transformed[0, index]),
                "threshold": float(tree.threshold[node]),
                "operator": "≤" if goes_left else ">",
                "imputed": bool(pd.isna(observed)),
            }
        )
        node = int(tree.children_left[node] if goes_left else tree.children_right[node])
    return {"rules": rules, "leaf_matches": int(tree.n_node_samples[leaf])}


def explain_deployment(bundle, features, predicted_class):
    """Identify local tree decisions or signed linear contributions accurately."""
    base = bundle["base_estimator"]
    transformed = base[:-1].transform(features)
    classifier = base.named_steps["classifier"]
    columns = list(features.columns)
    if hasattr(classifier, "tree_"):
        data = np.asarray(transformed, dtype=np.float32)
        tree = classifier.tree_
        leaf = int(classifier.apply(data)[0])
        visited = set(classifier.decision_path(data).indices)
        node = 0
        rules = []
        scaler = base.named_steps["scaler"]
        while node != leaf:
            index = int(tree.feature[node])
            goes_left = int(tree.children_left[node]) in visited
            # Translate standardized split thresholds back to original feature units.
            threshold = (
                tree.threshold[node] * scaler.scale_[index] + scaler.mean_[index]
            )
            observed = features.iloc[0, index]
            imputed = base.named_steps["imputer"].transform(features)[0, index]
            feature = columns[index]
            rules.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(
                        feature, feature.replace("_", " ").capitalize()
                    ),
                    "value": float(imputed),
                    "threshold": float(threshold),
                    "operator": "≤" if goes_left else ">",
                    "imputed": bool(pd.isna(observed)),
                }
            )
            node = int(
                tree.children_left[node] if goes_left else tree.children_right[node]
            )
        return {
            "method": "tree",
            "rules": rules,
            "leaf_matches": int(tree.n_node_samples[leaf]),
            "calibrated": bundle["metadata"]["kind"] == "calibrated_tree",
        }
    if hasattr(classifier, "coef_"):
        probabilities = base.predict_proba(features)[0]
        competitors = [
            i
            for i, class_id in enumerate(classifier.classes_)
            if class_id != predicted_class
        ]
        rival_index = max(competitors, key=lambda i: probabilities[i])
        chosen_index = list(classifier.classes_).index(predicted_class)
        contributions = transformed[0] * (
            classifier.coef_[chosen_index] - classifier.coef_[rival_index]
        )
        indices = np.argsort(np.abs(contributions))[-5:][::-1]
        return {
            "method": "linear",
            "rules": [],
            "leaf_matches": None,
            "rival": ["H", "D", "A"][int(classifier.classes_[rival_index])],
            "contributions": [
                {
                    "label": FEATURE_LABELS.get(
                        columns[i], columns[i].replace("_", " ").capitalize()
                    ),
                    "value": float(contributions[i]),
                }
                for i in indices
            ],
            "intercept": float(
                classifier.intercept_[chosen_index] - classifier.intercept_[rival_index]
            ),
        }
    importance = classifier.feature_importances_
    indices = np.argsort(importance)[-5:][::-1]
    return {
        "method": "global",
        "rules": [],
        "leaf_matches": None,
        "contributions": [
            {
                "label": FEATURE_LABELS.get(
                    columns[i], columns[i].replace("_", " ").capitalize()
                ),
                "value": float(importance[i]),
            }
            for i in indices
        ],
    }
