"""Time-aware model selection and one-time final-season evaluation."""

from __future__ import annotations

import json
from math import prod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from premier_league_predictor.constants import MODEL_DISPLAY_NAMES
from premier_league_predictor.evaluation import (
    CLASS_IDS,
    CLASS_NAMES,
    classification_metrics,
    classifier_probabilities,
    save_confusion_matrix,
    save_roc_curves,
)
from premier_league_predictor.features import FEATURE_COLUMNS


FINAL_TEST_SEASON = "2025-26"
LABEL_TO_ID = {label: class_id for class_id, label in enumerate(CLASS_NAMES)}
RANDOM_SEED = 42


def _pipeline(classifier: Any, scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def _model_specs() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    return {
        "logistic_regression": (
            _pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED), scale=True),
            {"classifier__C": [0.1, 1.0, 10.0], "classifier__class_weight": [None, "balanced"]},
        ),
        "decision_tree": (
            _pipeline(DecisionTreeClassifier(criterion="gini", random_state=RANDOM_SEED), scale=False),
            {
                "classifier__max_depth": [3, 8],
                "classifier__min_samples_leaf": [10, 25],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            _pipeline(RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1), scale=False),
            {
                "classifier__n_estimators": [200],
                "classifier__max_depth": [8, None],
                "classifier__min_samples_leaf": [5, 15],
                "classifier__class_weight": [None, "balanced_subsample"],
            },
        ),
        "xgboost": (
            _pipeline(
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                ),
                scale=False,
            ),
            {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
            },
        ),
        "rbf_svm": (
            _pipeline(
                CalibratedClassifierCV(
                    SVC(kernel="rbf", random_state=RANDOM_SEED),
                    method="sigmoid",
                    cv=3,
                    ensemble=False,
                ),
                scale=True,
            ),
            {
                "classifier__estimator__C": [0.5, 2.0, 8.0],
                "classifier__estimator__gamma": ["scale", 0.03],
                "classifier__estimator__class_weight": [None, "balanced"],
            },
        ),
    }


def _macro_f1_scorer(estimator: Any, features: pd.DataFrame, target: np.ndarray) -> float:
    return float(f1_score(target, estimator.predict(features), labels=CLASS_IDS, average="macro", zero_division=0))


def _negative_log_loss_scorer(estimator: Any, features: pd.DataFrame, target: np.ndarray) -> float:
    return -float(log_loss(target, estimator.predict_proba(features), labels=CLASS_IDS))


def _select_best_index(cv_results: dict[str, Any]) -> int:
    macro_f1 = np.asarray(cv_results["mean_test_macro_f1"], dtype=float)
    negative_log_loss = np.asarray(cv_results["mean_test_neg_log_loss"], dtype=float)
    best_f1 = np.nanmax(macro_f1)
    candidates = np.flatnonzero(np.isclose(macro_f1, best_f1, equal_nan=False))
    return int(candidates[np.nanargmax(negative_log_loss[candidates])])


def _date_blocked_splits(dates: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window folds without splitting matches from one date."""
    normalized = pd.to_datetime(dates, errors="raise").dt.normalize().to_numpy()
    unique_dates = np.unique(normalized)
    date_splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_date_indices, validation_date_indices in date_splitter.split(unique_dates):
        train_dates = unique_dates[train_date_indices]
        validation_dates = unique_dates[validation_date_indices]
        train_indices = np.flatnonzero(np.isin(normalized, train_dates))
        validation_indices = np.flatnonzero(np.isin(normalized, validation_dates))
        folds.append((train_indices, validation_indices))
    return folds


def _elo_ablation(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Compare the selected pipeline with and without Elo on identical folds."""
    feature_sets = {
        "all_features": FEATURE_COLUMNS,
        "without_elo": [column for column in FEATURE_COLUMNS if column != "elo_difference"],
    }
    rows = []
    for feature_set, columns in feature_sets.items():
        fold_f1 = []
        fold_log_loss = []
        for train_indices, validation_indices in folds:
            fold_estimator = clone(estimator)
            fold_estimator.fit(X_train.iloc[train_indices][columns], y_train[train_indices])
            validation_features = X_train.iloc[validation_indices][columns]
            predictions = fold_estimator.predict(validation_features).astype(int)
            probabilities = classifier_probabilities(fold_estimator, validation_features)
            fold_f1.append(
                f1_score(
                    y_train[validation_indices],
                    predictions,
                    labels=CLASS_IDS,
                    average="macro",
                    zero_division=0,
                )
            )
            fold_log_loss.append(
                log_loss(y_train[validation_indices], probabilities, labels=CLASS_IDS)
            )
        rows.append(
            {
                "feature_set": feature_set,
                "mean_validation_macro_f1": float(np.mean(fold_f1)),
                "mean_validation_log_loss": float(np.mean(fold_log_loss)),
            }
        )
    return pd.DataFrame(rows)


def _validation_diagnostics(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    reports_path: Path,
) -> dict[str, Any]:
    """Save pooled fold predictions and a descriptive validation confusion matrix."""
    actual_parts = []
    prediction_parts = []
    probability_parts = []
    fold_scores = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        fold_estimator = clone(estimator)
        fold_estimator.fit(X_train.iloc[train_indices], y_train[train_indices])
        validation_features = X_train.iloc[validation_indices]
        actual = y_train[validation_indices]
        predictions = fold_estimator.predict(validation_features).astype(int)
        probabilities = classifier_probabilities(fold_estimator, validation_features)
        fold_metrics = classification_metrics(actual, predictions, probabilities)
        fold_scores.append(
            {
                "fold": fold_number,
                "validation_macro_f1": fold_metrics["macro_f1"],
                "validation_log_loss": fold_metrics["log_loss"],
            }
        )
        actual_parts.append(actual)
        prediction_parts.append(predictions)
        probability_parts.append(probabilities)

    actual = np.concatenate(actual_parts)
    predictions = np.concatenate(prediction_parts)
    probabilities = np.concatenate(probability_parts)
    metrics = classification_metrics(actual, predictions, probabilities)
    save_confusion_matrix(
        metrics,
        reports_path / f"validation_{name}_confusion_matrix.png",
        f"Expanding-window validation — {MODEL_DISPLAY_NAMES[name]}",
    )
    return {
        "model": name,
        "pooled_validation_macro_f1": metrics["macro_f1"],
        "pooled_validation_log_loss": metrics["log_loss"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_order": metrics["class_order"],
        "fold_scores": fold_scores,
    }


def _baseline_results(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, dict[str, Any]]:
    majority_class = int(np.bincount(y_train, minlength=len(CLASS_IDS)).argmax())
    predictions = {
        "majority_class": np.full(len(y_test), majority_class, dtype=int),
        "always_home": np.zeros(len(y_test), dtype=int),
    }
    return {
        name: classification_metrics(y_test, predicted)
        for name, predicted in predictions.items()
    }


def run_model_comparison(
    features: pd.DataFrame,
    reports_dir: str | Path,
    models_dir: str | Path,
    n_splits: int = 5,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Tune on all seasons before 2025/26, select by validation, then test once."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if FINAL_TEST_SEASON not in set(features["season"]):
        raise ValueError(f"Final test season {FINAL_TEST_SEASON} is missing from the feature table")

    test = (
        features.loc[features["season"] == FINAL_TEST_SEASON]
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    test_start = test["date"].min()
    train = (
        features.loc[features["date"] < test_start]
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]
    y_train = train["target"].map(LABEL_TO_ID).to_numpy(dtype=int)
    y_test = test["target"].map(LABEL_TO_ID).to_numpy(dtype=int)
    if len(test) != 380:
        raise ValueError(
            f"Expected 380 matches in the complete 2025/26 final season, found {len(test)}"
        )
    if set(np.unique(y_train)) != set(CLASS_IDS) or set(np.unique(y_test)) != set(CLASS_IDS):
        raise ValueError("Training and final test data must contain all three result classes")

    split_count = min(n_splits, max(2, len(train) // 1000))
    folds = _date_blocked_splits(train["date"], split_count)
    fold_windows = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        train_dates = train.loc[train_indices, "date"]
        validation_dates = train.loc[validation_indices, "date"]
        fold_windows.append(
            {
                "fold": fold_number,
                "training_start": train_dates.min().date().isoformat(),
                "training_through": train_dates.max().date().isoformat(),
                "validation_start": validation_dates.min().date().isoformat(),
                "validation_through": validation_dates.max().date().isoformat(),
                "training_matches": int(len(train_indices)),
                "validation_matches": int(len(validation_indices)),
            }
        )
    searches: dict[str, GridSearchCV] = {}
    validation_rows: list[dict[str, Any]] = []

    for name, (pipeline, parameter_grid) in _model_specs().items():
        candidate_count = prod(len(values) for values in parameter_grid.values())
        print(
            f"Tuning {name}: {candidate_count} parameter sets across "
            f"{split_count} date-blocked folds"
        )
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring={"macro_f1": _macro_f1_scorer, "neg_log_loss": _negative_log_loss_scorer},
            refit=_select_best_index,
            cv=folds,
            n_jobs=n_jobs,
            error_score="raise",
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        result = search.cv_results_
        index = int(search.best_index_)
        validation_rows.append(
            {
                "model": name,
                "training_macro_f1": float(result["mean_train_macro_f1"][index]),
                "validation_macro_f1": float(result["mean_test_macro_f1"][index]),
                "validation_log_loss": float(-result["mean_test_neg_log_loss"][index]),
                "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
            }
        )
        searches[name] = search

    validation = pd.DataFrame(validation_rows).sort_values(
        ["validation_macro_f1", "validation_log_loss"], ascending=[False, True]
    ).reset_index(drop=True)
    selected_name = str(validation.iloc[0]["model"])
    selected_estimator = searches[selected_name].best_estimator_
    print(f"Running validation Elo ablation for {selected_name}")
    elo_ablation = _elo_ablation(selected_estimator, X_train, y_train, folds)

    predicted = selected_estimator.predict(X_test).astype(int)
    probabilities = classifier_probabilities(selected_estimator, X_test)
    selected_metrics = classification_metrics(y_test, predicted, probabilities)
    baselines = _baseline_results(y_train, y_test)

    reports_path = Path(reports_dir)
    models_path = Path(models_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    validation_diagnostics = []
    for name, search in searches.items():
        validation_diagnostics.append(
            _validation_diagnostics(
                name,
                search.best_estimator_,
                X_train,
                y_train,
                folds,
                reports_path,
            )
        )
    validation_diagnostics_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value in diagnostic.items()
                if key not in {"confusion_matrix", "class_order", "fold_scores"}
            }
            for diagnostic in validation_diagnostics
        ]
    )
    validation_diagnostics_frame.to_csv(reports_path / "validation_diagnostics.csv", index=False)
    validation_fold_scores = pd.DataFrame(
        [
            {**fold_windows[score["fold"] - 1], "model": diagnostic["model"], **score}
            for diagnostic in validation_diagnostics
            for score in diagnostic["fold_scores"]
        ]
    )
    validation_fold_scores.to_csv(reports_path / "validation_fold_scores.csv", index=False)
    (reports_path / "validation_confusion_matrices.json").write_text(
        json.dumps(
            {
                diagnostic["model"]: {
                    "class_order": diagnostic["class_order"],
                    "matrix": diagnostic["confusion_matrix"],
                }
                for diagnostic in validation_diagnostics
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    tree_importance_rows = []
    for name in ("decision_tree", "random_forest", "xgboost"):
        classifier = searches[name].best_estimator_.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            tree_importance_rows.extend(
                {
                    "model": name,
                    "feature": feature,
                    "importance": float(importance),
                }
                for feature, importance in zip(FEATURE_COLUMNS, classifier.feature_importances_)
            )
    pd.DataFrame(tree_importance_rows).sort_values(
        ["model", "importance"], ascending=[True, False]
    ).to_csv(reports_path / "tree_feature_importance.csv", index=False)

    logistic_classifier = searches["logistic_regression"].best_estimator_.named_steps["classifier"]
    logistic_coefficients = pd.DataFrame(
        logistic_classifier.coef_,
        columns=FEATURE_COLUMNS,
        index=[CLASS_NAMES[int(class_id)] for class_id in logistic_classifier.classes_],
    )
    logistic_coefficients.index.name = "result_class"
    logistic_coefficients.to_csv(reports_path / "logistic_regression_coefficients.csv")

    validation.to_csv(reports_path / "validation_model_comparison.csv", index=False)
    elo_ablation.to_csv(reports_path / "validation_elo_ablation.csv", index=False)
    (reports_path / "best_parameters.json").write_text(
        json.dumps({name: search.best_params_ for name, search in searches.items()}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    joblib.dump(selected_estimator, models_path / f"{selected_name}.joblib")
    save_confusion_matrix(
        selected_metrics,
        reports_path / "final_test_confusion_matrix.png",
        f"2025/26 final test — {MODEL_DISPLAY_NAMES[selected_name]}",
    )
    save_roc_curves(y_test, probabilities, reports_path / "final_test_roc.png")

    tree_estimator = searches["decision_tree"].best_estimator_.named_steps["classifier"]
    tree_rules = export_text(tree_estimator, feature_names=FEATURE_COLUMNS)
    (reports_path / "decision_tree_rules.txt").write_text(tree_rules + "\n", encoding="utf-8")

    result = {
        "selected_model": selected_name,
        "validation_comparison": validation,
        "validation_elo_ablation": elo_ablation,
        "validation_diagnostics": validation_diagnostics_frame,
        "validation_fold_scores": validation_fold_scores,
        "final_test": selected_metrics,
        "baselines": baselines,
        "final_test_matches": int(len(test)),
        "training_matches": int(len(train)),
        "n_splits": int(split_count),
    }
    (reports_path / "final_test_metrics.json").write_text(
        json.dumps(
            {
                "selected_model": selected_name,
                "metrics": selected_metrics,
                "baselines": baselines,
                "validation_elo_ablation": elo_ablation.to_dict(orient="records"),
                "training_matches": int(len(train)),
                "final_test_matches": int(len(test)),
                "final_test_season": FINAL_TEST_SEASON,
                "time_series_splits": int(split_count),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
