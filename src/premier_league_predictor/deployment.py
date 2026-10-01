"""Chronological probability experiments and immutable deployment artifacts."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import poisson
from premier_league_predictor.advanced_features import (
    build_advanced_features,
    ADVANCED_COLUMNS,
    CONTEXT_FEATURES,
)
from premier_league_predictor.context import load_context
from premier_league_predictor.data import load_matches, source_manifest
from premier_league_predictor.evaluation import (
    classification_metrics,
    classifier_probabilities,
)

LABELS = {"H": 0, "D": 1, "A": 2}


def probability_metrics(actual, probabilities):
    metrics = classification_metrics(
        actual, probabilities.argmax(axis=1), probabilities
    )
    metrics["brier_score"] = float(
        np.mean(np.sum((probabilities - np.eye(3)[actual]) ** 2, axis=1))
    )
    return metrics


def pipeline(kind):
    classifier = (
        DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=25, random_state=42, class_weight="balanced"
        )
        if kind == "tree"
        else (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=15,
                random_state=42,
                n_jobs=2,
            )
            if kind == "forest"
            else LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        )
    )
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", keep_empty_features=True, add_indicator=False
                ),
            ),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def fit_candidate(training, columns, kind):
    estimator = pipeline("tree" if kind == "calibrated_tree" else kind)
    if kind == "calibrated_tree":
        latest = training.date.max()
        calibration_start = pd.Timestamp(
            year=latest.year if latest.month >= 7 else latest.year - 1, month=7, day=1
        )
        base = training[training.date < calibration_start]
        calibration = training[training.date >= calibration_start]
        if len(base) < 200 or len(calibration) < 100:
            raise ValueError(
                "Calibration requires at least 200 earlier and 100 later matches"
            )
        estimator.fit(base[columns], base.target.map(LABELS))
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(estimator), method="sigmoid"
        )
        calibrated.fit(calibration[columns], calibration.target.map(LABELS))
        return estimator, calibrated, calibration_start.date().isoformat()
    estimator.fit(training[columns], training.target.map(LABELS))
    return estimator, estimator, None


def restrict_window(frame, cutoff, years):
    frame = frame[frame.date < cutoff]
    return frame[frame.date >= cutoff - pd.DateOffset(years=years)] if years else frame


def score_distribution(home_mean, away_mean):
    # Extend the grid far enough to retain virtually all Poisson mass.
    limit = max(
        10, int(max(poisson.ppf(0.999999, home_mean), poisson.ppf(0.999999, away_mean)))
    )
    grid = np.outer(
        poisson.pmf(np.arange(limit + 1), home_mean),
        poisson.pmf(np.arange(limit + 1), away_mean),
    )
    grid /= grid.sum()
    top = np.argsort(grid.ravel())[-3:][::-1]
    return {
        "expected_home_goals": float(home_mean),
        "expected_away_goals": float(away_mean),
        "scorelines": [
            {
                "home": int(i // (limit + 1)),
                "away": int(i % (limit + 1)),
                "probability": float(grid.ravel()[i]),
            }
            for i in top
        ],
        "outcome_probabilities": [
            float(np.tril(grid, -1).sum()),
            float(np.trace(grid)),
            float(np.triu(grid, 1).sum()),
        ],
    }


def eligible_context_columns(frame):
    """Decide input availability using history before the first validation fold."""
    initial = frame[frame.date < "2022-07-01"]
    return [
        column
        for column in CONTEXT_FEATURES
        if initial[column].notna().sum() >= 300 and initial[column].nunique() > 1
    ]


def train_deployment(root, data_dir=None, context_path=None):
    root = Path(root)
    data_dir = Path(data_dir or root / "data/raw")
    context_path = Path(context_path or root / "data/context/fixtures.csv")
    matches = load_matches(data_dir, include_statistics=True)
    if matches.duplicated(["date", "home_team", "away_team"]).any():
        raise ValueError("Duplicate results must be resolved before training")
    context = load_context(context_path)
    frame, state = build_advanced_features(matches, context)
    # Feature availability is determined before any validation or benchmark labels.
    initial = frame[frame.date < "2022-07-01"]
    if len(initial) < 500:
        raise ValueError("Training needs at least 500 results before July 2022")
    learned_context = eligible_context_columns(frame)
    columns = ADVANCED_COLUMNS + learned_context
    development = frame[frame.date < "2025-07-01"]
    experiments = []
    for years in (5, 10, None):
        for kind in ("tree", "calibrated_tree", "logistic", "forest"):
            fold_metrics = []
            for season in (2022, 2023, 2024):
                start = pd.Timestamp(f"{season}-07-01")
                end = pd.Timestamp(f"{season + 1}-07-01")
                training = restrict_window(development, start, years)
                validation = development[
                    (development.date >= start) & (development.date < end)
                ]
                if len(validation) < 100:
                    raise ValueError("Incomplete recent validation seasons")
                _, model, _ = fit_candidate(training, columns, kind)
                probabilities = classifier_probabilities(model, validation[columns])
                fold_metrics.append(
                    {
                        "season": f"{season}-{str(season + 1)[-2:]}",
                        "training_through": training.date.max().date().isoformat(),
                        **probability_metrics(
                            validation.target.map(LABELS).to_numpy(), probabilities
                        ),
                    }
                )
            row = {
                "kind": kind,
                "window_years": years,
                "log_loss": float(np.mean([m["log_loss"] for m in fold_metrics])),
                "brier_score": float(np.mean([m["brier_score"] for m in fold_metrics])),
                "macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
                "folds": fold_metrics,
            }
            experiments.append(row)
            print(
                f"{kind}, window={years or 'all'}: log loss={row['log_loss']:.4f}",
                flush=True,
            )
    selected = min(experiments, key=lambda row: (row["log_loss"], row["brier_score"]))
    benchmark = frame[frame.season == "2025-26"]
    before = restrict_window(
        frame, pd.Timestamp("2025-07-01"), selected["window_years"]
    )
    base, model, _ = fit_candidate(before, columns, selected["kind"])
    probabilities = classifier_probabilities(model, benchmark[columns])
    actual = benchmark.target.map(LABELS).to_numpy()
    priors = np.bincount(before.target.map(LABELS), minlength=3) / len(before)
    prior_probabilities = np.tile(priors, (len(benchmark), 1))
    reliability = {}
    for index, label in enumerate(LABELS):
        observed, predicted = calibration_curve(
            (actual == index).astype(int),
            probabilities[:, index],
            n_bins=6,
            strategy="uniform",
        )
        reliability[label] = {
            "predicted": predicted.tolist(),
            "observed": observed.tolist(),
        }
    evaluation = {
        "season": "2025-26",
        "matches": len(benchmark),
        **probability_metrics(actual, probabilities),
        "baseline_accuracy": float(np.mean(actual == int(priors.argmax()))),
        "baseline_macro_f1": probability_metrics(actual, prior_probabilities)[
            "macro_f1"
        ],
        "baseline": probability_metrics(actual, prior_probabilities),
        "reliability": reliability,
        "protocol": "Retrospective benchmark; original 2025/26 results were already inspected. Future saved forecasts are the prospective evaluation.",
    }
    deployment_training = restrict_window(
        frame, frame.date.max() + pd.Timedelta(days=1), selected["window_years"]
    )
    base, model, calibration_start = fit_candidate(
        deployment_training, columns, selected["kind"]
    )
    goal_models = {}
    for side in ("home", "away"):
        goal_models[side] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("regressor", PoissonRegressor(alpha=1, max_iter=1000)),
            ]
        ).fit(deployment_training[columns], deployment_training[f"{side}_goals"])
    # Score model has a separate, explicitly retrospective benchmark.
    goal_metrics = {}
    for side in ("home", "away"):
        retrospective = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("regressor", PoissonRegressor(alpha=1, max_iter=1000)),
            ]
        ).fit(before[columns], before[f"{side}_goals"])
        goal_metrics[f"{side}_mae"] = float(
            np.mean(
                np.abs(
                    retrospective.predict(benchmark[columns])
                    - benchmark[f"{side}_goals"]
                )
            )
        )
    manifest = source_manifest(data_dir)
    signature = json.dumps(
        {
            "sources": [(s["file"], s["sha256"]) for s in manifest],
            "context": (
                hashlib.sha256(context_path.read_bytes()).hexdigest()
                if context_path.exists()
                else None
            ),
            "selected": [selected["kind"], selected["window_years"]],
            "columns": columns,
            "schema": 2,
            "code": {
                name: hashlib.sha256(
                    (root / "src/premier_league_predictor" / name).read_bytes()
                ).hexdigest()
                for name in (
                    "advanced_features.py",
                    "context.py",
                    "data.py",
                    "deployment.py",
                )
            },
        },
        sort_keys=True,
    )
    version = hashlib.sha256(signature.encode()).hexdigest()[:16]
    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_through": matches.date.max().date().isoformat(),
        "training_matches": len(deployment_training),
        "history_matches": len(matches),
        "kind": selected["kind"],
        "window_years": selected["window_years"],
        "calibration_start": calibration_start,
        "feature_columns": columns,
        "learned_context": learned_context,
        "goal_metrics": goal_metrics,
        "source_files": manifest,
    }
    artifact = {
        "metadata": metadata,
        "estimator": model,
        "base_estimator": base,
        "goal_models": goal_models,
        "state": state,
        "evaluation": evaluation,
    }
    models = root / "models"
    models.mkdir(exist_ok=True)
    destination = models / f"deployment-{version}.joblib"
    if not destination.exists():
        joblib.dump(artifact, destination.with_suffix(".tmp"))
        destination.with_suffix(".tmp").replace(destination)
    else:
        metadata = joblib.load(destination)["metadata"]
    pointer = models / "current.json"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"file": destination.name, "version": version}, indent=2) + "\n"
    )
    temporary.replace(pointer)
    report = {
        "selected": {k: v for k, v in selected.items() if k != "folds"},
        "experiments": experiments,
        "evaluation": evaluation,
        "deployment": metadata,
    }
    (root / "reports/advanced_evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    # Export a standalone reliability figure without chart dependencies in the UI.
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect reliability")
    for label, points in reliability.items():
        axis.plot(points["predicted"], points["observed"], marker="o", label=label)
    axis.set(
        xlabel="Predicted probability",
        ylabel="Observed frequency",
        title="2025/26 retrospective reliability",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "reports/probability_reliability.png", dpi=160)
    plt.close(figure)
    return metadata


def load_deployment(root):
    root = Path(root)
    pointer = root / "models/current.json"
    if not pointer.exists():
        raise FileNotFoundError(
            "No deployment model. Run: python -m premier_league_predictor train"
        )
    selected = json.loads(pointer.read_text())
    name = selected["file"]
    if Path(name).name != name:
        raise ValueError("Invalid model artifact name")
    artifact = joblib.load(root / "models" / name)
    if artifact["metadata"]["version"] != selected["version"]:
        raise ValueError("Model version mismatch")
    return artifact
