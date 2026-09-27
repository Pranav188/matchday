"""Classification metrics and plots for the three match-result classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from premier_league_predictor.constants import MODEL_DISPLAY_NAMES

CLASS_NAMES = ("H", "D", "A")
CLASS_IDS = np.arange(len(CLASS_NAMES))


def classifier_probabilities(estimator: Any, features: Any) -> np.ndarray:
    """Return probability columns in the stable H, D, A class order."""
    raw_probabilities = estimator.predict_proba(features)
    probabilities = np.zeros((len(features), len(CLASS_NAMES)), dtype=float)
    for source_index, class_id in enumerate(estimator.classes_):
        probabilities[:, int(class_id)] = raw_probabilities[:, source_index]
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sums).all() or (row_sums <= 0).any():
        raise ValueError("Classifier returned probabilities with an invalid row sum")
    return probabilities / row_sums


def classification_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    matrix = confusion_matrix(actual, predicted, labels=CLASS_IDS)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(actual, predicted, labels=CLASS_IDS, average="macro", zero_division=0)
        ),
        "cohen_kappa": float(cohen_kappa_score(actual, predicted)),
        "precision_by_class": {
            name: float(value)
            for name, value in zip(
                CLASS_NAMES,
                precision_score(actual, predicted, labels=CLASS_IDS, average=None, zero_division=0),
            )
        },
        "recall_by_class": {
            name: float(value)
            for name, value in zip(
                CLASS_NAMES,
                recall_score(actual, predicted, labels=CLASS_IDS, average=None, zero_division=0),
            )
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "class_order": list(CLASS_NAMES),
    }
    if probabilities is not None:
        metrics["log_loss"] = float(log_loss(actual, probabilities, labels=CLASS_IDS))
        if set(np.unique(actual)) == set(CLASS_IDS):
            metrics["macro_ovr_roc_auc"] = float(
                roc_auc_score(actual, probabilities, labels=CLASS_IDS, multi_class="ovr", average="macro")
            )
        else:
            metrics["macro_ovr_roc_auc"] = None
    else:
        metrics["log_loss"] = None
        metrics["macro_ovr_roc_auc"] = None
    return metrics


def save_confusion_matrix(metrics: dict[str, Any], path: str | Path, title: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=CLASS_IDS,
        yticks=CLASS_IDS,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted result",
        ylabel="Actual result",
        title=title,
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Matches")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def save_roc_curves(actual: np.ndarray, probabilities: np.ndarray, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    one_vs_rest = label_binarize(actual, classes=CLASS_IDS)
    figure, axis = plt.subplots(figsize=(6, 5))
    for class_id, name in enumerate(CLASS_NAMES):
        false_positive_rate, true_positive_rate, _ = roc_curve(one_vs_rest[:, class_id], probabilities[:, class_id])
        axis.plot(false_positive_rate, true_positive_rate, label=name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Final-season one-vs-rest ROC")
    axis.legend(title="Actual class", loc="lower right")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
