"""Write a concise, evidence-bounded project report from saved results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from premier_league_predictor.constants import MODEL_DISPLAY_NAMES


def _common_error(metrics: dict[str, Any]) -> str:
    matrix = metrics["confusion_matrix"]
    labels = metrics["class_order"]
    mistakes = [
        (matrix[row][column], labels[row], labels[column])
        for row in range(len(labels))
        for column in range(len(labels))
        if row != column
    ]
    count, actual, predicted = max(mistakes, default=(0, "", ""))
    if not count:
        return "No off-diagonal errors were recorded."
    return f"The largest error group was {count} actual {actual} matches predicted as {predicted}."


def _validation_table(comparison: pd.DataFrame) -> str:
    header = "| Model | Train macro F1 | Validation macro F1 | Validation log loss |\n| --- | ---: | ---: | ---: |"
    rows = [
        f"| {MODEL_DISPLAY_NAMES[row.model]} | {row.training_macro_f1:.3f} | "
        f"{row.validation_macro_f1:.3f} | {row.validation_log_loss:.3f} |"
        for row in comparison.itertuples(index=False)
    ]
    return "\n".join([header, *rows])


def _ablation_table(ablation: pd.DataFrame) -> str:
    header = "| Feature set | Mean validation macro F1 | Mean validation log loss |\n| --- | ---: | ---: |"
    display_names = {"all_features": "All features", "without_elo": "Without Elo"}
    rows = [
        f"| {display_names.get(row.feature_set, row.feature_set)} | "
        f"{row.mean_validation_macro_f1:.3f} | {row.mean_validation_log_loss:.3f} |"
        for row in ablation.itertuples(index=False)
    ]
    return "\n".join([header, *rows])


def _fold_table(scores: pd.DataFrame) -> str:
    folds = scores[
        [
            "fold",
            "training_start",
            "training_through",
            "validation_start",
            "validation_through",
            "training_matches",
            "validation_matches",
        ]
    ].drop_duplicates("fold")
    header = (
        "| Fold | Training dates | Validation dates | Training matches | Validation matches |\n"
        "| ---: | --- | --- | ---: | ---: |"
    )
    rows = [
        f"| {row.fold} | {row.training_start} to {row.training_through} | "
        f"{row.validation_start} to {row.validation_through} | "
        f"{row.training_matches:,} | {row.validation_matches:,} |"
        for row in folds.itertuples(index=False)
    ]
    return "\n".join([header, *rows])


def write_project_report(
    audit: dict[str, Any],
    result: dict[str, Any],
    source_files: list[dict[str, Any]],
    destination: str | Path,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = result["final_test"]
    target_counts = audit["target_counts"]
    source_file_count = len(source_files)
    baseline_display_names = {"majority_class": "Majority class", "always_home": "Always home"}
    baseline_rows = "\n".join(
        f"| {baseline_display_names.get(name, name)} | {values['accuracy']:.3f} | {values['macro_f1']:.3f} |"
        for name, values in result["baselines"].items()
    )
    class_rows = "\n".join(
        f"| {name} | {metrics['precision_by_class'][name]:.3f} | {metrics['recall_by_class'][name]:.3f} |"
        for name in metrics["class_order"]
    )
    validation_figures = "\n".join(
        f"- {MODEL_DISPLAY_NAMES[row.model]}: "
        f"[confusion matrix](validation_{row.model}_confusion_matrix.png)"
        for row in result["validation_diagnostics"].itertuples(index=False)
    )
    feature_importance = pd.read_csv(path.parent / "tree_feature_importance.csv")
    random_forest_importance = feature_importance.loc[
        feature_importance["model"] == "random_forest"
    ].nlargest(5, "importance")
    importance_rows = "\n".join(
        f"| `{row.feature}` | {row.importance:.3f} |"
        for row in random_forest_importance.itertuples(index=False)
    )
    confusion = metrics["confusion_matrix"]
    draw_as_home = confusion[1][0]
    draw_as_away = confusion[1][2]
    baselines_are_equal = (
        result["baselines"]["always_home"]["confusion_matrix"]
        == result["baselines"]["majority_class"]["confusion_matrix"]
    )
    baseline_note = (
        "Home wins were the most common final-season class, so these baselines make identical predictions."
        if baselines_are_equal
        else "The majority-class baseline predicts the most common training result; always-home predicts H for every fixture."
    )

    content = f"""# Premier League Match Outcome Prediction

## Question

Can classical supervised models predict home win (`H`), draw (`D`), or away win (`A`) from information available before a Premier League fixture?

## Data

The audited dataset contains {audit['rows']:,} fixtures from {len(audit['seasons'])} seasons, spanning {audit['date_start']} to {audit['date_end']}. The untouched final test season contains {result['final_test_matches']} matches from 2025/26. The source is [Football-Data.co.uk's England results archive](https://www.football-data.co.uk/englandm.php); {source_file_count} season files were audited.

| Result | Matches |
| --- | ---: |
| Home win | {target_counts['H']:,} |
| Draw | {target_counts['D']:,} |
| Away win | {target_counts['A']:,} |

The six required source headers are recorded for each season in [`data_audit.json`](data_audit.json), along with target distribution, duplicate checks, ignored blank rows, and SHA-256 hashes. Only date, home/away team, full-time goals, and full-time result are used. The current fixture's result creates its label but never its features. Matches with the same date share a pre-date history snapshot before results update team state. See [`docs/feature_contract.md`](../docs/feature_contract.md).

## Evaluation

The newest complete season, 2025/26, was reserved before model selection. Earlier seasons were evaluated with five expanding-window folds. All fixtures on a calendar date stay in one fold because kickoff times are unavailable. Imputation and scaling are fitted inside each training fold. The selected model was chosen by validation macro F1, with validation log loss as the tie-breaker; it was evaluated once on the final season.

{_fold_table(result['validation_fold_scores'])}

Per-model scores for each fold are in [`validation_fold_scores.csv`](validation_fold_scores.csv).

{_validation_table(result['validation_comparison'])}

### Elo feature ablation

The selected model and tuned parameters were refit on the same expanding-window folds, once with all features and once without `elo_difference`. This validation-only diagnostic suggests that Elo improved the selected model's mean fold macro F1; it is not an independent causal estimate.

{_ablation_table(result['validation_elo_ablation'])}

Selected model: **{MODEL_DISPLAY_NAMES[result['selected_model']]}**

### Validation error review

Pooled fold confusion matrices are available for all five models. These are descriptive predictions from the same folds used for tuning, not a second independent evaluation.

{validation_figures}

The bounded Decision Tree split rules are in [`decision_tree_rules.txt`](decision_tree_rules.txt). Standardized Logistic Regression coefficients are in [`logistic_regression_coefficients.csv`](logistic_regression_coefficients.csv). The top Random Forest features by impurity importance are:

| Feature | Importance |
| --- | ---: |
{importance_rows}

| Metric | Final test |
| --- | ---: |
| Accuracy | {metrics['accuracy']:.3f} |
| Balanced accuracy | {metrics['balanced_accuracy']:.3f} |
| Macro F1 | {metrics['macro_f1']:.3f} |
| Log loss | {metrics['log_loss']:.3f} |
| Cohen's kappa | {metrics['cohen_kappa']:.3f} |
| Macro one-vs-rest ROC AUC | {metrics['macro_ovr_roc_auc']:.3f} |

| Result | Precision | Recall |
| --- | ---: | ---: |
{class_rows}

{_common_error(metrics)}

Of {sum(confusion[1])} actual draws, {draw_as_home} were predicted as home wins and {draw_as_away} as away wins. Draw recall is {metrics['recall_by_class']['D']:.3f}.

The majority-class and always-home baselines were also evaluated on the same final season:

| Baseline | Accuracy | Macro F1 |
| --- | ---: | ---: |
{baseline_rows}

{baseline_note}

The final-season confusion matrix and ROC curves are saved alongside this report. Detailed metrics are in [`final_test_metrics.json`](final_test_metrics.json).

## Limitations

This is a historical, single-league evaluation. Results-based form and Elo features do not include lineups, injuries, tactical changes, or other team news. The date-batch rule is conservative when kickoff timestamps are unavailable. A held-out season provides evidence about this evaluation period; it does not establish future performance, a causal explanation, or betting profitability.

## Reproducibility

Raw source CSVs are excluded from version control. The audit records each filename, required headers, byte count, and SHA-256. Direct dependency versions are pinned in `requirements.txt` and `requirements-dev.txt`; documented download, audit, test, and evaluation commands are in the README.
"""
    path.write_text(content, encoding="utf-8")
    return path
