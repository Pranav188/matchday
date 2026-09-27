# Premier League Match Outcome Prediction

## Question

Can classical supervised models predict home win (`H`), draw (`D`), or away win (`A`) from information available before a Premier League fixture?

## Data

The audited dataset contains 12,704 fixtures from 33 seasons, spanning 1993-08-14 to 2026-05-24. The untouched final test season contains 380 matches from 2025/26. The source is [Football-Data.co.uk's England results archive](https://www.football-data.co.uk/englandm.php); 33 season files were audited.

| Result | Matches |
| --- | ---: |
| Home win | 5,797 |
| Draw | 3,246 |
| Away win | 3,661 |

The six required source headers are recorded for each season in [`data_audit.json`](data_audit.json), along with target distribution, duplicate checks, ignored blank rows, and SHA-256 hashes. Only date, home/away team, full-time goals, and full-time result are used. The current fixture's result creates its label but never its features. Matches with the same date share a pre-date history snapshot before results update team state. See [`docs/feature_contract.md`](../docs/feature_contract.md).

## Evaluation

The newest complete season, 2025/26, was reserved before model selection. Earlier seasons were evaluated with five expanding-window folds. All fixtures on a calendar date stay in one fold because kickoff times are unavailable. Imputation and scaling are fitted inside each training fold. The selected model was chosen by validation macro F1, with validation log loss as the tie-breaker; it was evaluated once on the final season.

| Fold | Training dates | Validation dates | Training matches | Validation matches |
| ---: | --- | --- | ---: | ---: |
| 1 | 1993-08-14 to 1998-09-08 | 1998-09-09 to 2004-03-17 | 2,095 | 2,149 |
| 2 | 1993-08-14 to 2004-03-17 | 2004-03-20 to 2009-12-29 | 4,244 | 2,194 |
| 3 | 1993-08-14 to 2009-12-29 | 2009-12-30 to 2015-08-16 | 6,438 | 2,105 |
| 4 | 1993-08-14 to 2015-08-16 | 2015-08-17 to 2020-12-13 | 8,543 | 1,998 |
| 5 | 1993-08-14 to 2020-12-13 | 2020-12-15 to 2025-05-25 | 10,541 | 1,783 |

Per-model scores for each fold are in [`validation_fold_scores.csv`](validation_fold_scores.csv).

| Model | Train macro F1 | Validation macro F1 | Validation log loss |
| --- | ---: | ---: | ---: |
| Decision Tree | 0.445 | 0.465 | 1.017 |
| Random Forest | 0.663 | 0.452 | 1.012 |
| Logistic Regression | 0.431 | 0.446 | 1.011 |
| XGBoost | 0.650 | 0.408 | 1.019 |
| RBF SVM | 0.361 | 0.376 | 1.009 |

### Elo feature ablation

The selected model and tuned parameters were refit on the same expanding-window folds, once with all features and once without `elo_difference`. This validation-only diagnostic suggests that Elo improved the selected model's mean fold macro F1; it is not an independent causal estimate.

| Feature set | Mean validation macro F1 | Mean validation log loss |
| --- | ---: | ---: |
| All features | 0.465 | 1.017 |
| Without Elo | 0.368 | 1.078 |

Selected model: **Decision Tree**

### Validation error review

Pooled fold confusion matrices are available for all five models. These are descriptive predictions from the same folds used for tuning, not a second independent evaluation.

- Logistic Regression: [confusion matrix](validation_logistic_regression_confusion_matrix.png)
- Decision Tree: [confusion matrix](validation_decision_tree_confusion_matrix.png)
- Random Forest: [confusion matrix](validation_random_forest_confusion_matrix.png)
- XGBoost: [confusion matrix](validation_xgboost_confusion_matrix.png)
- RBF SVM: [confusion matrix](validation_rbf_svm_confusion_matrix.png)

The bounded Decision Tree split rules are in [`decision_tree_rules.txt`](decision_tree_rules.txt). Standardized Logistic Regression coefficients are in [`logistic_regression_coefficients.csv`](logistic_regression_coefficients.csv). The top Random Forest features by impurity importance are:

| Feature | Importance |
| --- | ---: |
| `elo_difference` | 0.343 |
| `head_to_head_count` | 0.082 |
| `home_home_points_last5` | 0.068 |
| `away_away_points_last5` | 0.067 |
| `home_points_last5` | 0.061 |

| Metric | Final test |
| --- | ---: |
| Accuracy | 0.434 |
| Balanced accuracy | 0.418 |
| Macro F1 | 0.419 |
| Log loss | 1.053 |
| Cohen's kappa | 0.142 |
| Macro one-vs-rest ROC AUC | 0.609 |

| Result | Precision | Recall |
| --- | ---: | ---: |
| H | 0.575 | 0.519 |
| D | 0.259 | 0.288 |
| A | 0.432 | 0.447 |

The largest error group was 48 actual H matches predicted as D.

Of 104 actual draws, 37 were predicted as home wins and 37 as away wins. Draw recall is 0.288.

The majority-class and always-home baselines were also evaluated on the same final season:

| Baseline | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Majority class | 0.426 | 0.199 |
| Always home | 0.426 | 0.199 |

Home wins were the most common final-season class, so these baselines make identical predictions.

The final-season confusion matrix and ROC curves are saved alongside this report. Detailed metrics are in [`final_test_metrics.json`](final_test_metrics.json).

## Limitations

This is a historical, single-league evaluation. Results-based form and Elo features do not include lineups, injuries, tactical changes, or other team news. The date-batch rule is conservative when kickoff timestamps are unavailable. A held-out season provides evidence about this evaluation period; it does not establish future performance, a causal explanation, or betting profitability.

## Reproducibility

Raw source CSVs are excluded from version control. The audit records each filename, required headers, byte count, and SHA-256. Direct dependency versions are pinned in `requirements.txt` and `requirements-dev.txt`; documented download, audit, test, and evaluation commands are in the README.
