# Matchday

A reproducible, pre-kickoff machine-learning study of English Premier League results: **home win (`H`), draw (`D`), or away win (`A`)**. The pipeline uses historical results and compares five classical classifiers. It is designed to make feature timing, evaluation choices, and limitations inspectable.

## Original held-out study

The model was selected using earlier seasons and evaluated once on all 380 fixtures in the reserved 2025/26 season.

| Selected model | Accuracy | Balanced accuracy | Macro F1 | Log loss | Macro OVR ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decision Tree | 0.434 | 0.418 | 0.419 | 1.053 | 0.609 |
| Majority / always-home baseline | 0.426 | 0.333 | 0.199 | — | — |

The final confusion matrix shows the largest error group was 48 home wins predicted as draws; draw recall was 0.288. The validation-selected Decision Tree improved macro F1 over the baseline, but the holdout result is modest and does not establish future performance.

![2025/26 test confusion matrix](reports/final_test_confusion_matrix.png)

See the [final-season ROC curves](reports/final_test_roc.png) for one-vs-rest class discrimination.

See the [full report](reports/project_report.md), [validation comparison](reports/validation_model_comparison.csv), [fold dates and scores](reports/validation_fold_scores.csv), and [final metrics](reports/final_test_metrics.json). The report also includes per-model validation confusion matrices, Logistic Regression coefficients, Decision Tree split rules, Random Forest and XGBoost feature importance, and an Elo ablation.

## Method

The audit covers 12,704 fixtures across 33 seasons (1993/94–2025/26). Predictors are five-match points and goal form, venue-specific form, rest days, prior head-to-head count, and pre-match Elo difference. Current-match result fields are labels and historical state updates only; they never enter that fixture's feature row. Same-date fixtures share one prior-history snapshot.

Model selection uses date-blocked expanding-window validation. Logistic Regression, a bounded Gini Decision Tree, Random Forest, XGBoost, and an RBF SVM are compared by macro F1 and log loss. Imputation and scaling remain inside each fold. The 2025/26 holdout is excluded from tuning. See the [feature contract](docs/feature_contract.md) and [data audit](reports/data_audit.json) for definitions, actual source headers, checks, and hashes.

The validation-only Elo ablation changed mean macro F1 from 0.368 without Elo to 0.465 with all features for the selected model. This is a descriptive experiment on the model-selection folds, not an independent or causal estimate.

The key methodological safeguard is explicit feature availability: same-fixture performance statistics are excluded, and every feature uses completed matches from earlier dates only. Fixtures on one date are also kept together during validation.

## Matchday application

Browse all 380 Premier League 2026/27 fixtures and predict an upcoming match without entering a date. The interface shows outcome probabilities, a separate Poisson goals estimate with likely scores, recent shots and league workload, model explanations, and an archive of forecasts saved before kickoff. Probability estimates and scorelines are uncertain forecasts.

The new probability-focused experiment compares Logistic Regression, Random Forest, raw and temporally calibrated trees over 5-year, 10-year and full-history training windows. Rolling features include previous 3/5/10-match points, shots, shots on target, defensive shot pressure, and 7/14-day league workload. Model selection uses mean validation log loss, then Brier score, on 2022/23--2024/25 seasons. The saved deployment artifact includes source hashes, feature schema, model version, historical state and goals models. The API loads it without retraining or replaying history per request.

The [advanced evaluation](reports/advanced_evaluation.json) records all experiments, probability baselines and a **retrospective** 2025/26 benchmark. That season was already inspected during the original study, so this is not a new blind test. The [reliability diagram](reports/probability_reliability.png) compares forecast probabilities with observed frequencies. The original study and report above remain unchanged. New prospective performance is measured using the earliest saved forecast per fixture, even when models are updated.

The current deployment selects full-history Logistic Regression. Its retrospective accuracy is **48.4%** (always-home: **42.6%**), macro F1 **0.358**, log loss **1.047**, and Brier score **0.627**. Compared with the original tree, accuracy is higher and macro F1 is lower. No draws were selected as the most likely class in this benchmark. Draw probabilities are still estimated; this model does not establish reliable draw classification or future improvement. The separate goals model has mean absolute errors of **0.950 home goals** and **0.832 away goals** on the same retrospective season.

### Projected final table

Open `/standings`, or use **Final table** in the header. This separate page lists all 20 clubs with matches left, current points and expected final points. Completed results award actual points; remaining fixtures contribute `3 × P(win) + P(draw)` to each club. Positions use expected points, with equal totals sharing a position. Decimal points describe averages, not a single simulated season or official goal-difference tiebreaks.

The projection uses the same artifact, historical form and timestamped inputs as match predictions. Form remains fixed until the model refreshes; unknown future injuries and changes in form are not simulated. Calculating the table does not populate the saved forecast archive. Fixtures awaiting completed results must refresh before a new projection is shown.

### Start and stop

Install Python dependencies using the setup below, then:

```sh
python -m premier_league_predictor download
python -m premier_league_predictor train
npm --prefix web ci
./scripts/dev.sh
```

Open http://localhost:3000. Ctrl+C stops both services. Alternatively, run `python -m premier_league_predictor.api` in one terminal and `npm --prefix web run dev` in another. For a production preview, use `npm --prefix web run build` followed by `npm --prefix web run start`.

### Refresh and external context

```sh
python -m premier_league_predictor refresh
# Optional scheduler: refresh immediately, then every 24 hours.
python scripts/refresh.py
# Validate/import timestamped external context, then retrain.
python -m premier_league_predictor context --context-file /path/to/context.csv
python -m premier_league_predictor train
```

The running API reloads atomic model, fixture and context updates on its next request. Imported injury/suspension, expected lineup, manager and all-competition workload features follow the [context data contract](docs/context-data.md), with a [CSV header template](docs/context.example.csv). Missing information remains unknown. Historical coverage is required before external factors can influence training; the current trained model has **no learned injury or lineup effects**. Public FPL availability snapshots are informational, expire after 48 hours, and are displayed only for fixtures within seven days.

Forecasts are stored locally in `data/history/forecasts.sqlite`, excluded from Git. Each record captures prediction time, kickoff, feature values, probabilities, model version and data cutoff. Completed results are reconciled when fixture snapshots refresh. Viewing past predictions never regenerates them using later results.

### Containers and deployment

After training a local artifact:

```sh
docker compose up --build
# From another terminal:
docker compose down
```

Only the web service is exposed, at localhost:3000. The API model volume is read-only, and a named volume persists forecast history. For a public demonstration, host these services behind your platform's HTTPS endpoint and supply persistent model/context/history volumes. No public deployment has been provisioned. CI runs Python tests and frontend type/build checks on pushes and pull requests.

## Project layout

```text
src/premier_league_predictor/  features, evaluation, deployment models, API, history
web/                           Next.js and React forecast interface
tests/                         data, feature-timing, and date-split checks
docs/                          pre-kickoff feature contract
data/fixtures/                 stored 2026/27 schedule with source and refresh timestamp
data/raw/                      downloaded season CSVs (ignored by Git)
data/processed/                generated features (ignored by Git)
models/                        fitted estimator (ignored by Git)
reports/                       audit, metrics, figures, and project report
```

## Run locally

Python 3.12 or newer is required. On macOS, install OpenMP for XGBoost if it is not already available (`brew install libomp`). Then:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
python -m premier_league_predictor download
python -m premier_league_predictor audit
python -m pytest
python -m premier_league_predictor evaluate --n-jobs 2
```

The download command retrieves source CSVs into `data/raw/`; the audit validates them and records per-file SHA-256 hashes. Evaluation rebuilds the feature table, searches models, evaluates the selected model on the reserved season, and writes reproducible outputs to `reports/`, `models/`, and `data/processed/`. Direct dependency versions are pinned in the requirements files.

## Data and attribution

Match results come from [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php). The source and the independent reference project are documented in [ATTRIBUTION.md](ATTRIBUTION.md). The reference informed data and feature-engineering choices; its code, report, and advanced-stat data were not copied.
