# Repository Guidelines

## Project Scope and Structure

This is a classical machine-learning project for predicting Premier League results (`H`, `D`, `A`) before kickoff. The project brief is at the repository root. Use `src/premier_league_predictor/` for reusable pipeline code, `tests/` for automated checks, `notebooks/` for focused exploration, `data/raw/` and `data/processed/` for data, `models/` for saved estimators, and `reports/` for audits and results. Keep the main analysis reproducible and explainable; optional UI or clustering work comes after the core pipeline.

## Setup and Commands

Set up a Python 3.12 environment with:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
```

Run `python -m pytest` for checks. Use `python -m premier_league_predictor download`, `audit`, and `evaluate --n-jobs 2` for the data and modeling workflow. On macOS, install `libomp` if XGBoost cannot find OpenMP. Keep exact dependency versions in the requirements files and record source-file hashes in audits.

## Python Style and Naming

Use four spaces for indentation, `snake_case` for modules, functions, and variables, and descriptive names for features and metrics. Keep feature generation, preprocessing, model fitting, and evaluation in small reusable functions under `src/`. Avoid adding dependencies until a project requirement needs them.

## Data, Modeling, and Tests

Only use information available before each fixture. Build and save a fixture's feature row before updating team history with that match. Split by time, reserve the newest complete season for final testing, and fit preprocessing only within training folds. Add tests under `tests/` named `test_*.py`; prioritize proving feature timing and leakage prevention, including that changing a fixture's result does not change its own pre-match features.

## Changes and Reviews

No Git history is present yet, so there is no established commit format. Use short imperative commit subjects (for example, `Add rolling-form feature tests`). Pull requests should explain the modeling or data change, describe validation performed, link relevant coursework issues, and include figures when results or plots change. Preserve source licenses and attribution for reused ideas or data.
