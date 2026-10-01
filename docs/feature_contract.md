# Pre-Kickoff Feature Contract

## Prediction and Source Columns

Predict the full-time result from the information available before kickoff. The expected Football-Data.co.uk result columns are `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, and `FTR`; Phase 1 must verify and normalize each season's actual headers. `FTR` is the label, with `FTHG` and `FTAG` used to validate it and update historical team state only after the fixture's feature row has been created. Team names and dates are state keys, not initial model features.

| Feature | Definition at the fixture cutoff |
| --- | --- |
| `home_points_last5`, `away_points_last5` | Sum of 3/1/0 points from each team's five most recent prior league matches, at either venue; use all available prior matches if fewer than five. |
| `home_goals_for_last5`, `away_goals_for_last5` | Mean goals scored in those prior matches. |
| `home_goals_against_last5`, `away_goals_against_last5` | Mean goals conceded in those prior matches. |
| `home_home_points_last5` | Points from the home team's five most recent prior home matches. |
| `away_away_points_last5` | Points from the away team's five most recent prior away matches. |
| `home_rest_days`, `away_rest_days` | Calendar days since each team's previous completed league match. |
| `head_to_head_count` | Number of earlier Premier League meetings between these clubs, regardless of venue. |
| `elo_difference` | Home pre-match Elo minus away pre-match Elo. Initialize clubs at 1500; use K=20, a 60-point home advantage, and the standard 400-point Elo scale. Update ratings only after features are captured. |

If a team has no prior match for a history feature, represent it as missing for training-fold imputation; the head-to-head count starts at zero and initial Elo difference is zero. Do not use raw team names as predictors in the initial model set.

## Timing and Leakage Rules

Sort matches chronologically. The current fixture's goals and result may not affect any of its features. Because the accepted input contract guarantees a date but not a reliable kickoff timestamp, calculate features for every match on the same date from histories ending before that date; update all team histories only after the date's feature rows are saved. This date-batch rule prevents same-day result leakage.

Exclude current-fixture goals, half-time results, shots, corners, cards, odds, and post-match standings from predictors. Historical match outcomes may contribute to later rolling features and Elo only after the historical fixtures have been completed. Keep all imputation, scaling, feature selection, and tuning inside training folds. The 2025/26 season remains untouched until one final evaluation.

## Future-Fixture Inference

For deployment forecasts, fit the validation-selected model on all completed matches currently available after the evaluation protocol is frozen. This may include the historical 2025/26 holdout; its published test metrics remain the fixed historical benchmark. Build the requested fixture's feature row from matches dated strictly before its fixture date. Do not update form, rest, head-to-head, or Elo state with an assumed result. Reject fixture dates on or before the latest result in the loaded data. Unplayed rows in a current-season source CSV are not training examples.


## Extended deployment features

The original study's result-only features and metrics remain frozen. `train` enables optional historical shots and shots-on-target columns (`HS`, `AS`, `HST`, `AST`) and constructs rolling averages **after** completed matches update state. Current-fixture shots never enter its feature row. Missing source statistics are kept missing, and medians are learned within the training period. Recent attacking/defensive shot statistics use five prior matches; additional form summaries use three and ten prior matches. League workload counts matches in `[fixture date - 7/14 days, fixture date)`.

The deployment model is selected by probability quality on recent chronological validation seasons. Optional context obeys the separate timestamped contract. The Poisson goals models estimate home and away goal counts independently, so their scoreline probabilities may differ from the outcome classifier's probabilities. They share the same pre-match features and report a separate retrospective goals-error benchmark.

The new 2025/26 comparison is explicitly retrospective because this season was already inspected. The original held-out report is preserved; forecasts saved before actual future kickoff provide the prospective evaluation. Calibration reserves a later season inside each training period, never the validation/test season. Saved model artifacts include historical state and source hashes, with an atomic pointer for reloads.
