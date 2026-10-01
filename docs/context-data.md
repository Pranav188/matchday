# Pre-match external context

Use `docs/context.example.csv` as the header template. Supply one snapshot per fixture/team/time. Run `python -m premier_league_predictor context --context-file /path/to/your.csv` to validate and import it, then run `train`. All timestamps must include a timezone. Match team names to the fixture catalog (`Man United`, `Tottenham`, etc.). Empty numeric fields mean **unknown**, never zero.

- `missing_minutes_share`: sum of recent minutes from injured/suspended players divided by total squad minutes over the same previous five matches. Include the full squad in the denominator; do not count doubtful players as certainly absent.
- `missing_attackers`, `missing_defenders`, `missing_goalkeepers`: absent regular starters by role. Define regular starter as starting at least three of the team's previous five matches. Attackers include midfielders and forwards. A player appearing in multiple reports is counted once.
- `suspended_starters`: unavailable regular starters specifically due to suspension.
- `lineup_minutes_share`: prior-five-match minutes of the expected/confirmed starting XI divided by total squad minutes. Record only lineups known at the snapshot time; a retrospectively published lineup is not an earlier forecast input.
- `manager_days`: days since the manager's appointment became effective, using the manager known at observation time.
- `all_matches_last7`, `all_matches_last14`: completed matches in all competitions, strictly before the forecast cutoff, including cup/European games. Leave blank when coverage is partial.

`observed_at` is when the record was available, not the injury start date. Historical date-only results conservatively accept context observed before UTC midnight on the match date; live forecasts use the actual prediction timestamp. Imports with late timestamps cannot affect earlier forecasts.

Context features enter training only when at least 300 non-missing, varying observations exist before July 2022, the first validation cutoff. This avoids selecting features from future coverage. Until sufficient timestamped history is imported, injuries, lineups, manager changes and cup workload are **context only**; they do not change model probabilities. This cutoff can be deliberately revised as a new evaluation protocol when a shorter historical dataset becomes available.

`availability` stores an official public FPL snapshot locally. Availability flags are informational, can be incomplete, and do not identify a learned player impact. Snapshots expire after 48 hours and are displayed only for fixtures within seven days. The refresh scheduler renews them; each snapshot records its source and observation time.
