# Attribution and Data Provenance

## Match results

Historical English Premier League results are downloaded from the [Football-Data.co.uk England archive](https://www.football-data.co.uk/englandm.php), using its `E0` season CSVs. The pipeline selects only `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, and `FTR`; it does not use odds or current-match statistics as predictors. Local raw CSVs are excluded from version control. The audit records each source filename, required headers, retrieval timestamp (marked when inferred from local file metadata), byte count, and SHA-256 hash. Check the source site for its current terms before redistributing data.

## Fixture schedule

The checked-in 2026/27 schedule comes from [Fixture Download](https://fixturedownload.com/results/epl-2026), using its [JSON feed](https://fixturedownload.com/feed/json/epl-2026). The published season dates were checked against the [Premier League fixture announcement](https://www.premierleague.com/en/news/4675097/all-380-fixtures-for-202627-premier-league-season). Each snapshot records the source URL and retrieval time; fixtures and kickoff times may change. Club aliases are normalized to match the Football-Data result history. Schedule scores are displayed only; training results continue to come from Football-Data. Run `python -m premier_league_predictor fixtures` to refresh.

## Reference project

The coursework brief identifies [ScooterStuff/beat-the-bookie](https://github.com/ScooterStuff/beat-the-bookie) as background for public data organization and feature-engineering ideas. Its repository is MIT-licensed. This project does not copy its source code, report text, or advanced-statistics data; the pipeline and analysis here are independently implemented. If code or data is reused in a future change, include its license text and exact attribution.

The independent implementation makes feature timing explicit: current-fixture statistics are excluded, same-date fixtures share prior history, and validation is chronological.

## Public player availability

Current player availability is fetched from the official [Fantasy Premier League public endpoint](https://fantasy.premierleague.com/api/bootstrap-static/) and stored with the retrieval time. Flags identify reported injuries, suspensions and doubts; they are not a complete medical record and do not establish historical pre-match availability. Current snapshots are context only until suitable historical imports exist. The full snapshot is local and excluded from Git.
