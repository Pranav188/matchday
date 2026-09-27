"""Command-line workflows for downloading, auditing, and evaluating results."""

from __future__ import annotations

import argparse
from pathlib import Path

from premier_league_predictor.data import audit_matches, load_matches, source_manifest, write_audit
from premier_league_predictor.constants import MODEL_DISPLAY_NAMES
from premier_league_predictor.download import BASE_URL, download_seasons
from premier_league_predictor.features import build_pre_match_features


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pl-predictor")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download historical E0 season CSVs")
    download.add_argument("--data-dir", default="data/raw")
    download.add_argument("--from-year", type=int, default=1993)
    download.add_argument("--through-year", type=int, default=2026)
    download.add_argument("--force", action="store_true")

    fixtures = commands.add_parser("fixtures", help="refresh the complete 2026/27 schedule")
    fixtures.add_argument("--output", default="data/fixtures/2026-27.json")

    audit = commands.add_parser("audit", help="validate and summarize local season CSVs")
    audit.add_argument("--data-dir", default="data/raw")
    audit.add_argument("--output", default="reports/data_audit.json")

    evaluate = commands.add_parser("evaluate", help="tune on earlier seasons and test on 2025/26")
    evaluate.add_argument("--data-dir", default="data/raw")
    evaluate.add_argument("--reports-dir", default="reports")
    evaluate.add_argument("--models-dir", default="models")
    evaluate.add_argument("--n-splits", type=int, default=5)
    evaluate.add_argument("--n-jobs", type=int, default=-1)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fixtures":
        from premier_league_predictor.fixtures import refresh_schedule
        print(f"Saved 380 fixtures to {refresh_schedule(args.output)}")
        return
    if args.command == "download":
        paths = download_seasons(
            args.data_dir,
            start_year=args.from_year,
            end_year=args.through_year,
            force=args.force,
        )
        print(f"Available season files: {len(paths)}")
        return

    matches = load_matches(args.data_dir)
    audit = audit_matches(matches)
    audit["source_files"] = source_manifest(args.data_dir)
    audit["source_url_template"] = BASE_URL

    if args.command == "audit":
        destination = write_audit(audit, args.output)
        print(
            f"Audited {audit['rows']:,} fixtures across {len(audit['seasons'])} seasons "
            f"({audit['date_start']} to {audit['date_end']})."
        )
        print(
            f"Target counts H/D/A: {audit['target_counts']['H']}/"
            f"{audit['target_counts']['D']}/{audit['target_counts']['A']}; "
            f"duplicate rows: {audit['duplicate_fixture_rows']}; "
            f"blank rows ignored: {audit['ignored_blank_rows']}."
        )
        print(f"Audit and source hashes written to {destination}")
        return

    reports_dir = Path(args.reports_dir)
    write_audit(audit, reports_dir / "data_audit.json")
    if audit["duplicate_fixture_rows"]:
        raise ValueError("Duplicate fixtures were found; inspect reports/data_audit.json before evaluation")
    from premier_league_predictor.modeling import run_model_comparison
    from premier_league_predictor.reporting import write_project_report

    feature_table = build_pre_match_features(matches)
    processed_path = Path("data/processed/pre_match_features.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(processed_path, index=False)

    result = run_model_comparison(
        feature_table,
        reports_dir=reports_dir,
        models_dir=args.models_dir,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
    )
    write_project_report(audit, result, audit["source_files"], reports_dir / "project_report.md")
    print("\nValidation comparison")
    print(result["validation_comparison"].to_string(index=False))
    print("\nElo ablation")
    print(result["validation_elo_ablation"].to_string(index=False))
    print(f"Selected model: {MODEL_DISPLAY_NAMES[result['selected_model']]}")
    print("Final test metrics")
    print(
        f"accuracy={result['final_test']['accuracy']:.3f}, "
        f"macro_f1={result['final_test']['macro_f1']:.3f}, "
        f"balanced_accuracy={result['final_test']['balanced_accuracy']:.3f}, "
        f"log_loss={result['final_test']['log_loss']:.3f}"
    )
    print(f"Report written to {reports_dir / 'project_report.md'}")


if __name__ == "__main__":
    main()
