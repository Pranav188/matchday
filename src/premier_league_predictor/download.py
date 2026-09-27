"""Download season result CSVs from Football-Data.co.uk."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
REQUIRED_COLUMNS = {"date", "hometeam", "awayteam", "fthg", "ftag", "ftr"}


def season_code(start_year: int) -> str:
    """Return the source's four-digit season code, such as 2526."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download_seasons(
    data_dir: str | Path,
    start_year: int = 1993,
    end_year: int = 2026,
    force: bool = False,
) -> list[Path]:
    """Download available Premier League result files; leave existing files intact."""
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year")

    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / ".download_manifest.json"
    if metadata_path.exists():
        retrieval_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        retrieval_metadata = {}
    downloaded: list[Path] = []
    missing: list[str] = []

    for year in range(start_year, end_year + 1):
        code = season_code(year)
        path = destination / f"{code}_E0.csv"
        if path.exists() and not force:
            downloaded.append(path)
            continue

        request = Request(
            BASE_URL.format(season=code),
            headers={"User-Agent": "premier-league-course-project/0.1"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read()
        except HTTPError as error:
            if error.code == 404:
                missing.append(code)
                continue
            raise RuntimeError(f"Download failed for season {code}: HTTP {error.code}") from error
        except URLError as error:
            raise RuntimeError(f"Download failed for season {code}: {error.reason}") from error

        header_line = payload.splitlines()[0].decode("utf-8-sig", errors="replace")
        header = next(csv.reader([header_line]), [])
        normalized_header = {column.strip().lower() for column in header}
        if not REQUIRED_COLUMNS.issubset(normalized_header):
            raise ValueError(f"Unexpected CSV header for season {code}: {header}")

        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_bytes(payload)
        temporary_path.replace(path)
        downloaded.append(path)
        retrieval_metadata[path.name] = {
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "timestamp_source": "downloaded_by_pipeline",
        }
        print(f"Downloaded {code}: {path}")

    if missing:
        print(f"No source file found for season codes: {', '.join(missing)}")
    for path in downloaded:
        if path.name not in retrieval_metadata:
            retrieval_metadata[path.name] = {
                "retrieved_at_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(timespec="seconds"),
                "timestamp_source": "inferred_from_file_mtime",
            }
    metadata_path.write_text(
        json.dumps(retrieval_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return downloaded
