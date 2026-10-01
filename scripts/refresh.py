#!/usr/bin/env python3
"""Refresh immediately and then daily; a failed run keeps the last model."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--interval-hours", type=float, default=24)
parser.add_argument("--once", action="store_true")
args = parser.parse_args()
if args.interval_hours <= 0:
    parser.error("interval-hours must be positive")
root = Path(__file__).resolve().parents[1]
try:
    while True:
        result = subprocess.run(
            [sys.executable, "-m", "premier_league_predictor", "refresh"], cwd=root
        )
        print(
            f"{datetime.now(timezone.utc).isoformat()} refresh exit={result.returncode}",
            flush=True,
        )
        if args.once:
            raise SystemExit(result.returncode)
        time.sleep(args.interval_hours * 3600)
except KeyboardInterrupt:
    print("Refresh scheduler stopped.")
