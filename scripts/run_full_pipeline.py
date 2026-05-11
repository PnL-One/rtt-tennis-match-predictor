from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    print("$ " + " ".join(command), flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"Step failed: {name} (exit code {completed.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RTT predictor update pipeline.")
    parser.add_argument("--skip-calendar", action="store_true", help="Skip RTT calendar/tournament master update.")
    parser.add_argument("--skip-matches", action="store_true", help="Skip match page download and parsing.")
    parser.add_argument("--skip-rankings", action="store_true", help="Skip ranking parser.")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip final dataset build.")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training.")
    args = parser.parse_args()

    print("Full RTT predictor pipeline")
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    print(f"Project root: {PROJECT_ROOT}")

    if not args.skip_calendar:
        run_step("Update tournament ids and details from RTT calendar", [sys.executable, "-u", "scripts/parse_rtt_calendar.py"])

    if not args.skip_matches:
        run_step("Download and parse RTT match pages", [sys.executable, "-u", "scripts/parse_rtt_matches.py"])

    if not args.skip_rankings:
        run_step("Download and parse RTT rankings", [sys.executable, "-u", "scripts/parse_rtt_rankings.py"])

    if not args.skip_dataset:
        run_step("Build final model dataset", [sys.executable, "-u", "scripts/build_final_dataset.py"])

    if not args.skip_training:
        run_step("Train model and save diagnostics", [sys.executable, "-u", "scripts/train_model.py"])

    run_step("Refresh data manifest", [sys.executable, "-u", "scripts/data_status.py", "--write-manifest"])
    run_step("Verify project", [sys.executable, "-u", "scripts/verify_project.py"])

    print("\nPipeline finished successfully.")
    print("Open notebooks/04_train_final_model.ipynb and use the final prediction widget for a specific match.")


if __name__ == "__main__":
    main()
