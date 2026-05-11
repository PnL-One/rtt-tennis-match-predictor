from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT_ROOT / "notebooks"


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    print("$ " + " ".join(command), flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"Step failed: {name} (exit code {completed.returncode})")


def run_notebook(path: Path) -> None:
    run_step(
        f"Execute {path.relative_to(PROJECT_ROOT)}",
        [
            sys.executable,
            "-u",
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=-1",
            str(path),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RTT predictor update pipeline.")
    parser.add_argument("--skip-calendar", action="store_true", help="Skip RTT calendar/tournament master update.")
    parser.add_argument("--skip-matches", action="store_true", help="Skip match page download and parsing notebook.")
    parser.add_argument("--skip-rankings", action="store_true", help="Skip ranking parser notebook.")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip final dataset build notebook.")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training notebook.")
    args = parser.parse_args()

    print("Full RTT predictor pipeline")
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    print(f"Project root: {PROJECT_ROOT}")

    if not args.skip_calendar:
        run_step("Update tournament ids and details from RTT calendar", [sys.executable, "-u", "scripts/parse_rtt_calendar.py"])

    if not args.skip_matches:
        run_notebook(NOTEBOOKS / "01_save_and_parse_matches.ipynb")

    if not args.skip_rankings:
        run_notebook(NOTEBOOKS / "02_parse_rankings.ipynb")

    if not args.skip_dataset:
        run_notebook(NOTEBOOKS / "03_build_final_dataset.ipynb")

    if not args.skip_training:
        run_notebook(NOTEBOOKS / "04_train_final_model.ipynb")

    run_step("Refresh data manifest", [sys.executable, "-u", "scripts/data_status.py", "--write-manifest"])
    run_step("Verify project", [sys.executable, "-u", "scripts/verify_project.py"])

    print("\nPipeline finished successfully.")
    print("Open notebooks/04_train_final_model.ipynb and use the final prediction widget for a specific match.")


if __name__ == "__main__":
    main()
