from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PROJECT_PYTHON = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    print("$ " + " ".join(command), flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"Step failed: {name} (exit code {completed.returncode})")


def preflight_check(args: argparse.Namespace) -> None:
    required = ["pandas", "openpyxl", "bs4"]
    if not args.skip_calendar or not args.skip_matches or not args.skip_rankings:
        required.append("playwright")
    if not args.skip_rankings:
        required.append("tqdm")
    if not args.skip_training:
        required.extend(["catboost", "sklearn"])

    print("\n=== Preflight dependency check ===", flush=True)
    print(f"Python executable: {PROJECT_PYTHON}", flush=True)

    probe_code = (
        "import importlib.util, sys; "
        f"packages={required!r}; "
        "missing=[p for p in packages if importlib.util.find_spec(p) is None]; "
        "print(','.join(missing)); "
        "sys.exit(1 if missing else 0)"
    )
    probe = subprocess.run(
        [str(PROJECT_PYTHON), "-c", probe_code],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    missing_text = (probe.stdout or "").strip()
    if probe.returncode != 0:
        print("Missing Python packages: " + missing_text, flush=True)
        print("Install them with:", flush=True)
        print(f"{PROJECT_PYTHON} -m pip install -r requirements.txt", flush=True)
        if "playwright" in missing_text.split(","):
            print(f"{PROJECT_PYTHON} -m playwright install firefox", flush=True)
        raise SystemExit(1)
    print("All required Python packages are importable.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RTT predictor update pipeline.")
    parser.add_argument("--skip-calendar", action="store_true", help="Skip RTT calendar/tournament master update.")
    parser.add_argument("--skip-matches", action="store_true", help="Skip match page download and parsing.")
    parser.add_argument("--skip-rankings", action="store_true", help="Skip ranking parser.")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip final dataset build.")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training.")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies and exit.")
    args = parser.parse_args()

    print("Full RTT predictor pipeline")
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Project Python: {PROJECT_PYTHON}")

    preflight_check(args)
    if args.check_only:
        print("Preflight check finished successfully.")
        return

    if not args.skip_calendar:
        run_step("Update tournament ids and details from RTT calendar", [str(PROJECT_PYTHON), "-u", "scripts/parse_rtt_calendar.py"])

    if not args.skip_matches:
        run_step("Download and parse RTT match pages", [str(PROJECT_PYTHON), "-u", "scripts/parse_rtt_matches.py"])

    if not args.skip_rankings:
        run_step("Download and parse RTT rankings", [str(PROJECT_PYTHON), "-u", "scripts/parse_rtt_rankings.py"])

    if not args.skip_dataset:
        run_step("Build final model dataset", [str(PROJECT_PYTHON), "-u", "scripts/build_final_dataset.py"])

    if not args.skip_training:
        run_step("Train model and save diagnostics", [str(PROJECT_PYTHON), "-u", "scripts/train_model.py"])

    run_step("Refresh data manifest", [str(PROJECT_PYTHON), "-u", "scripts/data_status.py", "--write-manifest"])
    run_step("Verify project", [str(PROJECT_PYTHON), "-u", "scripts/verify_project.py"])

    print("\nPipeline finished successfully.")
    print("Open notebooks/04_train_final_model.ipynb and use the final prediction widget for a specific match.")


if __name__ == "__main__":
    main()
