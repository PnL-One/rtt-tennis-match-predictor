from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
VENV_SITE_PACKAGES = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
STANDALONE_CPYTHON = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Programs"
    / "Python"
    / f"Python{sys.version_info.major}{sys.version_info.minor}"
    / "python.exe"
)

# The current project venv was created from Anaconda Python and has shown
# repeatable Windows access violations in pandas/BeautifulSoup. Prefer the
# official CPython of the same ABI while reusing the venv's installed wheels.
USE_STANDALONE_CPYTHON = STANDALONE_CPYTHON.exists() and VENV_SITE_PACKAGES.exists()
PROJECT_PYTHON = STANDALONE_CPYTHON if USE_STANDALONE_CPYTHON else (VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
PROJECT_PYTHON_PREFIX = [str(PROJECT_PYTHON), "-S"] if USE_STANDALONE_CPYTHON else [str(PROJECT_PYTHON)]
PLAYWRIGHT_BROWSERS_PATH = PROJECT_ROOT / "tmp" / "ms-playwright"


def project_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(PLAYWRIGHT_BROWSERS_PATH))
    if USE_STANDALONE_CPYTHON:
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(VENV_SITE_PACKAGES) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        env["PYTHONNOUSERSITE"] = "1"
    return env


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    print("$ " + " ".join(command), flush=True)

    env = project_subprocess_env()
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
        [*PROJECT_PYTHON_PREFIX, "-c", probe_code],
        cwd=PROJECT_ROOT,
        env=project_subprocess_env(),
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
    parser.add_argument(
        "--match-concurrency",
        type=int,
        default=4,
        help="Number of tournament match pages to render concurrently (default: 4).",
    )
    parser.add_argument("--continue-on-calendar-error", action="store_true", help="Continue with existing tournament master if RTT calendar is temporarily unavailable.")
    parser.add_argument(
        "--model-selection-mode",
        choices=["reuse", "recalibrate"],
        default="reuse",
        help=(
            "reuse: train with latest saved model-selection settings; "
            "recalibrate: rerun CatBoost/GBM/RF time-based grid search."
        ),
    )
    parser.add_argument(
        "--recalibrate-model-selection",
        action="store_true",
        help="Shortcut for --model-selection-mode recalibrate.",
    )
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies and exit.")
    args = parser.parse_args()

    if args.recalibrate_model_selection:
        args.model_selection_mode = "recalibrate"

    print("Full RTT predictor pipeline")
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Project Python: {PROJECT_PYTHON}")
    if USE_STANDALONE_CPYTHON:
        print(f"Project packages: {VENV_SITE_PACKAGES}")
        print("Runtime isolation: official CPython with venv packages")
    print(f"Model selection mode: {args.model_selection_mode}")
    print(f"Match-page concurrency: {max(1, args.match_concurrency)}")

    preflight_check(args)
    if args.check_only:
        print("Preflight check finished successfully.")
        return

    if not args.skip_calendar:
        try:
            run_step("Update tournament ids and details from RTT calendar", [*PROJECT_PYTHON_PREFIX, "-u", "scripts/parse_rtt_calendar.py"])
        except SystemExit:
            if not args.continue_on_calendar_error:
                raise
            print("Warning: RTT calendar update failed; continuing with existing data/tournaments_master.xlsx.", flush=True)

    if not args.skip_matches:
        run_step(
            "Download and parse RTT match pages",
            [
                *PROJECT_PYTHON_PREFIX,
                "-u",
                "scripts/parse_rtt_matches.py",
                "--concurrency",
                str(max(1, args.match_concurrency)),
            ],
        )

    if not args.skip_rankings:
        run_step("Download and parse RTT rankings", [*PROJECT_PYTHON_PREFIX, "-u", "scripts/parse_rtt_rankings.py"])

    if not args.skip_dataset:
        run_step("Build final model dataset", [*PROJECT_PYTHON_PREFIX, "-u", "scripts/build_final_dataset.py"])

    if not args.skip_training:
        run_step(
            "Train model and save diagnostics",
            [
                *PROJECT_PYTHON_PREFIX,
                "-u",
                "scripts/train_model.py",
                "--model-selection-mode",
                args.model_selection_mode,
            ],
        )

    run_step("Refresh data manifest", [*PROJECT_PYTHON_PREFIX, "-u", "scripts/data_status.py", "--write-manifest"])
    run_step("Verify project", [*PROJECT_PYTHON_PREFIX, "-u", "scripts/verify_project.py"])

    print("\nPipeline finished successfully.")
    print("Open notebooks/00_data_control_panel.ipynb to review model quality and use the match prediction dashboard.")


if __name__ == "__main__":
    main()
