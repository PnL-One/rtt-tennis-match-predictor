from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TOURNAMENTS_MASTER_PATH = DATA_DIR / "tournaments_master.xlsx"
MANIFEST_PATH = DATA_DIR / "data_manifest.json"
FINAL_DATASET_PATH = PROJECT_ROOT / "assembled_predictor" / "predictor_model_dataset_from_parsers.xlsx"
RANKINGS_CSV_PATH = PROJECT_ROOT / "rtt_rankings_saved" / "rtt_rankings_all_dates.csv"
SAVED_HTML_DIR = PROJECT_ROOT / "saved_rtt_pages" / "html"


def file_info(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path.relative_to(PROJECT_ROOT))}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def read_tournaments_status() -> dict:
    info = file_info(TOURNAMENTS_MASTER_PATH)
    if not info["exists"]:
        return {"file": info, "rows": 0}

    df = pd.read_excel(TOURNAMENTS_MASTER_PATH)
    status = {
        "file": info,
        "rows": int(len(df)),
        "unique_tour_ids": int(df["tour_id"].nunique()) if "tour_id" in df.columns else 0,
    }
    if "start_date" in df.columns and not df.empty:
        dates = pd.to_datetime(df["start_date"], errors="coerce")
        status["min_start_date"] = str(dates.min().date()) if dates.notna().any() else None
        status["max_start_date"] = str(dates.max().date()) if dates.notna().any() else None
    if "matches_page_saved" in df.columns:
        status["matches_page_saved_count"] = int(df["matches_page_saved"].fillna(False).astype(bool).sum())
    return status


def read_dataset_status() -> dict:
    info = file_info(FINAL_DATASET_PATH)
    if not info["exists"]:
        return {"file": info}

    xl = pd.ExcelFile(FINAL_DATASET_PATH)
    status = {"file": info, "sheets": xl.sheet_names}

    if "coverage" in xl.sheet_names:
        coverage = pd.read_excel(FINAL_DATASET_PATH, sheet_name="coverage")
        status["coverage"] = {
            str(row["metric"]): row["value"]
            for _, row in coverage.iterrows()
            if "metric" in coverage.columns and "value" in coverage.columns
        }

    if "ml_dataset" in xl.sheet_names:
        sample = pd.read_excel(FINAL_DATASET_PATH, sheet_name="ml_dataset", usecols=["match_date"])
        dates = pd.to_datetime(sample["match_date"], errors="coerce")
        status["ml_rows"] = int(len(sample))
        status["max_match_date"] = str(dates.max().date()) if dates.notna().any() else None
        status["min_match_date"] = str(dates.min().date()) if dates.notna().any() else None

    return status


def read_rankings_status() -> dict:
    info = file_info(RANKINGS_CSV_PATH)
    if not info["exists"]:
        return {"file": info}

    df = pd.read_csv(RANKINGS_CSV_PATH, usecols=lambda col: col in {"ranking_date", "age_group_filter", "rni_final"})
    status = {
        "file": info,
        "rows": int(len(df)),
    }
    if "rni_final" in df.columns:
        status["unique_rni"] = int(df["rni_final"].nunique())
    if "ranking_date" in df.columns:
        dates = pd.to_datetime(df["ranking_date"], errors="coerce", dayfirst=True)
        status["min_ranking_date"] = str(dates.min().date()) if dates.notna().any() else None
        status["max_ranking_date"] = str(dates.max().date()) if dates.notna().any() else None
    if {"ranking_date", "age_group_filter"}.issubset(df.columns):
        status["ranking_date_age_group_pairs"] = int(df[["ranking_date", "age_group_filter"]].drop_duplicates().shape[0])
    return status


def read_saved_pages_status() -> dict:
    html_files = sorted(SAVED_HTML_DIR.glob("*.html")) if SAVED_HTML_DIR.exists() else []
    return {
        "html_dir": str(SAVED_HTML_DIR.relative_to(PROJECT_ROOT)),
        "html_pages": len(html_files),
        "latest_html_modified_at": (
            datetime.fromtimestamp(max(path.stat().st_mtime for path in html_files), timezone.utc).isoformat()
            if html_files else None
        ),
    }


def build_manifest() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tournaments": read_tournaments_status(),
        "saved_pages": read_saved_pages_status(),
        "rankings": read_rankings_status(),
        "dataset": read_dataset_status(),
    }


def print_summary(manifest: dict) -> None:
    tournaments = manifest["tournaments"]
    rankings = manifest["rankings"]
    dataset = manifest["dataset"]
    saved_pages = manifest["saved_pages"]

    print("Data status")
    print("===========")
    print(f"Tournaments master: {tournaments.get('rows', 0)} rows, {tournaments.get('unique_tour_ids', 0)} unique tour_id")
    print(f"Tournament period: {tournaments.get('min_start_date')} -> {tournaments.get('max_start_date')}")
    print(f"Saved HTML pages: {saved_pages.get('html_pages', 0)}")
    print(f"Rankings rows: {rankings.get('rows', 0)} | max ranking date: {rankings.get('max_ranking_date')}")
    print(f"Dataset rows: {dataset.get('ml_rows', 0)} | max match date: {dataset.get('max_match_date')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show local RTT predictor data status.")
    parser.add_argument("--json", action="store_true", help="Print full manifest JSON.")
    parser.add_argument("--write-manifest", action="store_true", help="Write data/data_manifest.json.")
    args = parser.parse_args()

    manifest = build_manifest()

    if args.write_manifest:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")

    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print_summary(manifest)


if __name__ == "__main__":
    main()
