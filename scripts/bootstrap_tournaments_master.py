from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "tournaments_master.xlsx"

LEGACY_COLUMNS = {
    "Турнир": "tournament_name",
    "Возрастная категория": "age_category",
    "Дата начала": "start_date",
    "Город": "city",
    "Ссылка на страницу с матчами": "matches_url",
}

RU_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

MASTER_COLUMNS = [
    "tour_id",
    "tournament_name",
    "age_category",
    "gender",
    "draw_type",
    "federal_district",
    "city",
    "start_date",
    "status",
    "category",
    "system",
    "matches_url",
    "calendar_url",
    "matches_page_saved",
    "source",
    "source_file",
    "created_at",
    "updated_at",
]


def find_legacy_tournaments_file() -> Path:
    candidates = [
        path
        for path in PROJECT_ROOT.glob("*.xlsx")
        if {"Турнир", "Ссылка на страницу с матчами"}.issubset(pd.read_excel(path, nrows=0).columns)
    ]
    if not candidates:
        raise FileNotFoundError("Could not find source Excel with tournament links in the project root.")
    return candidates[0]


def extract_tour_id(url: str) -> str | None:
    match = re.search(r"/tours/(\d+)/", str(url))
    return match.group(1) if match else None


def parse_ru_date(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).normalize()

    text = str(value).strip().lower()
    match = re.search(r"(\d{1,2})\s+([а-яё]+),?\s+(\d{4})", text)
    if not match:
        return pd.to_datetime(value, errors="coerce").normalize()

    day = int(match.group(1))
    month = RU_MONTHS.get(match.group(2))
    year = int(match.group(3))
    if month is None:
        return pd.NaT
    return pd.Timestamp(year=year, month=month, day=day)


def normalize_legacy_frame(df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    missing = [column for column in LEGACY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Source file is missing required columns: {missing}")

    now = datetime.now(timezone.utc).isoformat()
    out = df[list(LEGACY_COLUMNS)].rename(columns=LEGACY_COLUMNS).copy()
    out["tour_id"] = out["matches_url"].map(extract_tour_id)
    out["start_date"] = out["start_date"].map(parse_ru_date)
    out["gender"] = "Женский"
    out["draw_type"] = "Одиночный"
    out["federal_district"] = "Центральный ФО"
    out["status"] = "unknown"
    out["category"] = "Все"
    out["system"] = "Все"
    out["source"] = "legacy_excel"
    out["source_file"] = source_file.name
    out["calendar_url"] = pd.NA
    out["matches_page_saved"] = False
    out["created_at"] = now
    out["updated_at"] = now

    out = out[MASTER_COLUMNS]
    out = out.dropna(subset=["tour_id"]).drop_duplicates(subset=["tour_id"], keep="first")
    return out.sort_values(["start_date", "tour_id"]).reset_index(drop=True)


def merge_master(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return incoming.copy()

    now = datetime.now(timezone.utc).isoformat()
    existing = existing.copy()
    incoming = incoming.copy()
    for column in MASTER_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA
        if column not in incoming.columns:
            incoming[column] = pd.NA

    existing["tour_id"] = existing["tour_id"].map(lambda value: str(int(value)) if pd.notna(value) and isinstance(value, float) and value.is_integer() else str(value))
    incoming["tour_id"] = incoming["tour_id"].map(str)
    existing = existing.drop_duplicates(subset=["tour_id"], keep="first")
    incoming = incoming.drop_duplicates(subset=["tour_id"], keep="first")

    existing_indexed = existing[MASTER_COLUMNS].set_index("tour_id", drop=False)
    incoming_indexed = incoming[MASTER_COLUMNS].set_index("tour_id", drop=False)

    new_ids = incoming_indexed.index.difference(existing_indexed.index)
    changed_ids = incoming_indexed.index.intersection(existing_indexed.index)

    for tour_id in changed_ids:
        current = existing_indexed.loc[tour_id].copy()
        fresh = incoming_indexed.loc[tour_id]
        created_at = current.get("created_at", pd.NA)
        for column in MASTER_COLUMNS:
            if column in {"created_at", "updated_at"}:
                continue
            if pd.notna(fresh[column]):
                current[column] = fresh[column]
        current["created_at"] = created_at if pd.notna(created_at) else now
        current["updated_at"] = now
        existing_indexed.loc[tour_id] = current

    combined = pd.concat([existing_indexed, incoming_indexed.loc[new_ids]], axis=0)
    combined.index.name = None
    combined = combined.drop_duplicates(subset=["tour_id"], keep="first")
    return combined[MASTER_COLUMNS].sort_values(["start_date", "tour_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or update data/tournaments_master.xlsx.")
    parser.add_argument("--source", type=Path, default=None, help="Source Excel with tournament links.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output master Excel path.")
    args = parser.parse_args()

    source = args.source or find_legacy_tournaments_file()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    incoming = normalize_legacy_frame(pd.read_excel(source), source)
    existing = pd.read_excel(output) if output.exists() else None
    master = merge_master(existing, incoming)
    master.to_excel(output, index=False)

    print(f"Source: {source.relative_to(PROJECT_ROOT)}")
    print(f"Output: {output.relative_to(PROJECT_ROOT)}")
    print(f"Incoming unique tournaments: {len(incoming)}")
    print(f"Master tournaments: {len(master)}")
    print(f"Max start_date: {master['start_date'].max().date()}")


if __name__ == "__main__":
    main()
