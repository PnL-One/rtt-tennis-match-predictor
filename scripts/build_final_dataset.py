#!/usr/bin/env python
# coding: utf-8



# In[1]:


# Если зависимости уже установлены, эту ячейку можно пропустить.
# !pip install -q pandas openpyxl beautifulsoup4 lxml


# In[2]:


from __future__ import annotations

def display(*objects, **kwargs):
    for obj in objects:
        if obj is None:
            continue
        try:
            shape = getattr(obj, "shape", None)
            if shape is not None:
                print(f"[display] {type(obj).__name__} shape={shape}")
            else:
                print(obj)
        except Exception:
            print(repr(obj))


def Markdown(text):
    return text

# # RTT: сборка финального файла для модели из результатов парсинга
# 
# Этот ноутбук собирает финальный Excel-файл для модели из двух результатов парсинга:
# 
# 1. `matches.zip` — сохраненные HTML-страницы матчей турниров.
# 2. `rtt_rankings_all_dates.csv` — история рейтингов RTT по датам и возрастным группам.
# 
# Дополнительно можно подключить старый predictor-файл как источник ручных/ранее проверенных соответствий `ФИО → РНИ`.
# Это полезно для неоднозначных ФИО, когда по фамилии и инициалам в рейтинге найдено несколько кандидатов.
# 
# Главный принцип по рейтингам:
# 
# > Для матча в категории `до 15`, `до 17`, `до 19`, `взрослые` берется рейтинг игрока именно в этой возрастной группе, последний доступный снимок рейтинга **не позже даты матча**.
# 
# То есть очки у игрока могут совпадать между категориями, но `rank_pre` берется из соответствующей возрастной группы.
from pathlib import Path
from collections import defaultdict
from typing import Any, Optional

import json
import re
import zipfile

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


# ## 1. Настройки

# In[3]:


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    raise FileNotFoundError("Could not find project root. Run the notebook from the repository folder or a subfolder.")

PROJECT_ROOT = find_project_root()

MATCHES_HTML_DIR = PROJECT_ROOT / "saved_rtt_pages" / "html"
MATCHES_ZIP_PATH = PROJECT_ROOT / "saved_rtt_pages" / "matches.zip"
MATCHES_SOURCE_PATH = MATCHES_HTML_DIR if MATCHES_HTML_DIR.exists() else MATCHES_ZIP_PATH
RANKINGS_CSV_PATH = PROJECT_ROOT / "rtt_rankings_saved" / "rtt_rankings_all_dates.csv"

# Необязательный файл. Используется только как источник уже проверенного player_matching.
# Если файла нет или USE_EXISTING_PLAYER_MATCHING_AS_OVERRIDE = False, маппинг строится только из рейтингов.
EXISTING_PREDICTOR_PATH = PROJECT_ROOT / "predictor_full_ml_dataset_with_ratings_UPDATED_RNI_RATINGS.xlsx"
USE_EXISTING_PLAYER_MATCHING_AS_OVERRIDE = True

OUTPUT_DIR = PROJECT_ROOT / "assembled_predictor"
OUTPUT_XLSX_PATH = OUTPUT_DIR / "predictor_model_dataset_from_parsers.xlsx"

# Рекомендуемый режим: exact_only.
# Он означает: для матча до 17 лет используется рейтинг до 17 лет, для до 15 — рейтинг до 15 и т.д.
RATING_AGE_GROUP_MODE = "exact_only"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"PROJECT_ROOT: {PROJECT_ROOT}")


# ## 2. Базовые утилиты

# In[4]:


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_lower(value: Any) -> str:
    return normalize_text(value).lower().replace("ё", "е")


def normalize_tournament_name(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
    text = text.replace('Турнир “Турнир ', 'Турнир "').replace('Турнир "Турнир ', 'Турнир "')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_player_key(value: Any) -> str:
    text = normalize_lower(value)
    text = text.replace(".", " ")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def surname_initials_key(full_name: Any) -> str:
    """
    Приводит полное ФИО из рейтинга к ключу вида:
    'сереброва о ю'.

    Такой же ключ получается из турнирного имени 'Сереброва О.Ю.'.
    """
    text = normalize_player_key(full_name)
    parts = text.split()
    if not parts:
        return ""

    surname = parts[0]
    initials = [part[0] for part in parts[1:3] if part]
    return " ".join([surname] + initials)


def parse_ru_date(value: Any) -> pd.Timestamp:
    text = normalize_text(value)
    if not text:
        return pd.NaT

    match = re.search(r"(\d{2}\.\d{2}\.\d{4})", text)
    if match:
        return pd.to_datetime(match.group(1), format="%d.%m.%Y", errors="coerce")

    return pd.to_datetime(text, errors="coerce", dayfirst=True)


def normalize_age_group(value: Any) -> str:
    text = normalize_lower(value)
    text = text.replace("до15", "до 15").replace("до17", "до 17").replace("до19", "до 19")
    text = re.sub(r"\s+", " ", text).strip()

    if "15" in text:
        return "до 15 лет"
    if "17" in text:
        return "до 17 лет"
    if "19" in text:
        return "до 19 лет"
    if "взрос" in text:
        return "взрослые"

    return text


def normalize_rni(value: Any) -> str:
    """
    Нормализует РНИ:
    - 42749 -> '42749'
    - 42749.0 -> '42749'
    - 'RNI:42749' -> '42749'
    """
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    if re.fullmatch(r"\d+\.0", text):
        return text.split(".")[0]

    if re.fullmatch(r"\d+", text):
        return text

    match = re.search(r"\d+", text)
    return match.group(0) if match else ""


def safe_rni_id(value: Any) -> str:
    rni = normalize_rni(value)
    return f"RNI:{rni}" if rni else ""


def extract_tour_id(source_file: str) -> str:
    file_name = Path(source_file).name
    match = re.match(r"\d+_(\d+)_", file_name)
    return match.group(1) if match else ""


# ## 3. Парсинг HTML-страниц матчей из `matches.zip`

# In[5]:


from pathlib import Path

# Локальная защита: если предыдущая utility-ячейка не была выполнена,
# функция as_path все равно будет доступна в этой ячейке.
if "as_path" not in globals():
    def as_path(value):
        text = str(value)
        text = text.replace("\r", "/").replace("\n", "/")
        return Path(text).expanduser()


def extract_lines_from_html(html_text: str) -> list[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    page_text = soup.get_text("\n")
    lines = [normalize_text(line) for line in page_text.split("\n")]
    return [line for line in lines if line and line != "\u200b"]


def value_after_label(lines: list[str], label: str, default: str = "") -> str:
    target = label.rstrip(":").strip().lower()
    for i, line in enumerate(lines):
        current = line.rstrip(":").strip().lower()
        if current == target and i + 1 < len(lines):
            return lines[i + 1]
    return default


def parse_tournament_metadata(lines: list[str], source_file: str) -> dict[str, Any]:
    tournament_name = ""

    for line in lines:
        if line.startswith("Матчи турнира"):
            tournament_name = normalize_text(line.replace("Матчи турнира", ""))
            break

    if not tournament_name:
        for i, line in enumerate(lines):
            if line == "Карточка турнира":
                for j in range(i + 1, min(i + 6, len(lines))):
                    if lines[j] != "---":
                        tournament_name = lines[j]
                        break
                break

    location = value_after_label(lines, "Место проведения")
    city = location.split(",")[0].strip() if location else ""

    start_date = pd.NaT
    for line in lines:
        match = re.search(r"(\d{2}\.\d{2}\.\d{4})\s*-\s*(\d{2}\.\d{2}\.\d{4})", line)
        if match:
            start_date = parse_ru_date(match.group(1))
            break

    age_group = value_after_label(lines, "Возрастная группа")
    if not age_group:
        file_stem = Path(source_file).stem
        for candidate in ["до 15 лет", "до 17 лет", "до 19 лет", "Взрослые", "взрослые"]:
            if candidate.lower() in file_stem.lower():
                age_group = candidate
                break

    return {
        "source_file": source_file,
        "tour_id": extract_tour_id(source_file),
        "tournament_name": normalize_tournament_name(tournament_name),
        "tournament_city": city,
        "tournament_location_full": location,
        "tournament_age_category": normalize_age_group(age_group),
        "tournament_start_date": start_date,
        "tournament_category": value_after_label(lines, "Категория турнира"),
    }


SUMMARY_SCORE_RE = re.compile(r"^(\d+)\s*[-–—:]\s*(\d+)$")


def is_match_number(text: Any) -> bool:
    return bool(re.fullmatch(r"\d{1,4}", normalize_text(text)))


def is_summary_score(text: Any) -> bool:
    return bool(SUMMARY_SCORE_RE.match(normalize_text(text)))


def parse_match_rows_from_lines(lines: list[str], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Парсит блок 'Матчи турнира'.

    Ожидаемый порядок строк в отрендеренной странице:
    №, Участник 1, Участник 2, Счет, Этап турнира,
    затем повторяющиеся блоки:
    номер, player1, player2, score, detailed score, draw_type, Статус, status, Дата, date, Начало, time.
    """
    rows: list[dict[str, Any]] = []

    header_starts = []
    for i, line in enumerate(lines):
        if (
            line == "№"
            and i + 4 < len(lines)
            and "Участник 1" in lines[i + 1]
            and "Участник 2" in lines[i + 2]
            and "Счет" in lines[i + 3]
        ):
            header_starts.append(i + 5)

    if not header_starts:
        return rows

    # На некоторых сохраненных страницах текстовый блок может повторяться ниже.
    # Для модели нам нужен первый фактический блок матчей.
    start = header_starts[0]
    i = start

    while i < len(lines) - 5:
        if not is_match_number(lines[i]):
            if lines[i] in {"2026 ©", "Карточка турнира", "Напишите нам, мы онлайн!"}:
                break
            i += 1
            continue

        match_number = lines[i]

        if i + 5 >= len(lines):
            break

        player1_raw = lines[i + 1]
        player2_raw = lines[i + 2]
        score_raw = lines[i + 3]
        detailed_raw = lines[i + 4]
        draw_type = lines[i + 5]

        if not is_summary_score(score_raw):
            i += 1
            continue

        status = ""
        match_date = pd.NaT
        match_start_time = ""

        j = i + 6

        if j < len(lines) and lines[j].startswith("Статус"):
            status = lines[j + 1] if j + 1 < len(lines) else ""
            j += 2

        if j < len(lines) and lines[j].startswith("Дата"):
            match_date = parse_ru_date(lines[j + 1] if j + 1 < len(lines) else "")
            j += 2

        if j < len(lines) and lines[j].startswith("Начало"):
            match_start_time = lines[j + 1] if j + 1 < len(lines) else ""
            j += 2

        rows.append({
            **metadata,
            "match_number": match_number,
            "player1_raw": player1_raw,
            "player2_raw": player2_raw,
            "score_raw": score_raw,
            "detailed_raw": detailed_raw,
            "draw_type": draw_type,
            "match_date": match_date,
            "match_status": status,
            "match_start_time": match_start_time,
        })

        i = max(j, i + 6)

    return rows


def finalize_parsed_matches(rows: list[dict[str, Any]]) -> pd.DataFrame:
    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["match_date"] = pd.to_datetime(result["match_date"], errors="coerce")
    result["tournament_start_date"] = pd.to_datetime(result["tournament_start_date"], errors="coerce")

    return result


def parse_matches_html_dir(matches_html_dir: Path) -> pd.DataFrame:
    matches_html_dir = as_path(matches_html_dir)
    all_rows: list[dict[str, Any]] = []

    html_paths = sorted(matches_html_dir.glob("*.html"))
    for html_path in html_paths:
        html_text = html_path.read_text(encoding="utf-8", errors="ignore")
        lines = extract_lines_from_html(html_text)
        metadata = parse_tournament_metadata(lines, html_path.name)
        rows = parse_match_rows_from_lines(lines, metadata)
        all_rows.extend(rows)

    return finalize_parsed_matches(all_rows)


def parse_matches_zip(matches_zip_path: Path) -> pd.DataFrame:
    matches_zip_path = as_path(matches_zip_path)
    all_rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(matches_zip_path) as archive:
        html_names = [name for name in archive.namelist() if name.lower().endswith(".html")]

        for name in html_names:
            html_text = archive.read(name).decode("utf-8", errors="ignore")
            lines = extract_lines_from_html(html_text)
            metadata = parse_tournament_metadata(lines, name)
            rows = parse_match_rows_from_lines(lines, metadata)
            all_rows.extend(rows)

    return finalize_parsed_matches(all_rows)


def parse_matches_source(matches_html_dir: Path, matches_zip_path: Path) -> pd.DataFrame:
    matches_html_dir = as_path(matches_html_dir)
    matches_zip_path = as_path(matches_zip_path)

    if matches_html_dir.exists() and any(matches_html_dir.glob("*.html")):
        html_count = len(list(matches_html_dir.glob("*.html")))
        print(f"Parsing saved match HTML directory: {matches_html_dir} ({html_count} files)")
        return parse_matches_html_dir(matches_html_dir)

    print(f"Saved match HTML directory is empty; falling back to zip: {matches_zip_path}")
    return parse_matches_zip(matches_zip_path)


matches_parsed_raw = parse_matches_source(MATCHES_HTML_DIR, MATCHES_ZIP_PATH)

print("Parsed raw match rows:", matches_parsed_raw.shape)
display(matches_parsed_raw.head())


# ## 4. Расчет признаков из счета матча

# In[6]:


SET_RE = re.compile(r"\[?(\d+)\s*[-:]\s*(\d+)\]?(?:\((\d+)\))?")


def parse_score_features(score_raw: Any, detailed_raw: Any) -> dict[str, Any]:
    summary = normalize_text(score_raw)
    detailed = normalize_text(detailed_raw)

    summary_match = SUMMARY_SCORE_RE.match(summary)

    sets1 = np.nan
    sets2 = np.nan
    winner_player1 = np.nan

    if summary_match:
        sets1 = int(summary_match.group(1))
        sets2 = int(summary_match.group(2))
        if sets1 != sets2:
            winner_player1 = int(sets1 > sets2)

    parts = [part.strip() for part in detailed.split(",") if part.strip()]

    total_games_p1 = 0
    total_games_p2 = 0
    total_points_tiebreak = 0
    has_final_tb = 0
    normal_set_count = 0

    result = {
        "summary_score_raw": summary,
        "result_sets": detailed,
        "match_sets_score": f"{int(sets1)}-{int(sets2)}" if pd.notna(sets1) and pd.notna(sets2) else np.nan,
        "match_sets_diff": int(sets1 - sets2) if pd.notna(sets1) and pd.notna(sets2) else np.nan,
        "winner_player1": winner_player1,
        "final_tb_score": np.nan,
    }

    for set_idx in range(1, 4):
        result[f"set{set_idx}_score"] = np.nan
        result[f"set{set_idx}_tb"] = np.nan
        result[f"set{set_idx}_has_tb"] = 0

    for part in parts:
        is_final_tb = part.startswith("[") and part.endswith("]")
        match = SET_RE.search(part)

        if not match:
            continue

        games1 = int(match.group(1))
        games2 = int(match.group(2))
        tb_points = match.group(3)

        if is_final_tb:
            has_final_tb = 1
            result["final_tb_score"] = part
            total_points_tiebreak += games1 + games2
            continue

        normal_set_count += 1
        if normal_set_count <= 3:
            result[f"set{normal_set_count}_score"] = f"{games1}-{games2}"
            result[f"set{normal_set_count}_tb"] = float(tb_points) if tb_points is not None else np.nan
            result[f"set{normal_set_count}_has_tb"] = int(tb_points is not None)

        total_games_p1 += games1
        total_games_p2 += games2

        if tb_points is not None:
            total_points_tiebreak += int(tb_points)

    games_diff = total_games_p1 - total_games_p2 if normal_set_count > 0 else np.nan
    total_games = total_games_p1 + total_games_p2 if normal_set_count > 0 else np.nan

    result.update({
        "has_final_tb": has_final_tb,
        "n_sets": normal_set_count + has_final_tb if parts else np.nan,
        "total_games_p1": float(total_games_p1) if normal_set_count > 0 else np.nan,
        "total_games_p2": float(total_games_p2) if normal_set_count > 0 else np.nan,
        "games_diff": float(games_diff) if pd.notna(games_diff) else np.nan,
        "total_points_tiebreak": float(total_points_tiebreak) if total_points_tiebreak > 0 else np.nan,
        "games_diff_per_set": float(games_diff / normal_set_count) if normal_set_count > 0 else np.nan,
        "dominance_index": float(games_diff / total_games) if normal_set_count > 0 and total_games else np.nan,
    })

    return result


# ## 5. Подготовка истории рейтингов

# In[7]:


from pathlib import Path

# Локальная защита: если предыдущая utility-ячейка не была выполнена,
# функция as_path все равно будет доступна в этой ячейке.
if "as_path" not in globals():
    def as_path(value):
        text = str(value)
        text = text.replace("\r", "/").replace("\n", "/")
        return Path(text).expanduser()


def load_rating_history(rankings_csv_path: Path) -> pd.DataFrame:
    rankings_csv_path = as_path(rankings_csv_path)
    raw = pd.read_csv(rankings_csv_path, dtype=str)

    def numeric_column(*names: str) -> pd.Series:
        for name in names:
            if name in raw.columns:
                return pd.to_numeric(raw[name], errors="coerce")
        return pd.Series(np.nan, index=raw.index)

    def text_column(*names: str) -> pd.Series:
        for name in names:
            if name in raw.columns:
                return raw[name].fillna("").map(normalize_text)
        return pd.Series("", index=raw.index)

    def date_column(*names: str) -> pd.Series:
        for name in names:
            if name in raw.columns:
                return pd.to_datetime(raw[name], errors="coerce", dayfirst=not name.endswith("_dt"))
        return pd.Series(pd.NaT, index=raw.index)

    rating_history = pd.DataFrame({
        "Место": numeric_column("place_num", "place"),
        "ФИО": text_column("fio", "fio_from_link"),
        "Пол": text_column("gender"),
        "РНИ": text_column("rni_final", "rni").map(normalize_rni),
        "Дата рождения": date_column("birth_date_dt", "birth_date"),
        "Город": text_column("city"),
        "Всего турниров": numeric_column("total_tournaments_num", "total_tournaments"),
        "Из них зачетных": numeric_column("counting_tournaments_num", "counting_tournaments"),
        "Возрастная группа": text_column("age_group_filter", "age_group_in_table").map(normalize_age_group),
        "Очки": numeric_column("points_num", "points"),
        "Дата классификации": date_column("ranking_date", "ranking_date_dt"),
    })

    rating_history = rating_history[
        rating_history["РНИ"].ne("")
        & rating_history["Дата классификации"].notna()
        & rating_history["Возрастная группа"].ne("")
    ].copy()

    rating_history["name_key"] = rating_history["ФИО"].map(surname_initials_key)

    rating_history = (
        rating_history
        .sort_values(["РНИ", "Возрастная группа", "Дата классификации", "Место"])
        .drop_duplicates(subset=["РНИ", "Возрастная группа", "Дата классификации"], keep="first")
        .reset_index(drop=True)
    )

    return rating_history


rating_history = load_rating_history(RANKINGS_CSV_PATH)

print("Rating history rows:", rating_history.shape)
display(rating_history.head())
display(
    rating_history
    .groupby(["Дата классификации", "Возрастная группа"], dropna=False)
    .size()
    .reset_index(name="rows")
    .tail(12)
)


# ## 6. Маппинг игроков `ФИО → РНИ`

# In[8]:


from pathlib import Path

# Локальная защита: если предыдущая utility-ячейка не была выполнена,
# функция as_path все равно будет доступна в этой ячейке.
if "as_path" not in globals():
    def as_path(value):
        text = str(value)
        text = text.replace("\r", "/").replace("\n", "/")
        return Path(text).expanduser()


def load_existing_player_matching(path: Path) -> dict[str, str]:
    path = as_path(path)
    if not path.exists():
        return {}

    try:
        existing = pd.read_excel(path, sheet_name="player_matching")
    except Exception as exc:
        print(f"WARNING: не удалось прочитать player_matching из {path}: {exc}")
        return {}

    required = {"raw_name", "RNI"}
    if not required.issubset(existing.columns):
        return {}

    existing = existing.copy()
    existing["name_key"] = existing["raw_name"].map(normalize_player_key)
    existing["RNI_norm"] = existing["RNI"].map(normalize_rni)
    existing = existing[existing["RNI_norm"].ne("")].copy()

    return (
        existing
        .drop_duplicates("name_key", keep="first")
        .set_index("name_key")["RNI_norm"]
        .to_dict()
    )


def build_player_matching(
    matches_raw: pd.DataFrame,
    rating_history: pd.DataFrame,
    existing_predictor_path: Path | None = None,
    use_existing_override: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_names = pd.concat(
        [matches_raw["player1_raw"], matches_raw["player2_raw"]],
        ignore_index=True,
    ).dropna().map(normalize_text)

    player_stats = (
        player_names
        .value_counts()
        .rename_axis("raw_name")
        .reset_index(name="appearances")
    )
    player_stats["name_key"] = player_stats["raw_name"].map(normalize_player_key)

    female_markers = ("девушка", "женщина", "юниорка")
    candidates = rating_history.copy()
    candidates = candidates[
        candidates["Пол"].fillna("").astype(str).str.lower().apply(
            lambda value: any(marker in value for marker in female_markers) or value == ""
        )
    ].copy()

    candidate_summary = (
        candidates
        .groupby("name_key", dropna=False)
        .agg(
            rni_candidates=("РНИ", lambda x: sorted(set(map(str, x)))),
            fio_candidates=("ФИО", lambda x: sorted(set(map(str, x)))),
            gender_candidates=("Пол", lambda x: sorted(set(map(str, x)))),
            birth_date_candidates=(
                "Дата рождения",
                lambda x: sorted(set(str(pd.Timestamp(v).date()) for v in x.dropna())),
            ),
            city_candidates=("Город", lambda x: sorted(set(map(str, x)))),
        )
        .reset_index()
    )

    audit = player_stats.merge(candidate_summary, on="name_key", how="left")
    audit["RNI"] = ""
    audit["match_status"] = "unmatched"
    audit["update_source"] = "rankings_by_surname_initials"
    audit["update_comment"] = ""

    for idx, row in audit.iterrows():
        rni_candidates = row["rni_candidates"] if isinstance(row["rni_candidates"], list) else []

        if len(rni_candidates) == 1:
            audit.at[idx, "RNI"] = rni_candidates[0]
            audit.at[idx, "match_status"] = "matched"
        elif len(rni_candidates) > 1:
            audit.at[idx, "match_status"] = "ambiguous"
            audit.at[idx, "update_comment"] = "several RTT candidates with same surname/initials"

    if use_existing_override and existing_predictor_path is not None:
        existing_map = load_existing_player_matching(existing_predictor_path)
        if existing_map:
            for idx, row in audit.iterrows():
                name_key = row["name_key"]
                if name_key in existing_map:
                    audit.at[idx, "RNI"] = existing_map[name_key]
                    audit.at[idx, "match_status"] = "matched"
                    audit.at[idx, "update_source"] = "existing_player_matching_override"
                    audit.at[idx, "update_comment"] = "filled from existing predictor player_matching"

    player_matching = audit[["raw_name", "name_key", "RNI", "match_status", "appearances"]].copy()
    ambiguous = audit[audit["match_status"].eq("ambiguous")].copy()

    return player_matching, audit, ambiguous


player_matching, player_matching_audit, ambiguous_rni_candidates = build_player_matching(
    matches_raw=matches_parsed_raw,
    rating_history=rating_history,
    existing_predictor_path=EXISTING_PREDICTOR_PATH,
    use_existing_override=USE_EXISTING_PLAYER_MATCHING_AS_OVERRIDE,
)

print("Player matching status:")
display(player_matching["match_status"].value_counts(dropna=False).rename_axis("status").reset_index(name="players"))
display(player_matching.head())


# ## 7. Обогащение матчей РНИ и рейтингами

# In[9]:


def attach_rni_to_matches(matches_raw: pd.DataFrame, player_matching: pd.DataFrame) -> pd.DataFrame:
    out = matches_raw.copy()

    rni_map = player_matching.set_index("raw_name")["RNI"].to_dict()
    status_map = player_matching.set_index("raw_name")["match_status"].to_dict()

    for side in ["player1", "player2"]:
        out[f"{side}_key"] = out[f"{side}_raw"].map(normalize_player_key)
        out[f"{side}_RNI"] = out[f"{side}_raw"].map(rni_map).map(normalize_rni)
        out[f"{side}_match_status"] = out[f"{side}_raw"].map(status_map).fillna("unmatched")
        out[f"{side}_id"] = out[f"{side}_RNI"].map(safe_rni_id)

    return out


def attach_rating_for_side(
    matches_df: pd.DataFrame,
    rating_history: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    out = matches_df.copy()

    key_col = f"{side}_RNI"
    out[key_col] = out[key_col].map(normalize_rni)
    out["tournament_age_category_norm"] = out["tournament_age_category"].map(normalize_age_group)

    result_parts = []

    for age_group, left_part in out.groupby("tournament_age_category_norm", dropna=False):
        left_part = left_part.reset_index().rename(columns={"index": "_orig_idx"}).copy()

        right_part = rating_history[
            rating_history["Возрастная группа"].map(normalize_age_group).eq(age_group)
        ].copy()

        right_part = right_part.rename(columns={"РНИ": key_col})
        right_part[key_col] = right_part[key_col].map(normalize_rni)

        valid_left_mask = left_part[key_col].ne("") & left_part["match_date"].notna()
        left_valid = left_part[valid_left_mask].copy()
        left_invalid = left_part[~valid_left_mask].copy()

        if not left_valid.empty and not right_part.empty:
            merged_valid = pd.merge_asof(
                left_valid.sort_values("match_date"),
                right_part.sort_values("Дата классификации"),
                left_on="match_date",
                right_on="Дата классификации",
                by=key_col,
                direction="backward",
            )
        else:
            merged_valid = left_valid.copy()
            for col in ["Дата классификации", "Место", "Очки", "Возрастная группа", "Всего турниров", "Из них зачетных"]:
                merged_valid[col] = np.nan

        if not left_invalid.empty:
            left_invalid = left_invalid.copy()
            for col in ["Дата классификации", "Место", "Очки", "Возрастная группа", "Всего турниров", "Из них зачетных"]:
                left_invalid[col] = np.nan

        result_parts.append(pd.concat([merged_valid, left_invalid], ignore_index=True))

    result = (
        pd.concat(result_parts, ignore_index=True)
        .set_index("_orig_idx")
        .sort_index()
    )

    out[f"{side}_rating_date_pre"] = result["Дата классификации"].values
    out[f"{side}_rank_pre"] = result["Место"].values
    out[f"{side}_points_pre"] = result["Очки"].values
    out[f"{side}_rating_age_group_pre"] = result["Возрастная группа"].values
    out[f"{side}_rated_tournaments_pre"] = result["Всего турниров"].values
    out[f"{side}_rated_counting_tournaments_pre"] = result["Из них зачетных"].values

    out = out.drop(columns=["tournament_age_category_norm"], errors="ignore")

    return out


def build_matches_enriched(
    matches_raw: pd.DataFrame,
    player_matching: pd.DataFrame,
    rating_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = matches_raw.copy()

    score_features = df.apply(
        lambda row: parse_score_features(row.get("score_raw"), row.get("detailed_raw")),
        axis=1,
        result_type="expand",
    )
    df = pd.concat([df, score_features], axis=1)

    invalid_mask = (
        df["player1_raw"].map(normalize_text).eq("")
        | df["player2_raw"].map(normalize_text).eq("")
        | df["match_date"].isna()
        | df["winner_player1"].isna()
    )

    dropped_invalid_matches = df.loc[invalid_mask].copy()
    df = df.loc[~invalid_mask].copy()

    df = attach_rni_to_matches(df, player_matching)
    df = attach_rating_for_side(df, rating_history, "player1")
    df = attach_rating_for_side(df, rating_history, "player2")

    df["points_diff_pre"] = pd.to_numeric(df["player1_points_pre"], errors="coerce") - pd.to_numeric(df["player2_points_pre"], errors="coerce")
    df["rank_diff_pre"] = pd.to_numeric(df["player1_rank_pre"], errors="coerce") - pd.to_numeric(df["player2_rank_pre"], errors="coerce")

    dedupe_subset = [
        "match_date",
        "player1_raw",
        "player2_raw",
        "result_sets",
        "summary_score_raw",
        "winner_player1",
        "tournament_name",
        "tournament_city",
        "tournament_age_category",
    ]
    dedupe_subset = [col for col in dedupe_subset if col in df.columns]

    duplicate_mask = df.duplicated(subset=dedupe_subset, keep=False)
    dropped_duplicate_matches = df.loc[duplicate_mask].copy()

    df = (
        df
        .drop_duplicates(subset=dedupe_subset, keep="first")
        .reset_index(drop=True)
    )

    return df, dropped_invalid_matches, dropped_duplicate_matches


matches_enriched, dropped_invalid_matches, dropped_duplicate_matches = build_matches_enriched(
    matches_raw=matches_parsed_raw,
    player_matching=player_matching,
    rating_history=rating_history,
)

print("Matches enriched:", matches_enriched.shape)
print("Dropped invalid:", dropped_invalid_matches.shape)
print("Dropped duplicate rows:", dropped_duplicate_matches.shape)

display(matches_enriched.head())


# ## 8. Сборка double-entry `ml_dataset`

# In[10]:


def numeric(value: Any) -> float:
    return pd.to_numeric(value, errors="coerce")


def build_ml_dataset(matches_enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for match in matches_enriched.itertuples(index=False):
        data = match._asdict()

        for side_number in [1, 2]:
            opposite_number = 2 if side_number == 1 else 1
            win = int(data["winner_player1"]) if side_number == 1 else 1 - int(data["winner_player1"])

            player_points = numeric(data.get(f"player{side_number}_points_pre"))
            opponent_points = numeric(data.get(f"player{opposite_number}_points_pre"))
            player_rank = numeric(data.get(f"player{side_number}_rank_pre"))
            opponent_rank = numeric(data.get(f"player{opposite_number}_rank_pre"))

            row = {
                "player1_raw": data.get("player1_raw"),
                "player2_raw": data.get("player2_raw"),
                "result_sets": data.get("result_sets"),
                "summary_score_raw": data.get("summary_score_raw"),
                "match_date": data.get("match_date"),
                "draw_type": data.get("draw_type"),

                "player": data.get(f"player{side_number}_raw"),
                "opponent": data.get(f"player{opposite_number}_raw"),
                "player_RNI": data.get(f"player{side_number}_RNI"),
                "opponent_RNI": data.get(f"player{opposite_number}_RNI"),
                "player_id": data.get(f"player{side_number}_id"),
                "opponent_id": data.get(f"player{opposite_number}_id"),
                "player_match_status": data.get(f"player{side_number}_match_status"),
                "opponent_match_status": data.get(f"player{opposite_number}_match_status"),
                "win": win,

                "player_points_pre": player_points,
                "opponent_points_pre": opponent_points,
                "diff_points_pre": player_points - opponent_points,

                "player_rank_pre": player_rank,
                "opponent_rank_pre": opponent_rank,
                "diff_rank_pre": player_rank - opponent_rank,

                "player_rating_date_pre": data.get(f"player{side_number}_rating_date_pre"),
                "opponent_rating_date_pre": data.get(f"player{opposite_number}_rating_date_pre"),
                "player_rating_age_group_pre": data.get(f"player{side_number}_rating_age_group_pre"),
                "opponent_rating_age_group_pre": data.get(f"player{opposite_number}_rating_age_group_pre"),

                "player_rated_tournaments_pre": data.get(f"player{side_number}_rated_tournaments_pre"),
                "opponent_rated_tournaments_pre": data.get(f"player{opposite_number}_rated_tournaments_pre"),
                "player_rated_counting_tournaments_pre": data.get(f"player{side_number}_rated_counting_tournaments_pre"),
                "opponent_rated_counting_tournaments_pre": data.get(f"player{opposite_number}_rated_counting_tournaments_pre"),
            }

            match_feature_cols = [
                "match_sets_score",
                "match_sets_diff",
                "set1_score",
                "set1_tb",
                "set1_has_tb",
                "set2_score",
                "set2_tb",
                "set2_has_tb",
                "set3_score",
                "set3_tb",
                "set3_has_tb",
                "final_tb_score",
                "has_final_tb",
                "n_sets",
                "total_games_p1",
                "total_games_p2",
                "games_diff",
                "total_points_tiebreak",
                "games_diff_per_set",
                "dominance_index",
                "winner_player1",
                "tournament_name",
                "tournament_city",
                "tournament_age_category",
                "tournament_start_date",
            ]

            for col in match_feature_cols:
                row[col] = data.get(col)

            side_specific_cols = [
                col for col in matches_enriched.columns
                if col.startswith("player1_") or col.startswith("player2_") or col in {"points_diff_pre", "rank_diff_pre"}
            ]

            for col in side_specific_cols:
                row[col] = data.get(col)

            rows.append(row)

    ml_dataset = pd.DataFrame(rows)

    for side in ["player", "opponent"]:
        points = pd.to_numeric(ml_dataset[f"{side}_points_pre"], errors="coerce")
        ml_dataset[f"{side}_points_missing_pre"] = points.isna().astype(int)
        ml_dataset[f"{side}_log_points_pre"] = np.log1p(points)

    ml_dataset["diff_log_points_pre"] = ml_dataset["player_log_points_pre"] - ml_dataset["opponent_log_points_pre"]

    return ml_dataset


ml_dataset = build_ml_dataset(matches_enriched)

print("ML dataset:", ml_dataset.shape)
display(ml_dataset.head())


# ## 9. Контроль качества и audit-таблицы

# In[11]:


def build_coverage_table(
    matches_enriched: pd.DataFrame,
    ml_dataset: pd.DataFrame,
    player_matching: pd.DataFrame,
    rating_history: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"metric": "completed_matches", "value": len(matches_enriched)},
        {"metric": "double_entry_rows", "value": len(ml_dataset)},
        {"metric": "unique_players_in_matches", "value": int(player_matching["raw_name"].nunique())},
        {"metric": "matched_players", "value": int(player_matching["match_status"].eq("matched").sum())},
        {"metric": "ambiguous_players", "value": int(player_matching["match_status"].eq("ambiguous").sum())},
        {"metric": "unmatched_players", "value": int(player_matching["match_status"].eq("unmatched").sum())},
        {"metric": "rating_history_rows", "value": len(rating_history)},
        {"metric": "rating_history_unique_rni", "value": int(rating_history["РНИ"].nunique())},
        {"metric": "player1_rni_match_rate", "value": float(matches_enriched["player1_RNI"].map(normalize_rni).ne("").mean())},
        {"metric": "player2_rni_match_rate", "value": float(matches_enriched["player2_RNI"].map(normalize_rni).ne("").mean())},
        {"metric": "player1_rating_coverage", "value": float(matches_enriched["player1_points_pre"].notna().mean())},
        {"metric": "player2_rating_coverage", "value": float(matches_enriched["player2_points_pre"].notna().mean())},
    ]
    return pd.DataFrame(rows)


def build_players_without_rating(matches_enriched: pd.DataFrame) -> pd.DataFrame:
    left = matches_enriched[
        matches_enriched["player1_points_pre"].isna()
        | matches_enriched["player2_points_pre"].isna()
    ].copy()

    cols = [
        "match_date",
        "tournament_name",
        "tournament_age_category",
        "player1_raw",
        "player1_RNI",
        "player1_match_status",
        "player1_rating_age_group_pre",
        "player1_points_pre",
        "player2_raw",
        "player2_RNI",
        "player2_match_status",
        "player2_rating_age_group_pre",
        "player2_points_pre",
    ]
    cols = [col for col in cols if col in left.columns]
    return left[cols].drop_duplicates().reset_index(drop=True)


def build_rating_age_group_check(matches_enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for side in ["player1", "player2"]:
        temp = matches_enriched[[
            "match_date",
            "tournament_age_category",
            f"{side}_raw",
            f"{side}_RNI",
            f"{side}_rating_age_group_pre",
            f"{side}_points_pre",
            f"{side}_rank_pre",
        ]].copy()

        temp["side"] = side
        temp["tournament_age_norm"] = temp["tournament_age_category"].map(normalize_age_group)
        temp["rating_age_norm"] = temp[f"{side}_rating_age_group_pre"].map(normalize_age_group)
        temp["age_group_match"] = (
            temp["rating_age_norm"].eq(temp["tournament_age_norm"])
            | temp[f"{side}_points_pre"].isna()
        )
        rows.append(temp)

    return pd.concat(rows, ignore_index=True)


coverage = build_coverage_table(matches_enriched, ml_dataset, player_matching, rating_history)
players_without_rni = player_matching[player_matching["match_status"].ne("matched")].copy()
players_without_rating = build_players_without_rating(matches_enriched)
rating_age_group_check = build_rating_age_group_check(matches_enriched)

display(coverage)
display(rating_age_group_check["age_group_match"].value_counts(dropna=False).rename_axis("age_group_match").reset_index(name="rows"))


# ## 10. Сохранение финального Excel-файла

# In[12]:


from pathlib import Path

# Локальная защита: если предыдущая utility-ячейка не была выполнена,
# функция as_path все равно будет доступна в этой ячейке.
if "as_path" not in globals():
    def as_path(value):
        text = str(value)
        text = text.replace("\r", "/").replace("\n", "/")
        return Path(text).expanduser()


def write_final_predictor_file(
    output_path: Path,
    ml_dataset: pd.DataFrame,
    matches_enriched: pd.DataFrame,
    player_matching: pd.DataFrame,
    coverage: pd.DataFrame,
    rating_history: pd.DataFrame,
    matches_parsed_raw: pd.DataFrame,
    players_without_rni: pd.DataFrame,
    players_without_rating: pd.DataFrame,
    dropped_invalid_matches: pd.DataFrame,
    dropped_duplicate_matches: pd.DataFrame,
    player_matching_audit: pd.DataFrame,
    ambiguous_rni_candidates: pd.DataFrame,
    rating_age_group_check: pd.DataFrame,
) -> None:
    output_path = as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    build_summary = pd.DataFrame([
        {"section": "input_files", "metric": "matches_source", "value": str(MATCHES_SOURCE_PATH)},
        {"section": "input_files", "metric": "matches_zip_fallback", "value": str(MATCHES_ZIP_PATH)},
        {"section": "input_files", "metric": "rankings_csv_source", "value": str(RANKINGS_CSV_PATH)},
        {"section": "input_files", "metric": "existing_predictor_override", "value": str(EXISTING_PREDICTOR_PATH) if USE_EXISTING_PLAYER_MATCHING_AS_OVERRIDE else "not_used"},
        {"section": "output", "metric": "output_xlsx", "value": str(output_path)},
        {"section": "settings", "metric": "rating_age_group_mode", "value": RATING_AGE_GROUP_MODE},
        {"section": "result", "metric": "matches_enriched_rows", "value": len(matches_enriched)},
        {"section": "result", "metric": "ml_dataset_rows", "value": len(ml_dataset)},
        {"section": "result", "metric": "rating_history_rows", "value": len(rating_history)},
    ])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        ml_dataset.to_excel(writer, sheet_name="ml_dataset", index=False)
        matches_enriched.to_excel(writer, sheet_name="matches_enriched", index=False)
        player_matching.to_excel(writer, sheet_name="player_matching", index=False)
        coverage.to_excel(writer, sheet_name="coverage", index=False)
        rating_history.drop(columns=["name_key"], errors="ignore").to_excel(writer, sheet_name="rating_history", index=False)
        matches_parsed_raw.to_excel(writer, sheet_name="matches_parsed_raw", index=False)

        players_without_rni.to_excel(writer, sheet_name="players_without_rni", index=False)
        players_without_rating.to_excel(writer, sheet_name="players_without_rating", index=False)
        dropped_invalid_matches.to_excel(writer, sheet_name="dropped_invalid_matches", index=False)
        dropped_duplicate_matches.to_excel(writer, sheet_name="dropped_duplicate_matches", index=False)

        player_matching_audit.to_excel(writer, sheet_name="player_matching_audit", index=False)
        ambiguous_rni_candidates.to_excel(writer, sheet_name="ambiguous_rni_candidates", index=False)
        rating_age_group_check.to_excel(writer, sheet_name="rating_age_group_check", index=False)
        build_summary.to_excel(writer, sheet_name="build_summary", index=False)

    print("Финальный файл сохранен:")
    print(output_path)


write_final_predictor_file(
    output_path=OUTPUT_XLSX_PATH,
    ml_dataset=ml_dataset,
    matches_enriched=matches_enriched,
    player_matching=player_matching,
    coverage=coverage,
    rating_history=rating_history,
    matches_parsed_raw=matches_parsed_raw,
    players_without_rni=players_without_rni,
    players_without_rating=players_without_rating,
    dropped_invalid_matches=dropped_invalid_matches,
    dropped_duplicate_matches=dropped_duplicate_matches,
    player_matching_audit=player_matching_audit,
    ambiguous_rni_candidates=ambiguous_rni_candidates,
    rating_age_group_check=rating_age_group_check,
)

