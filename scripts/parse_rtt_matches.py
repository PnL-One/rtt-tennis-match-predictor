#!/usr/bin/env python
# coding: utf-8

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

# # RTT: сохранение страниц турниров и попытка извлечения матчей
# 
# Этот ноутбук делает две вещи:
# 
# 1. Открывает ссылки из Excel через **Firefox + Playwright**, ждет рендер JavaScript и сохраняет:
#    - готовый HTML,
#    - скриншот страницы.
# 2. Пытается автоматически вытащить из сохраненных HTML данные о матчах и сохранить их в Excel.
# 
# ## Ожидаемый входной файл
# Excel с колонками:
# - `Турнир`
# - `Возрастная категория`
# - `Дата начала`
# - `Город`
# - `Ссылка на страницу с матчами`
# 
# ## Что получится на выходе
# - папка `saved_rtt_pages/`
# - файл `rtt_matches_extracted.xlsx`
# - файл `rtt_matches_extracted.csv`
# 
# > Важно: структура RTT может меняться. Поэтому в ноутбуке сделаны несколько стратегий парсинга:
# > - через HTML-таблицы,
# > - через JSON внутри `<script>`,
# > - через поиски типовых блоков с игроками и счетом.
# 


# In[1]:


# Если нужно установить зависимости, раскомментируй и запусти эту ячейку
# !playwright install firefox


# In[2]:


from datetime import date
from pathlib import Path
import os
import re
import asyncio
import json
from typing import Any, Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


# In[3]:


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    raise FileNotFoundError("Could not find project root. Run the notebook from the repository folder or a subfolder.")

PROJECT_ROOT = find_project_root()

# =========================
# Конфигурация
# =========================

TOURNAMENTS_MASTER_PATH = PROJECT_ROOT / "data" / "tournaments_master.xlsx"
if TOURNAMENTS_MASTER_PATH.exists():
    INPUT_EXCEL_PATH = TOURNAMENTS_MASTER_PATH
    LINK_COLUMN_NAME = "matches_url"
else:
    INPUT_EXCEL_CANDIDATES = sorted(PROJECT_ROOT.glob("*.xlsx"))
    if not INPUT_EXCEL_CANDIDATES:
        raise FileNotFoundError("Could not find tournaments input Excel file.")
    INPUT_EXCEL_PATH = INPUT_EXCEL_CANDIDATES[0]
    LINK_COLUMN_NAME = "Ссылка на страницу с матчами"

OUTPUT_DIR = PROJECT_ROOT / "saved_rtt_pages"
HTML_DIR = OUTPUT_DIR / "html"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"

HEADLESS = True
PAGE_TIMEOUT_MS = 60000
WAIT_AFTER_OPEN_SECONDS = 4
SCROLL_PAUSE_SECONDS = 1.0
MAX_SCROLL_ROUNDS = 15

RESULT_EXCEL_PATH = PROJECT_ROOT / "rtt_matches_extracted.xlsx"
RESULT_CSV_PATH = PROJECT_ROOT / "rtt_matches_extracted.csv"
FAILED_LINKS_PATH = PROJECT_ROOT / "rtt_failed_links.xlsx"
SAVE_LOG_PATH = OUTPUT_DIR / "rtt_match_page_save_log.xlsx"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT: {PROJECT_ROOT}")


# In[4]:


# Проверка входного файла

source_df = pd.read_excel(INPUT_EXCEL_PATH)
print(source_df.shape)
display(source_df.head(3))

master_columns = ["tour_id", "tournament_name", "age_category", "start_date", "city", "matches_url"]
legacy_columns = [
    "Турнир",
    "Возрастная категория",
    "Дата начала",
    "Город",
    "Ссылка на страницу с матчами",
]

if all(col in source_df.columns for col in master_columns):
    print("Input format: tournaments_master")
elif all(col in source_df.columns for col in legacy_columns):
    print("Input format: legacy Excel")
else:
    raise ValueError(
        "Input file must contain either tournaments_master columns "
        f"{master_columns} or legacy columns {legacy_columns}."
    )


# In[5]:


def sanitize_file_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[<>:\"/\\|?*]+", "_", text)
    text = re.sub(r"\s+", " ", text)
    return text[:180].strip(" ._")

def extract_tour_id(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    match = re.search(r"/tours/(\d+)/", url)
    return match.group(1) if match else None

def build_output_file_base(row_index: int, tournament_name: str, age_category: str, url: str) -> str:
    tour_id = extract_tour_id(url) or f"row_{row_index + 1}"
    tournament_name = sanitize_file_name(tournament_name)
    age_category = sanitize_file_name(age_category)
    return f"{row_index + 1:04d}_{tour_id}_{tournament_name}_{age_category}"

def normalize_status(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower().replace("ё", "е")

def is_cancelled_status(value: Any) -> bool:
    status = normalize_status(value)
    return "отмен" in status or "аннулир" in status

def is_completed_status(value: Any) -> bool:
    status = normalize_status(value)
    return "заверш" in status

def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "да"}

def parse_start_date(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", dayfirst=True)

def resolve_existing_html_path(expected_path: Path, tour_id: str | None) -> Path | None:
    if expected_path.exists():
        return expected_path
    if tour_id:
        matches = sorted(HTML_DIR.glob(f"*_{tour_id}_*.html"))
        if matches:
            return matches[0]
    return None

def build_rows_to_process(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = df[df[LINK_COLUMN_NAME].notna()].copy()
    rows = rows[rows[LINK_COLUMN_NAME].astype(str).str.strip() != ""]
    rows = rows.reset_index(drop=False)

    today = pd.Timestamp(date.today())
    rows["_start_date_dt"] = rows.get("start_date", pd.Series(pd.NaT, index=rows.index)).map(parse_start_date)
    rows["_is_future"] = rows["_start_date_dt"].notna() & rows["_start_date_dt"].gt(today)
    rows["_is_cancelled"] = rows.get("status", pd.Series("", index=rows.index)).map(is_cancelled_status)
    rows["_is_completed"] = rows.get("status", pd.Series("", index=rows.index)).map(is_completed_status)
    rows["_matches_page_saved_bool"] = rows.get("matches_page_saved", pd.Series(False, index=rows.index)).map(bool_value)

    skipped_rows: List[Dict[str, Any]] = []
    process_mask = []

    for _, row in rows.iterrows():
        source_index = int(row["index"])
        tournament_name = str(row.get("tournament_name", row.get("Турнир", ""))).strip()
        age_category = str(row.get("age_category", row.get("Возрастная категория", ""))).strip()
        url = str(row[LINK_COLUMN_NAME]).strip()
        tour_id = extract_tour_id(url)
        file_base = build_output_file_base(source_index, tournament_name, age_category, url)
        expected_html_path = HTML_DIR / f"{file_base}.html"
        cached_html_path = resolve_existing_html_path(expected_html_path, tour_id)

        skip_reason = ""
        save_status = ""
        use_cached = False

        if row["_is_future"]:
            skip_reason = "future_start_date"
            save_status = "skipped_future"
        elif row["_is_cancelled"]:
            skip_reason = "cancelled"
            save_status = "skipped_cancelled"
        elif cached_html_path is not None:
            skip_reason = "cached_html"
            save_status = "ok"
            use_cached = True

        if skip_reason:
            skipped_rows.append({
                "source_index": source_index,
                "Турнир": tournament_name,
                "Возрастная категория": age_category,
                "Дата начала": str(row.get("start_date", "")).strip(),
                "Город": str(row.get("city", row.get("Город", ""))).strip(),
                "Ссылка на страницу с матчами": url,
                "tour_id": tour_id,
                "html_path": str(cached_html_path or expected_html_path),
                "screenshot_path": str(SCREENSHOT_DIR / f"{file_base}.png"),
                "save_status": save_status,
                "save_error": "",
                "skip_reason": skip_reason,
                "used_cached_html": use_cached,
            })
            process_mask.append(False)
        else:
            process_mask.append(True)

    rows_to_process = rows[process_mask].copy().reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped_rows)
    print(
        "Tournament download plan: "
        f"{len(rows)} eligible links, {len(rows_to_process)} to download, {len(skipped_df)} skipped/cached"
    )
    if not skipped_df.empty:
        print(skipped_df["skip_reason"].value_counts(dropna=False).to_string())
    return rows_to_process, skipped_df

async def auto_scroll_page(page) -> None:
    previous_height = 0

    for _ in range(MAX_SCROLL_ROUNDS):
        current_height = await page.evaluate("document.body.scrollHeight")
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(int(SCROLL_PAUSE_SECONDS * 1000))

        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == previous_height or new_height == current_height:
            break

        previous_height = new_height

async def click_expand_buttons_if_any(page, max_clicks: int = 8) -> None:
    possible_texts = [
        "Показать еще",
        "Показать ещё",
        "Еще",
        "Ещё",
        "Show more",
        "Load more",
        "Подробнее",
    ]

    clicks = 0
    for button_text in possible_texts:
        try:
            while True:
                if clicks >= max_clicks:
                    return
                locator = page.get_by_text(button_text, exact=False)
                count = await locator.count()
                if count == 0:
                    break

                first_button = locator.first
                if not await first_button.is_visible():
                    break

                await first_button.click(timeout=3000)
                clicks += 1
                await page.wait_for_timeout(1000)
        except Exception:
            pass


# ## 1. Сохранение отрендеренных страниц RTT

# In[6]:


async def save_rendered_pages_from_excel(input_excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(input_excel_path)

    rows_to_process, skipped_df = build_rows_to_process(df)

    save_log_rows: List[Dict[str, Any]] = []

    async with async_playwright() as playwright:
        browser = await playwright.firefox.launch(headless=HEADLESS)
        context = await browser.new_context(
            viewport={"width": 1600, "height": 2400},
            locale="ru-RU",
        )

        try:
            for i, row in rows_to_process.iterrows():
                source_index = int(row["index"])
                tournament_name = str(row.get("tournament_name", row.get("Турнир", ""))).strip()
                age_category = str(row.get("age_category", row.get("Возрастная категория", ""))).strip()
                start_date = str(row.get("start_date", row.get("Дата начала", ""))).strip()
                city = str(row.get("city", row.get("Город", ""))).strip()
                url = str(row[LINK_COLUMN_NAME]).strip()

                file_base = build_output_file_base(source_index, tournament_name, age_category, url)
                html_path = HTML_DIR / f"{file_base}.html"
                screenshot_path = SCREENSHOT_DIR / f"{file_base}.png"

                print(f"[{i + 1}/{len(rows_to_process)}] opening {url}", flush=True)

                page = await context.new_page()
                status = "ok"
                error_text = ""

                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)

                    try:
                        print(f"[{i + 1}/{len(rows_to_process)}] waiting for rendered content", flush=True)
                        await page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:
                        pass

                    await page.wait_for_timeout(int(WAIT_AFTER_OPEN_SECONDS * 1000))
                    print(f"[{i + 1}/{len(rows_to_process)}] expanding and scrolling", flush=True)
                    await click_expand_buttons_if_any(page)
                    await auto_scroll_page(page)
                    await click_expand_buttons_if_any(page)
                    await auto_scroll_page(page)

                    print(f"[{i + 1}/{len(rows_to_process)}] saving html", flush=True)
                    html_content = await page.content()
                    html_path.write_text(html_content, encoding="utf-8")

                    try:
                        print(f"[{i + 1}/{len(rows_to_process)}] saving screenshot", flush=True)
                        await page.screenshot(path=str(screenshot_path), full_page=True)
                    except Exception:
                        pass

                except PlaywrightTimeoutError:
                    status = "timeout"
                    error_text = "PlaywrightTimeoutError"
                except Exception as exc:
                    status = "error"
                    error_text = str(exc)
                finally:
                    await page.close()

                save_log_rows.append({
                    "source_index": source_index,
                    "Турнир": tournament_name,
                    "Возрастная категория": age_category,
                    "Дата начала": start_date,
                    "Город": city,
                    "Ссылка на страницу с матчами": url,
                    "tour_id": extract_tour_id(url),
                    "html_path": str(html_path),
                    "screenshot_path": str(screenshot_path),
                    "save_status": status,
                    "save_error": error_text,
                    "skip_reason": "",
                    "used_cached_html": False,
                })

        finally:
            await context.close()
            await browser.close()

    save_log_df = pd.DataFrame(save_log_rows)
    if not skipped_df.empty:
        save_log_df = pd.concat([save_log_df, skipped_df], ignore_index=True)
    save_log_df.to_excel(SAVE_LOG_PATH, index=False)

    if TOURNAMENTS_MASTER_PATH.exists() and "tour_id" in df.columns and "matches_page_saved" in df.columns:
        ok_tour_ids = set(save_log_df.loc[save_log_df["save_status"].eq("ok"), "tour_id"].dropna().astype(str))
        if ok_tour_ids:
            updated_master = df.copy()
            updated_master["tour_id"] = updated_master["tour_id"].astype(str)
            updated_master.loc[updated_master["tour_id"].isin(ok_tour_ids), "matches_page_saved"] = True
            updated_master.to_excel(TOURNAMENTS_MASTER_PATH, index=False)
            print(f"Updated matches_page_saved for {len(ok_tour_ids)} tournaments in {TOURNAMENTS_MASTER_PATH}")

    return save_log_df


# **Важно:** если Firefox/Playwright ещё не установлен, сначала запусти ячейку установки вверху (`!playwright install firefox`), а затем перезапусти kernel.

# In[7]:


# Запуск сохранения страниц
save_log_df = asyncio.run(save_rendered_pages_from_excel(INPUT_EXCEL_PATH))

display(save_log_df.head(3))
print(save_log_df["save_status"].value_counts(dropna=False))
print(f"Лог сохранён в: {SAVE_LOG_PATH}")


# ## 2. Парсинг матчей из сохраненных HTML
# 
# Здесь несколько попыток извлечения:
# 1. через HTML-таблицы (`pandas.read_html`);
# 2. через JSON внутри `<script>`;
# 3. через поиск текстовых блоков, похожих на карточки матчей.
# 

# In[8]:


def normalize_space(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def looks_like_score(text: str) -> bool:
    text = normalize_space(text)
    if not text:
        return False

    patterns = [
        r"^\d+:\d+$",
        r"^\d+-\d+$",
        r"^\d+:\d+\s+\d+:\d+$",
        r"^\d+-\d+\s+\d+-\d+$",
        r"^(\d+:\d+|\d+-\d+)(\s+(\d+:\d+|\d+-\d+)){1,4}$",
    ]
    return any(re.match(pattern, text) for pattern in patterns)

def is_player_like(text: str) -> bool:
    text = normalize_space(text)
    if len(text) < 3:
        return False

    # Фамилия И.О. / Имя Фамилия / Фамилия Имя
    if re.search(r"[А-ЯA-Z][а-яa-z\-']+\s+[А-ЯA-Z]\.[А-ЯA-Z]\.", text):
        return True
    if len(text.split()) in {2, 3} and re.search(r"[А-ЯA-Zа-яa-z]", text):
        return True
    return False

def flatten_json(obj: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: List[tuple] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            items.extend(flatten_json(value, new_key, sep=sep).items())
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            new_key = f"{parent_key}{sep}{idx}" if parent_key else str(idx)
            items.extend(flatten_json(value, new_key, sep=sep).items())
    else:
        items.append((parent_key, obj))

    return dict(items)


# In[9]:


def extract_tables_from_html(html_text: str) -> List[pd.DataFrame]:
    tables: List[pd.DataFrame] = []
    try:
        candidate_tables = pd.read_html(html_text)
    except Exception:
        return tables

    for table in candidate_tables:
        if table is None or table.empty:
            continue

        temp = table.copy()
        temp.columns = [normalize_space(str(col)) for col in temp.columns]
        for col in temp.columns:
            temp[col] = temp[col].astype(str).map(normalize_space)

        tables.append(temp)

    return tables

def table_to_match_rows(
    table: pd.DataFrame,
    source_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if table.empty:
        return rows

    joined_text = " | ".join(table.astype(str).fillna("").head(10).stack().tolist()).lower()
    useful_keywords = ["match", "игрок", "счет", "счёт", "player", "score", "winner", "побед"]
    if not any(keyword in joined_text for keyword in useful_keywords):
        # Но если таблица узкая и содержит имена/счет, тоже считаем
        pass

    for row_idx in range(len(table)):
        row = table.iloc[row_idx].astype(str).to_dict()
        values = [normalize_space(v) for v in row.values() if normalize_space(v)]

        player_candidates = [v for v in values if is_player_like(v)]
        score_candidates = [v for v in values if looks_like_score(v)]

        if len(player_candidates) >= 2 or score_candidates:
            rows.append({
                **source_meta,
                "source_parser": "html_table",
                "table_columns": " | ".join(table.columns.astype(str)),
                "table_row_index": row_idx,
                "player1": player_candidates[0] if len(player_candidates) >= 1 else "",
                "player2": player_candidates[1] if len(player_candidates) >= 2 else "",
                "score": score_candidates[0] if score_candidates else "",
                "round": "",
                "status": "",
                "winner": "",
                "raw_row": json.dumps(row, ensure_ascii=False),
            })

    return rows


# In[10]:


def extract_json_objects_from_scripts(html_text: str) -> List[Any]:
    soup = BeautifulSoup(html_text, "html.parser")
    json_objects: List[Any] = []

    for script in soup.find_all("script"):
        script_text = script.string or script.get_text(" ", strip=False) or ""
        script_text = script_text.strip()
        if not script_text:
            continue

        # Попытка 1: application/ld+json или просто чистый JSON
        if script.get("type") == "application/ld+json":
            try:
                json_objects.append(json.loads(script_text))
            except Exception:
                pass

        # Попытка 2: вытащить большие JSON-фрагменты по шаблонам
        candidate_patterns = [
            r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;",
            r"window\.__NUXT__\s*=\s*(\{.*?\})\s*;",
            r"window\.__NEXT_DATA__\s*=\s*(\{.*?\})\s*;",
            r"__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;",
            r"__data\s*=\s*(\{.*?\})\s*;",
        ]

        for pattern in candidate_patterns:
            matches = re.findall(pattern, script_text, flags=re.DOTALL)
            for match_text in matches:
                try:
                    json_objects.append(json.loads(match_text))
                except Exception:
                    pass

    return json_objects

def json_object_to_match_rows(obj: Any, source_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Стратегия: смотрим на уплощенный JSON и пытаемся собрать сущности с players/score/round
    if isinstance(obj, (dict, list)):
        flat = flatten_json(obj)
    else:
        return rows

    flat_items = list(flat.items())
    flat_text = " ".join(f"{k}={v}" for k, v in flat_items[:3000]).lower()

    if not any(keyword in flat_text for keyword in ["score", "player", "match", "round", "winner", "participant"]):
        return rows

    # Попытка выделить записи по индексам вроде matches.0.player1.name
    grouped: Dict[str, Dict[str, Any]] = {}

    for key, value in flat_items:
        key_norm = key.lower()
        parts = key.split(".")
        group_key = None

        for i, part in enumerate(parts):
            if part.isdigit():
                group_key = ".".join(parts[: i + 1])
                break

        if group_key is None:
            continue

        grouped.setdefault(group_key, {})
        grouped[group_key][key_norm] = value

    for group_key, item in grouped.items():
        text_blob = " | ".join([f"{k}={v}" for k, v in item.items()]).lower()
        if not any(keyword in text_blob for keyword in ["player", "score", "match", "winner", "participant"]):
            continue

        player_values = []
        score_values = []
        round_values = []
        status_values = []
        winner_values = []

        for key_norm, value in item.items():
            value_text = normalize_space(value)

            if any(x in key_norm for x in ["player", "participant", "competitor", "name"]):
                if is_player_like(value_text):
                    player_values.append(value_text)

            if "score" in key_norm and looks_like_score(value_text):
                score_values.append(value_text)

            if "round" in key_norm or "stage" in key_norm:
                if value_text:
                    round_values.append(value_text)

            if "status" in key_norm:
                if value_text:
                    status_values.append(value_text)

            if "winner" in key_norm:
                if value_text:
                    winner_values.append(value_text)

        if len(player_values) >= 2 or score_values:
            rows.append({
                **source_meta,
                "source_parser": "script_json",
                "table_columns": "",
                "table_row_index": "",
                "player1": player_values[0] if len(player_values) >= 1 else "",
                "player2": player_values[1] if len(player_values) >= 2 else "",
                "score": score_values[0] if score_values else "",
                "round": round_values[0] if round_values else "",
                "status": status_values[0] if status_values else "",
                "winner": winner_values[0] if winner_values else "",
                "raw_row": json.dumps(item, ensure_ascii=False),
            })

    return rows


# In[11]:


def extract_dom_card_rows(html_text: str, source_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_text, "html.parser")
    rows: List[Dict[str, Any]] = []

    # Берем блоки div/li/tr, в которых встречаются имена игроков/счет
    candidates = soup.find_all(["div", "li", "tr", "section", "article"])

    for block in candidates:
        text = normalize_space(block.get_text(" ", strip=True))
        if not text:
            continue

        if len(text) < 10 or len(text) > 700:
            continue

        score_match = re.search(r"(\d+:\d+(?:\s+\d+:\d+){0,4}|\d+-\d+(?:\s+\d+-\d+){0,4})", text)
        player_like_parts = re.findall(r"[А-ЯA-Z][а-яa-z\-']+\s+[А-ЯA-Z]\.[А-ЯA-Z]\.|[А-ЯA-Z][а-яa-z\-']+\s+[А-ЯA-Z][а-яa-z\-']+", text)

        if len(player_like_parts) >= 2 or score_match:
            rows.append({
                **source_meta,
                "source_parser": "dom_cards",
                "table_columns": "",
                "table_row_index": "",
                "player1": normalize_space(player_like_parts[0]) if len(player_like_parts) >= 1 else "",
                "player2": normalize_space(player_like_parts[1]) if len(player_like_parts) >= 2 else "",
                "score": normalize_space(score_match.group(1)) if score_match else "",
                "round": "",
                "status": "",
                "winner": "",
                "raw_row": text,
            })

    return rows


# In[12]:


def deduplicate_match_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    temp = df.copy()
    for col in ["player1", "player2", "score", "round", "status", "winner", "raw_row"]:
        if col not in temp.columns:
            temp[col] = ""
        temp[col] = temp[col].fillna("").astype(str).map(normalize_space)

    temp["_dup_key"] = (
        temp["tour_id"].fillna("").astype(str) + "||" +
        temp["player1"] + "||" +
        temp["player2"] + "||" +
        temp["score"] + "||" +
        temp["round"] + "||" +
        temp["raw_row"].str[:250]
    )

    temp = temp.drop_duplicates("_dup_key").drop(columns="_dup_key")
    return temp

def parse_one_html_file(html_path: str, source_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    html_text = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    all_rows: List[Dict[str, Any]] = []

    # 1) Таблицы
    tables = extract_tables_from_html(html_text)
    for table in tables:
        all_rows.extend(table_to_match_rows(table, source_meta))

    # 2) JSON из скриптов
    json_objects = extract_json_objects_from_scripts(html_text)
    for obj in json_objects:
        all_rows.extend(json_object_to_match_rows(obj, source_meta))

    # 3) DOM-блоки
    all_rows.extend(extract_dom_card_rows(html_text, source_meta))

    return all_rows

def parse_saved_htmls(save_log_df: pd.DataFrame) -> pd.DataFrame:
    match_rows: List[Dict[str, Any]] = []

    ok_rows = save_log_df[save_log_df["save_status"] == "ok"].copy()

    for _, row in ok_rows.iterrows():
        html_path = row["html_path"]
        if not Path(html_path).exists():
            continue

        source_meta = {
            "source_index": row.get("source_index", ""),
            "tour_id": row.get("tour_id", ""),
            "Турнир": row.get("Турнир", ""),
            "Возрастная категория": row.get("Возрастная категория", ""),
            "Дата начала": row.get("Дата начала", ""),
            "Город": row.get("Город", ""),
            "Ссылка на страницу с матчами": row.get("Ссылка на страницу с матчами", ""),
            "html_path": html_path,
        }

        try:
            rows = parse_one_html_file(html_path, source_meta)
            match_rows.extend(rows)
        except Exception as exc:
            match_rows.append({
                **source_meta,
                "source_parser": "parse_error",
                "table_columns": "",
                "table_row_index": "",
                "player1": "",
                "player2": "",
                "score": "",
                "round": "",
                "status": f"parse_error: {exc}",
                "winner": "",
                "raw_row": "",
            })

    result_df = pd.DataFrame(match_rows)
    result_df = deduplicate_match_rows(result_df)
    return result_df


# In[13]:


matches_df = parse_saved_htmls(save_log_df)

print(matches_df.shape)
display(matches_df.head(20))

matches_df.to_excel(RESULT_EXCEL_PATH, index=False)
matches_df.to_csv(RESULT_CSV_PATH, index=False, encoding="utf-8-sig")

failed_links_df = save_log_df[save_log_df["save_status"] != "ok"].copy()
failed_links_df.to_excel(FAILED_LINKS_PATH, index=False)

print("Готово:")
print(" -", RESULT_EXCEL_PATH)
print(" -", RESULT_CSV_PATH)
print(" -", FAILED_LINKS_PATH)


# ## 3. Быстрая диагностика результата
# 
# Эти ячейки помогают понять, насколько хорошо сработал парсер.
# 

# In[ ]:


if not matches_df.empty:
    print("Парсеры:")
    display(matches_df["source_parser"].value_counts(dropna=False).rename_axis("source_parser").reset_index(name="rows"))

    print("С турнирами:")
    display(
        matches_df.groupby(["Турнир", "Возрастная категория"], dropna=False)
        .size()
        .reset_index(name="rows_found")
        .sort_values("rows_found", ascending=False)
        .head(30)
    )

    print("Строки без игроков и без счета:")
    weak_rows = matches_df[
        (matches_df["player1"].fillna("").astype(str).str.strip() == "") &
        (matches_df["player2"].fillna("").astype(str).str.strip() == "") &
        (matches_df["score"].fillna("").astype(str).str.strip() == "")
    ]
    display(weak_rows.head(20))
else:
    print("Матчи не извлечены. Проверь saved HTML: возможно, RTT сохранился без данных или структура сильно отличается.")


# ## 4. Если хочешь вручную проверить конкретный HTML
# 
# Подставь путь к одному файлу из `saved_rtt_pages/html` и посмотри, что в нем есть.
# 

# In[ ]:


# Пример:
# test_html_path = "saved_rtt_pages/html/0001_76809_ТВД - _Воскресенск__до 17 лет.html"
# html_text = Path(test_html_path).read_text(encoding="utf-8", errors="ignore")
# print(html_text[:5000])

