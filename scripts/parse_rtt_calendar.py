from __future__ import annotations

import argparse
import asyncio
import html as html_module
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    BeautifulSoup = None

try:
    from bootstrap_tournaments_master import MASTER_COLUMNS, merge_master
except ModuleNotFoundError:
    from scripts.bootstrap_tournaments_master import MASTER_COLUMNS, merge_master


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://rtt.mytennis.online"
CALENDAR_URL = f"{BASE_URL}/public/tours/calendar"
DEFAULT_MASTER_PATH = PROJECT_ROOT / "data" / "tournaments_master.xlsx"
DEBUG_DIR = PROJECT_ROOT / "data" / "calendar_debug"

DEFAULT_AGE_CATEGORIES = ["до 17 лет", "до 19 лет", "Взрослые"]
SHORT_TIMEOUT_MS = 3_000
WAIT_AFTER_ACTION_MS = 700


@dataclass(frozen=True)
class CalendarConfig:
    date_from: date
    date_to: date
    status: str = "Все"
    draw_type: str = "Одиночный"
    gender: str = "Женский"
    age_categories: tuple[str, ...] = tuple(DEFAULT_AGE_CATEGORIES)
    category: str = "Все"
    system: str = "Все"
    federal_district: str = "Центральный ФО"
    subject: str | None = None
    city: str | None = None
    headless: bool = True
    browser_engine: str = "firefox"
    timeout_ms: int = 60_000


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def slugify(value: str) -> str:
    value = normalize_text(value).lower()
    value = re.sub(r"[^0-9a-zа-яё]+", "_", value, flags=re.IGNORECASE)
    return value.strip("_") or "empty"


def parse_cli_date(value: str) -> date:
    value = normalize_text(value)
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Unsupported date format: {value}. Use DD.MM.YYYY or YYYY-MM-DD.")


def default_date_from(master_path: Path) -> date:
    if not master_path.exists():
        return date.today() - timedelta(days=30)
    df = pd.read_excel(master_path, usecols=lambda col: col == "start_date")
    if df.empty or "start_date" not in df.columns:
        return date.today() - timedelta(days=30)
    dates = pd.to_datetime(df["start_date"], errors="coerce")
    if not dates.notna().any():
        return date.today() - timedelta(days=30)
    return dates.max().date() - timedelta(days=14)


def extract_tour_id(url: str) -> str | None:
    match = re.search(r"/tours/(\d+)", str(url))
    return match.group(1) if match else None


def absolute_url(href: str) -> str:
    href = str(href or "").strip()
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return f"{BASE_URL}{href}"
    return f"{BASE_URL}/{href}"


def matches_url_from_calendar_url(url: str) -> str:
    url = absolute_url(url).split("?")[0].rstrip("/")
    url = re.sub(r"/dashboard.*$", "", url)
    if url.endswith("/matches"):
        return url
    return f"{url}/matches"


def matches_url_from_tour_id(tour_id: str) -> str:
    return f"{BASE_URL}/public/tours/{tour_id}/matches"


def guess_city(text: str) -> str | None:
    lines = [normalize_text(line) for line in text.splitlines() if normalize_text(line)]
    location_pattern = re.compile(r"^(.+?),\s*(?:[^,]+,\s*)?Центральный ФО,\s*Россия$", re.IGNORECASE)
    for line in lines:
        match = location_pattern.search(line)
        if match:
            return normalize_text(match.group(1)) or None
    city_markers = ("г.", "город", "место проведения")
    for line in lines:
        line_lower = line.lower()
        if any(marker in line_lower for marker in city_markers):
            cleaned = re.sub(r"^(г\.|город|место проведения)\s*[:.-]?\s*", "", line, flags=re.IGNORECASE)
            return normalize_text(cleaned) or None
    return None


def guess_start_date(text: str) -> pd.Timestamp:
    patterns = [
        r"(\d{1,2}\.\d{1,2}\.\d{4})",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return pd.to_datetime(match.group(1), dayfirst=True, errors="coerce").normalize()

    ru_months = {
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
    match = re.search(r"(\d{1,2})\s+([а-яё]+)\s+[—-]\s+\d{1,2}\s+[а-яё]+,?\s+(\d{4})", text, re.IGNORECASE)
    if match:
        month = ru_months.get(match.group(2).lower())
        if month:
            return pd.Timestamp(year=int(match.group(3)), month=month, day=int(match.group(1))).normalize()
    return pd.NaT


def looks_like_draw_description(text: str) -> bool:
    text_norm = normalize_text(text).lower()
    return (
        text_norm.startswith("основной турнир")
        or "по рейтингу" in text_norm
        or " поэ:" in text_norm
        or " ск:" in text_norm
    )


def pick_tournament_name(anchor, container_text: str) -> str:
    candidates = [normalize_text(anchor.get_text(" "))]
    for line in container_text.split("\n"):
        line = normalize_text(line)
        if line:
            candidates.append(line)

    for candidate in candidates:
        candidate_lower = candidate.lower()
        if len(candidate) < 3:
            continue
        if candidate_lower in {"подробнее", "заявка", "матчи", "турнир", "dashboard"}:
            continue
        if looks_like_draw_description(candidate):
            continue
        if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}.*", candidate):
            continue
        return candidate

    fallback = normalize_text(container_text.split("\n", 1)[0])
    return fallback or normalize_text(anchor.get_text(" ")) or "RTT tournament"


def collect_anchor_context_text(anchor, max_chars: int = 4_000) -> str:
    parts: list[str] = []
    for parent in anchor.parents:
        text = normalize_text(parent.get_text("\n"))
        if text and len(text) <= max_chars:
            parts.append(text)
    return "\n".join(parts)


def count_tour_links(element) -> int:
    ids: set[str] = set()
    for link in element.find_all("a", href=True):
        tour_id = extract_tour_id(link["href"])
        if tour_id:
            ids.add(tour_id)
    return len(ids)


def best_container(anchor):
    for parent in anchor.parents:
        if getattr(parent, "name", None) in {"body", "html"}:
            break
        text = normalize_text(parent.get_text("\n"))
        if not text or len(text) > 2_000:
            continue
        if count_tour_links(parent) == 1:
            return parent

    for parent in anchor.parents:
        if getattr(parent, "name", None) in {"tr", "li"}:
            return parent
        classes = " ".join(parent.get("class", [])) if hasattr(parent, "get") else ""
        if re.search(r"(tour|tournament|event|calendar|card|row|item)", classes, re.IGNORECASE):
            text = normalize_text(parent.get_text(" "))
            anchor_text = normalize_text(anchor.get_text(" "))
            if len(text) > 20 and (not looks_like_draw_description(anchor_text) or len(text) > len(anchor_text) + 30):
                return parent
    return anchor.parent or anchor


class CalendarAnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors: list[dict[str, str]] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href and extract_tour_id(href):
            self._current_href = href
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href:
            self.anchors.append({"href": self._current_href, "text": normalize_text(" ".join(self._current_text))})
            self._current_href = None
            self._current_text = []


def parse_calendar_html_without_bs4(html: str, config: CalendarConfig, age_category: str, fetched_at: str) -> pd.DataFrame:
    parser = CalendarAnchorParser()
    parser.feed(html)
    rows: list[dict] = []
    seen: set[str] = set()
    page_text = normalize_text(re.sub(r"<[^>]+>", " ", html))

    for anchor in parser.anchors:
        href = absolute_url(anchor["href"])
        tour_id = extract_tour_id(href)
        if not tour_id or tour_id in seen:
            continue
        calendar_url = absolute_url(href)
        matches_url = matches_url_from_tour_id(tour_id)
        rows.append(
            {
                "tour_id": tour_id,
                "tournament_name": f"RTT tournament {tour_id}",
                "age_category": age_category,
                "gender": config.gender,
                "draw_type": config.draw_type,
                "federal_district": config.federal_district,
                "city": config.city or pd.NA,
                "start_date": pd.NaT,
                "status": config.status,
                "category": config.category,
                "system": config.system,
                "matches_url": matches_url,
                "calendar_url": calendar_url,
                "matches_page_saved": False,
                "source": "rtt_calendar",
                "source_file": pd.NA,
                "created_at": fetched_at,
                "updated_at": fetched_at,
            }
        )
        seen.add(tour_id)

    if not rows:
        return pd.DataFrame(columns=MASTER_COLUMNS)
    return pd.DataFrame(rows)[MASTER_COLUMNS].drop_duplicates(subset=["tour_id"], keep="first").reset_index(drop=True)


def parse_calendar_html(html: str, config: CalendarConfig, age_category: str, fetched_at: str) -> pd.DataFrame:
    if BeautifulSoup is None:
        return parse_calendar_html_without_bs4(html, config, age_category, fetched_at)

    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = absolute_url(anchor["href"])
        tour_id = extract_tour_id(href)
        if not tour_id or tour_id in seen:
            continue

        container = best_container(anchor)
        container_text = normalize_text(container.get_text("\n"))
        context_text = container_text
        tournament_name = pick_tournament_name(anchor, container_text)

        calendar_url = absolute_url(href)
        matches_url = matches_url_from_tour_id(tour_id)
        rows.append(
            {
                "tour_id": tour_id,
                "tournament_name": f"RTT tournament {tour_id}",
                "age_category": age_category,
                "gender": config.gender,
                "draw_type": config.draw_type,
                "federal_district": config.federal_district,
                "city": config.city or pd.NA,
                "start_date": pd.NaT,
                "status": config.status,
                "category": config.category,
                "system": config.system,
                "matches_url": matches_url,
                "calendar_url": calendar_url,
                "matches_page_saved": False,
                "source": "rtt_calendar",
                "source_file": pd.NA,
                "created_at": fetched_at,
                "updated_at": fetched_at,
            }
        )
        seen.add(tour_id)

    if not rows:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    df = pd.DataFrame(rows)
    df = df[MASTER_COLUMNS]
    df = df.drop_duplicates(subset=["tour_id"], keep="first")
    return df


def text_from_class(soup, class_name: str) -> str | None:
    if BeautifulSoup is None:
        return None
    node = soup.find(class_=lambda value: value and class_name in str(value).split())
    return normalize_text(node.get_text(" ")) if node else None


def strip_html(value: str) -> str:
    value = re.sub(r"<script\b[^>]*>.*?</script>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<style\b[^>]*>.*?</style>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    return normalize_text(html_module.unescape(value))


def regex_text_from_class(html: str, class_name: str) -> str | None:
    pattern = rf"<[^>]+class=[\"'][^\"']*\b{re.escape(class_name)}\b[^\"']*[\"'][^>]*>(.*?)</[^>]+>"
    match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
    return strip_html(match.group(1)) if match else None


def regex_label_value(html: str, label: str) -> str | None:
    pattern = rf"<div[^>]*>\s*{re.escape(label)}:?\s*</div>\s*<div[^>]*>(.*?)</div>"
    match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
    return strip_html(match.group(1)) if match else None


def parse_tournament_detail_html_regex(html: str, row: dict, config: CalendarConfig) -> dict:
    tournament_name = regex_text_from_class(html, "name")
    if tournament_name:
        row["tournament_name"] = tournament_name

    date_range = regex_text_from_class(html, "small-name")
    start_date = guess_start_date(date_range or "")
    if pd.notna(start_date):
        row["start_date"] = start_date

    place = regex_label_value(html, "Место проведения")
    if place:
        row["city"] = guess_city(place) or place.split(",", 1)[0].strip()
        row["federal_district"] = "Центральный ФО" if "Центральный ФО" in place else config.federal_district

    field_map = {
        "Пол игроков": "gender",
        "Возрастная группа": "age_category",
        "Разряд турнира": "draw_type",
        "Категория турнира": "category",
        "Система проведения": "system",
    }
    for label, column in field_map.items():
        value = regex_label_value(html, label)
        if value:
            row[column] = value

    status_text = strip_html(html)
    for status in ("Турнир завершен", "В процессе проведения", "Подача поздних заявок", "Подача заявок"):
        if status in status_text:
            row["status"] = status
            break
    return row


def parse_label_values(soup) -> dict[str, str]:
    values: dict[str, str] = {}
    if BeautifulSoup is None:
        return values

    for row in soup.find_all(class_=lambda value: value and "two-columns" in str(value).split()):
        children = [child for child in row.find_all("div", recursive=False)]
        if len(children) < 2:
            continue
        label = normalize_text(children[0].get_text(" ")).rstrip(":")
        value = normalize_text(children[1].get_text(" "))
        if label and value:
            values[label] = value
    return values


def parse_tournament_detail_html(
    html: str,
    base_row: dict,
    config: CalendarConfig,
    fetched_at: str,
) -> dict:
    tour_id = str(base_row["tour_id"])
    row = {column: base_row.get(column, pd.NA) for column in MASTER_COLUMNS}
    row["tour_id"] = tour_id
    row["matches_url"] = matches_url_from_tour_id(tour_id)
    row["calendar_url"] = base_row.get("calendar_url", f"{BASE_URL}/public/tours/{tour_id}/dashboard")
    row["source"] = "rtt_calendar"
    row["source_file"] = pd.NA
    row["matches_page_saved"] = False
    row["created_at"] = base_row.get("created_at", fetched_at)
    row["updated_at"] = fetched_at

    if BeautifulSoup is None:
        return parse_tournament_detail_html_regex(html, row, config)

    soup = BeautifulSoup(html, "html.parser")
    labels = parse_label_values(soup)

    tournament_name = text_from_class(soup, "name")
    if tournament_name:
        row["tournament_name"] = tournament_name

    date_range = text_from_class(soup, "small-name")
    start_date = guess_start_date(date_range or "")
    if pd.notna(start_date):
        row["start_date"] = start_date

    place = labels.get("Место проведения")
    if place:
        row["city"] = guess_city(place) or place.split(",", 1)[0].strip()
        row["federal_district"] = "Центральный ФО" if "Центральный ФО" in place else config.federal_district

    row["gender"] = labels.get("Пол игроков", row.get("gender", config.gender))
    row["age_category"] = labels.get("Возрастная группа", row.get("age_category", pd.NA))
    row["draw_type"] = labels.get("Разряд турнира", row.get("draw_type", config.draw_type))
    row["category"] = labels.get("Категория турнира", row.get("category", config.category))
    row["system"] = labels.get("Система проведения", row.get("system", config.system))

    status_text = normalize_text(soup.get_text(" "))
    for status in ("Турнир завершен", "В процессе проведения", "Подача поздних заявок", "Подача заявок"):
        if status in status_text:
            row["status"] = status
            break

    return row


async def safe_wait(page, ms: int = WAIT_AFTER_ACTION_MS) -> None:
    await page.wait_for_timeout(ms)


async def save_debug_artifacts(page, stem: str) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    base = DEBUG_DIR / slugify(stem)
    try:
        html = await page.content()
        base.with_suffix(".html").write_text(html, encoding="utf-8")
    except Exception:
        pass
    try:
        await page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
    except Exception:
        pass


async def click_visible_option_by_text(page, value: str, exact: bool = True) -> bool:
    value_norm = normalize_text(value).lower()
    selectors = [
        "[role='option']",
        ".dropdown-menu.show li",
        ".dropdown-item",
        ".p-dropdown-item",
        ".multiselect__option",
        ".select2-results__option",
        ".v-list-item",
        ".vs__dropdown-option",
        ".el-select-dropdown__item",
        ".ng-option",
        "li",
        "button",
        "a",
        "span",
        "div",
    ]
    for selector in selectors:
        try:
            loc = page.locator(selector)
            count = await loc.count()
            for idx in range(count):
                item = loc.nth(idx)
                if not await item.is_visible():
                    continue
                text_norm = normalize_text(await item.inner_text(timeout=1_000)).lower()
                ok = text_norm == value_norm if exact else value_norm in text_norm
                if ok:
                    await item.click(timeout=SHORT_TIMEOUT_MS)
                    await safe_wait(page)
                    return True
        except Exception:
            continue
    return False


async def click_visible_text_by_mouse(page, value: str, exact: bool = True) -> bool:
    js = r"""
    ([value, exact]) => {
        const norm = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
        const target = norm(value);
        const visible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 4 && rect.height > 4;
        };
        const candidates = Array.from(document.querySelectorAll('[role="option"], .v-list-item, .v-list-item__title, .v-menu__content *'))
            .filter(visible)
            .map(el => ({ el, text: norm(el.innerText || el.textContent || '') }))
            .filter(item => exact ? item.text === target : item.text.includes(target));
        const item = candidates[0];
        if (!item) return null;
        const rect = item.el.getBoundingClientRect();
        return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2, text: item.text };
    }
    """
    point = await page.evaluate(js, [value, exact])
    if not point:
        return False
    await page.mouse.click(point["x"], point["y"])
    await safe_wait(page)
    return True


async def find_field_handle_by_label(page, label_text: str):
    js = r"""
    (labelText) => {
        const norm = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
        const target = norm(labelText);
        const visible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 4 && rect.height > 4;
        };
        const labels = Array.from(document.querySelectorAll('label, div, span, p, small'))
            .filter(visible)
            .filter(el => {
                const t = norm(el.innerText || el.textContent);
                return t === target || t.replace(':', '') === target || t.includes(target);
            });
        const interactiveSelector = [
            'select', 'input', 'button', '[role="combobox"]',
            '.v-input:not(.v-input--is-disabled)',
            '.v-input__control', '.v-input__slot', '.v-select__slot',
            '.v-select', '.v-text-field',
            '.dropdown-toggle', '.p-dropdown', '.p-dropdown-label',
            '.multiselect', '.multiselect__select', '.select2-selection',
            '.v-select', '.vs__dropdown-toggle', '.el-select', '.el-input',
            '.form-control', '[class*="select"]'
        ].join(',');
        const candidates = Array.from(document.querySelectorAll(interactiveSelector))
            .filter(visible)
            .filter(el => !el.disabled);
        let best = null;
        let bestScore = Infinity;
        for (const label of labels) {
            const lr = label.getBoundingClientRect();
            const labelCx = (lr.left + lr.right) / 2;
            for (const cand of candidates) {
                const cr = cand.getBoundingClientRect();
                const candCx = (cr.left + cr.right) / 2;
                const verticalGap = cr.top - lr.bottom;
                const sameColumn = Math.abs(cr.left - lr.left) < 300 || Math.abs(candCx - labelCx) < 300;
                const belowOrSameLine = verticalGap >= -24 && verticalGap < 130;
                if (!sameColumn || !belowOrSameLine) continue;
                let score = Math.abs(verticalGap) * 4 + Math.abs(cr.left - lr.left);
                const candText = norm(cand.innerText || cand.value || cand.getAttribute('aria-label') || '');
                if (candText.includes('показать') || candText.includes('найти') || candText.includes('очистить')) score += 10000;
                if (cand.matches && cand.matches('input')) score -= 20;
                if (cand.classList && (cand.classList.contains('v-input__slot') || cand.classList.contains('v-select__slot'))) score -= 15;
                if (score < bestScore) {
                    bestScore = score;
                    best = cand;
                }
            }
        }
        if (!best) {
            const order = [
                'статус проведения',
                'разряд',
                'пол',
                'возраст',
                'категория',
                'система проведения',
                'период проведения',
                'федеральный округ',
                'субъект',
                'город',
                'название турнира',
                'организатор'
            ];
            const index = order.indexOf(target);
            const inputs = Array.from(document.querySelectorAll('.ToursCalendar .v-input, .ToursCalendar input'))
                .filter(visible)
                .filter(el => {
                    const t = norm(el.innerText || el.value || el.getAttribute('aria-label') || '');
                    return !t.includes('найти') && !t.includes('очистить');
                });
            if (index >= 0 && index < inputs.length) {
                best = inputs[index];
            }
        }
        return best;
    }
    """
    return await page.evaluate_handle(js, label_text)


async def click_field_by_label(page, label_text: str) -> bool:
    js = r"""
    (labelText) => {
        const norm = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
        const target = norm(labelText);
        const visible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 4 && rect.height > 4;
        };
        const labels = Array.from(document.querySelectorAll('.ToursCalendar label, .ToursCalendar .v-label, .ToursCalendar div, .ToursCalendar span'))
            .filter(visible)
            .filter(el => {
                const t = norm(el.innerText || el.textContent);
                return t === target || t.replace(':', '') === target;
            });
        const controls = Array.from(document.querySelectorAll('.ToursCalendar .v-input:not(.v-input--is-disabled), .ToursCalendar input'))
            .filter(visible);
        let best = null;
        let bestScore = Infinity;
        for (const label of labels) {
            const lr = label.getBoundingClientRect();
            for (const control of controls) {
                const cr = control.getBoundingClientRect();
                const verticalGap = cr.top - lr.bottom;
                const horizontalNear = Math.abs(cr.left - lr.left) < 80 || (cr.left <= lr.left && cr.right >= lr.left);
                if (verticalGap < -30 || verticalGap > 85 || !horizontalNear) continue;
                const score = Math.abs(verticalGap) * 5 + Math.abs(cr.left - lr.left);
                if (score < bestScore) {
                    bestScore = score;
                    best = control;
                }
            }
        }
        if (!best) {
            const order = [
                'статус проведения',
                'разряд',
                'пол',
                'возраст',
                'категория',
                'система проведения',
                'период проведения',
                'федеральный округ',
                'субъект',
                'город',
                'название турнира',
                'организатор'
            ];
            const index = order.indexOf(target);
            if (index >= 0 && index < controls.length) best = controls[index];
        }
        if (!best) return null;
        const slot = best.querySelector('.v-input__slot, .v-select__slot, input') || best;
        const rect = slot.getBoundingClientRect();
        return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2, text: norm(best.innerText || best.value || '') };
    }
    """
    point = await page.evaluate(js, label_text)
    if not point:
        return False
    await page.mouse.click(point["x"], point["y"])
    await safe_wait(page)
    return True


async def set_native_select_by_label(page, label_text: str, value: str) -> bool:
    try:
        handle = await find_field_handle_by_label(page, label_text)
        element = handle.as_element() if handle else None
        if not element:
            return False
        tag_name = await element.evaluate("(el) => el.tagName.toLowerCase()")
        if tag_name != "select":
            return False
        options = await element.evaluate(
            "(el) => Array.from(el.options).map(o => ({value: o.value, text: (o.textContent || '').trim()}))"
        )
        value_norm = normalize_text(value).lower()
        chosen = None
        for option in options:
            text_norm = normalize_text(option["text"]).lower()
            if text_norm == value_norm or value_norm in text_norm:
                chosen = option["value"]
                break
        if chosen is None:
            return False
        await element.select_option(chosen)
        await safe_wait(page)
        return True
    except Exception:
        return False


async def set_dropdown_by_label(page, label_text: str, value: str, exact: bool = True) -> bool:
    if value is None or value == "":
        return True
    current_value = await get_field_text_by_label(page, label_text)
    if current_value and normalize_text(value).lower() in current_value.lower():
        return True
    if await set_native_select_by_label(page, label_text, value):
        return True
    try:
        if not await click_field_by_label(page, label_text):
            return False
        try:
            await page.keyboard.press("Control+A")
            await page.keyboard.type(value)
            await safe_wait(page, 300)
        except Exception:
            pass
        clicked = await click_visible_option_by_text(page, value, exact=exact)
        if not clicked:
            clicked = await click_visible_text_by_mouse(page, value, exact=exact)
        if not clicked:
            await page.keyboard.press("Escape")
        return clicked
    except Exception:
        return False


async def get_field_text_by_label(page, label_text: str) -> str | None:
    try:
        handle = await find_field_handle_by_label(page, label_text)
        element = handle.as_element() if handle else None
        if not element:
            return None
        text = await element.evaluate(
            "(el) => (el.innerText || el.value || el.getAttribute('aria-label') || '').replace(/\\s+/g, ' ').trim()"
        )
        return normalize_text(text)
    except Exception:
        return None


async def fill_text_by_label(page, label_text: str, value: str) -> bool:
    if value is None or value == "":
        return True
    try:
        if not await click_field_by_label(page, label_text):
            return False
        await page.keyboard.press("Control+A")
        await page.keyboard.type(value)
        await page.keyboard.press("Enter")
        await safe_wait(page)
        return True
    except Exception:
        return False


async def click_find(page) -> None:
    for name in ("Найти", "Показать"):
        try:
            button = page.get_by_role("button", name=name, exact=False)
            count = await button.count()
            for idx in range(count):
                item = button.nth(idx)
                if await item.is_visible():
                    await item.click(timeout=SHORT_TIMEOUT_MS)
                    await safe_wait(page, 2_500)
                    return
        except Exception:
            continue
    raise RuntimeError("Could not find the calendar search button.")


async def set_rows_per_page(page, value: str = "100") -> bool:
    try:
        footer_select = page.locator(".v-data-footer__select .v-select, .v-data-footer__select .v-input")
        if await footer_select.count() == 0:
            return False
        await footer_select.first.click(timeout=SHORT_TIMEOUT_MS)
        await safe_wait(page, 300)
        if await click_visible_option_by_text(page, value, exact=True):
            await page.wait_for_timeout(1_500)
            return True
        if await click_visible_text_by_mouse(page, value, exact=True):
            await page.wait_for_timeout(1_500)
            return True
        await page.keyboard.press("Escape")
        return False
    except Exception:
        return False


async def click_next_calendar_page(page) -> bool:
    js = r"""
    () => {
        const visible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 4 && rect.height > 4;
        };
        const candidates = Array.from(document.querySelectorAll([
            '.v-data-footer__icons-after button',
            '.v-data-footer button[aria-label*="Next"]',
            '.v-data-footer button[aria-label*="След"]',
            '.v-data-footer button'
        ].join(','))).filter(visible);

        const enabled = candidates.filter(button => {
            const disabled = button.disabled || button.getAttribute('aria-disabled') === 'true';
            const classes = button.className || '';
            return !disabled && !String(classes).includes('v-btn--disabled');
        });
        const button = enabled[0];
        if (!button) return null;
        const rect = button.getBoundingClientRect();
        return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
    }
    """
    point = await page.evaluate(js)
    if not point:
        return False
    await page.mouse.click(point["x"], point["y"])
    await page.wait_for_timeout(1_500)
    return True


async def parse_all_calendar_pages(page, config: CalendarConfig, age_category: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen_ids: set[str] = set()
    fetched_at = datetime.now(timezone.utc).isoformat()

    await set_rows_per_page(page, "100")

    for page_number in range(1, 101):
        html = await page.content()
        df = parse_calendar_html(html, config, age_category, fetched_at)
        if df.empty:
            if page_number == 1:
                await save_debug_artifacts(page, f"no_tournaments_{age_category}")
            break

        new_df = df[~df["tour_id"].astype(str).isin(seen_ids)].copy()
        if new_df.empty and page_number > 1:
            break
        frames.append(new_df)
        seen_ids.update(new_df["tour_id"].astype(str))

        if not await click_next_calendar_page(page):
            break

    if not frames:
        return pd.DataFrame(columns=MASTER_COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["tour_id"], keep="first")


async def wait_for_tournament_detail(page) -> None:
    try:
        await page.get_by_text("Карточка турнира", exact=False).first.wait_for(timeout=30_000)
    except Exception:
        await page.wait_for_timeout(3_000)


async def fetch_tournament_detail(page, base_row: dict, config: CalendarConfig) -> dict:
    tour_id = str(base_row["tour_id"])
    url = matches_url_from_tour_id(tour_id)
    await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    await wait_for_tournament_detail(page)
    html = await page.content()
    return parse_tournament_detail_html(html, base_row, config, datetime.now(timezone.utc).isoformat())


async def enrich_calendar_details(page, calendar_df: pd.DataFrame, config: CalendarConfig) -> pd.DataFrame:
    if calendar_df.empty:
        return calendar_df

    rows: list[dict] = []
    total = len(calendar_df)
    for index, (_, base_row) in enumerate(calendar_df.iterrows(), start=1):
        tour_id = str(base_row["tour_id"])
        print(f"Fetching tournament details [{index}/{total}]: {tour_id}")
        try:
            rows.append(await fetch_tournament_detail(page, base_row.to_dict(), config))
        except Exception as exc:
            print(f"Warning: failed to fetch details for tour_id={tour_id}: {exc}")
            rows.append(base_row.to_dict())
            try:
                await save_debug_artifacts(page, f"detail_fail_{tour_id}")
            except Exception:
                pass

    return pd.DataFrame(rows)[MASTER_COLUMNS].drop_duplicates(subset=["tour_id"], keep="first").reset_index(drop=True)


async def apply_filters(page, config: CalendarConfig, age_category: str) -> list[str]:
    failures: list[str] = []
    dropdowns = [
        ("Статус проведения", config.status, True, config.status != "Все"),
        ("Разряд", config.draw_type, True, True),
        ("Пол", config.gender, True, True),
        ("Возраст", age_category, True, True),
        ("Категория", config.category, True, config.category != "Все"),
        ("Система проведения", config.system, True, config.system != "Все"),
        ("Федеральный округ", config.federal_district, False, True),
        ("Субъект", config.subject, False, False),
        ("Город", config.city, False, False),
    ]
    for label, value, exact, required in dropdowns:
        if value and not await set_dropdown_by_label(page, label, value, exact=exact) and required:
            failures.append(label)

    period_value = f"с {config.date_from.strftime('%d.%m.%Y')} по {config.date_to.strftime('%d.%m.%Y')}"
    if not await fill_text_by_label(page, "Период проведения", period_value):
        failures.append("Период проведения")

    return failures


async def wait_for_calendar(page) -> None:
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            print(f"Opening RTT calendar, attempt {attempt}/3", flush=True)
            await page.goto(CALENDAR_URL, wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(5_000)

            heading = page.get_by_role("heading", name="Календарь турниров", exact=True)
            if await heading.count():
                await heading.first.wait_for(timeout=20_000)
                return

            body_text = normalize_text(await page.locator("body").inner_text(timeout=20_000))
            if (
                "Календарь турниров" in body_text
                or "Статус проведения" in body_text
                or "Период проведения" in body_text
            ):
                return

            raise RuntimeError("Calendar form text was not found after page load.")
        except Exception as exc:
            last_error = exc
            await save_debug_artifacts(page, f"calendar_open_fail_attempt_{attempt}")
            await page.wait_for_timeout(3_000 * attempt)

    raise RuntimeError(f"Could not open RTT calendar after 3 attempts: {last_error}") from last_error


async def fetch_calendar_age_group(page, config: CalendarConfig, age_category: str) -> pd.DataFrame:
    await wait_for_calendar(page)
    failures = await apply_filters(page, config, age_category)
    if failures:
        await save_debug_artifacts(page, f"filter_fail_{age_category}")
        print(f"Warning: failed to set filters for {age_category}: {', '.join(failures)}")
    await click_find(page)
    await page.wait_for_timeout(3_000)
    return await parse_all_calendar_pages(page, config, age_category)


async def fetch_calendar(config: CalendarConfig) -> pd.DataFrame:
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Playwright is not installed in the current Python environment. "
            "Run: python -m pip install playwright && python -m playwright install firefox"
        ) from exc

    frames: list[pd.DataFrame] = []
    async with async_playwright() as playwright:
        launcher = getattr(playwright, config.browser_engine)
        browser = await launcher.launch(headless=config.headless)
        page = await browser.new_page()
        page.set_default_timeout(config.timeout_ms)
        try:
            for age_category in config.age_categories:
                print(f"Parsing calendar age group: {age_category}")
                frames.append(await fetch_calendar_age_group(page, config, age_category))
            calendar_ids = (
                pd.concat(frames, ignore_index=True).drop_duplicates(subset=["tour_id"], keep="first")
                if frames
                else pd.DataFrame(columns=MASTER_COLUMNS)
            )
            result = await enrich_calendar_details(page, calendar_ids, config)
        finally:
            await browser.close()

    return result[MASTER_COLUMNS].reset_index(drop=True)


def merge_into_master(incoming: pd.DataFrame, master_path: Path) -> pd.DataFrame:
    existing = pd.read_excel(master_path) if master_path.exists() else None
    master = merge_master(existing, incoming)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_excel(master_path, index=False)
    return master


def parse_age_categories(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        for part in str(value).split(","):
            text = normalize_text(part)
            if text:
                result.append(text)
    return tuple(result) or tuple(DEFAULT_AGE_CATEGORIES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse RTT tournament calendar and update data/tournaments_master.xlsx.")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER_PATH, help="Tournament master Excel path.")
    parser.add_argument("--date-from", type=parse_cli_date, default=None, help="Calendar start date: DD.MM.YYYY or YYYY-MM-DD.")
    parser.add_argument("--date-to", type=parse_cli_date, default=date.today(), help="Calendar end date: DD.MM.YYYY or YYYY-MM-DD.")
    parser.add_argument("--age", action="append", default=None, help="Age category. Can be repeated or comma-separated.")
    parser.add_argument("--federal-district", default="Центральный ФО")
    parser.add_argument("--gender", default="Женский")
    parser.add_argument("--draw-type", default="Одиночный")
    parser.add_argument("--status", default="Все")
    parser.add_argument("--category", default="Все")
    parser.add_argument("--system", default="Все")
    parser.add_argument("--subject", default=None)
    parser.add_argument("--city", default=None)
    parser.add_argument("--headed", action="store_true", help="Show browser window while parsing.")
    args = parser.parse_args()

    master_path = args.master if args.master.is_absolute() else PROJECT_ROOT / args.master
    config = CalendarConfig(
        date_from=args.date_from or default_date_from(master_path),
        date_to=args.date_to,
        status=args.status,
        draw_type=args.draw_type,
        gender=args.gender,
        age_categories=parse_age_categories(args.age or DEFAULT_AGE_CATEGORIES),
        category=args.category,
        system=args.system,
        federal_district=args.federal_district,
        subject=args.subject,
        city=args.city,
        headless=not args.headed,
    )

    incoming = asyncio.run(fetch_calendar(config))
    if incoming.empty:
        print("No tournaments parsed. Master file was not changed.")
        return

    existing_rows = len(pd.read_excel(master_path)) if master_path.exists() else 0
    master = merge_into_master(incoming, master_path)
    print(f"Parsed tournaments: {len(incoming)}")
    print(f"Master before: {existing_rows}")
    print(f"Master after: {len(master)}")
    print(f"New tournaments: {max(len(master) - existing_rows, 0)}")
    print(f"Output: {master_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
