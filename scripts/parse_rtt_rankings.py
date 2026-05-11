#!/usr/bin/env python
# coding: utf-8



# In[1]:


# Если нужно установить зависимости, раскомментируй и запусти эту ячейку.
# В Google Colab / новом окружении обычно нужно выполнить обе команды.
# Ниже оставлен Firefox, как в исходном парсере матчей.

# !pip install -q pandas openpyxl beautifulsoup4 lxml playwright tqdm ipywidgets
# !python -m playwright install firefox


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

# # RTT: парсер рейтингов игроков по датам и возрастным группам
# 
# Ноутбук сохраняет рейтинг игроков с сайта RTT/MyTennis по всем доступным датам начиная с `01.02.2025`.
# 
# Что делает код:
# 
# 1. Открывает страницу рейтинга через Playwright, потому что страница рендерится JavaScript.
# 2. Автоматически собирает доступные даты классификации из выпадающего списка.
# 3. Фильтрует даты начиная с `START_DATE`.
# 4. Последовательно перебирает возрастные группы:
#    - `до 15 лет`
#    - `до 17 лет`
#    - `до 19 лет`
#    - `взрослые`
# 5. Для каждой пары `дата × возрастная группа`:
#    - нажимает `Показать`;
#    - пытается установить размер страницы `100`;
#    - парсит все страницы пагинации, а не только первую;
#    - сохраняет ФИО, место, пол, РНИ, дату рождения, город, турниры, очки;
#    - отдельно вытаскивает ссылку из ФИО и РНИ из гиперссылки.
# 6. Делает checkpoint-файлы по каждой паре `дата × возрастная группа`, чтобы при сбое можно было продолжить.
# 7. Собирает финальные файлы:
#    - `rtt_rankings_all_dates.xlsx`
#    - `rtt_rankings_all_dates.csv`
#    - `rtt_rni_mapping.xlsx`
#    - `rtt_ranking_parse_log.xlsx`
# 
# > Важно: если сайт изменит HTML/CSS, основные места для адаптации — функции `set_dropdown_by_label`, `click_show_button`, `try_set_page_size_100`, `click_next_page_if_possible`.
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


# ## 1. Настройки

# In[3]:


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    raise FileNotFoundError("Could not find project root. Run the notebook from the repository folder or a subfolder.")

PROJECT_ROOT = find_project_root()

# =========================
# Основные настройки
# =========================

BASE_URL = "https://rtt.mytennis.online/public/ranking/solo"

START_DATE = "01.02.2025"

AGE_GROUPS = [
    "до 15 лет",
    "до 17 лет",
    "до 19 лет",
    "взрослые",
]

AGE_GROUP_ALIASES = {
    "до 15 лет": ["до 15 лет", "До 15 лет"],
    "до 17 лет": ["до 17 лет", "До 17 лет"],
    "до 19 лет": ["до 19 лет", "До 19 лет"],
    "взрослые": ["взрослые", "Взрослые", "старше 19 лет", "Старше 19 лет"],
}

GENDER_FILTER_VALUE = "Все"

OUTPUT_DIR = PROJECT_ROOT / "rtt_rankings_saved"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
DEBUG_DIR = OUTPUT_DIR / "debug"

FINAL_XLSX_PATH = OUTPUT_DIR / "rtt_rankings_all_dates.xlsx"
FINAL_CSV_PATH = OUTPUT_DIR / "rtt_rankings_all_dates.csv"
FINAL_MAPPING_PATH = OUTPUT_DIR / "rtt_rni_mapping.xlsx"
FINAL_LOG_PATH = OUTPUT_DIR / "rtt_ranking_parse_log.xlsx"
DATE_OPTIONS_PATH = OUTPUT_DIR / "rtt_ranking_date_options.xlsx"

HEADLESS = True
BROWSER_ENGINE = "firefox"    # firefox / chromium / webkit
PAGE_TIMEOUT_MS = 90_000
SHORT_TIMEOUT_MS = 5_000
WAIT_AFTER_ACTION_MS = 800
WAIT_AFTER_SEARCH_MS = 1_500
MAX_PAGES_PER_COMBO = 200

# Если дата-выпадающий список не удалось прочитать,
# лучше оставить False, чтобы не получить тихо неверный набор дат.
ALLOW_MONTHLY_DATE_FALLBACK = False

RESUME_FROM_CHECKPOINTS = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
print(f"PROJECT_ROOT: {PROJECT_ROOT}")


# ## 2. Утилиты дат, строк и файлов

# In[4]:


def normalize_text(value: Any) -> str:
    """Нормализует текст: пробелы, переносы, non-breaking spaces."""
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_ru_date(value: Any) -> Optional[pd.Timestamp]:
    """Парсит дату формата dd.mm.yyyy."""
    text = normalize_text(value)
    if not text:
        return None
    match = re.search(r"(\d{2}\.\d{2}\.\d{4})", text)
    if not match:
        return None
    try:
        return pd.to_datetime(match.group(1), format="%d.%m.%Y")
    except Exception:
        return None


def format_ru_date(ts: Any) -> str:
    ts = pd.to_datetime(ts)
    return ts.strftime("%d.%m.%Y")


def date_sort_key(date_text: str) -> pd.Timestamp:
    parsed = parse_ru_date(date_text)
    if parsed is None:
        return pd.Timestamp.min
    return parsed


def slugify(text: Any) -> str:
    text = normalize_text(text).lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9]+", "_", text, flags=re.IGNORECASE)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "empty"


def extract_digits_from_href(href: str) -> Optional[str]:
    """
    Пытается достать РНИ/ID из ссылки игрока.

    На сайте ФИО игрока является гиперссылкой. Обычно в href есть числовой идентификатор.
    Если в ссылке несколько чисел, берем самое длинное число. Окончательный РНИ также
    берется из отдельной колонки таблицы.
    """
    if not href:
        return None
    numbers = re.findall(r"\d+", href)
    if not numbers:
        return None
    numbers = sorted(numbers, key=len, reverse=True)
    return numbers[0]


def checkpoint_path(ranking_date: str, age_group: str) -> Path:
    return CHECKPOINT_DIR / f"ranking_{slugify(ranking_date)}_{slugify(age_group)}.csv"


def debug_file_base(ranking_date: str, age_group: str, suffix: str) -> Path:
    return DEBUG_DIR / f"{slugify(ranking_date)}_{slugify(age_group)}_{suffix}"


# ## 3. DOM helpers для выпадающих списков и кнопок

# In[5]:


async def safe_wait(page, ms: int = WAIT_AFTER_ACTION_MS) -> None:
    await page.wait_for_timeout(ms)


async def save_debug_artifacts(page, ranking_date: str, age_group: str, reason: str) -> None:
    """Сохраняет HTML и скриншот для диагностики, если парсинг упал."""
    base = debug_file_base(ranking_date, age_group, slugify(reason))
    try:
        html = await page.content()
        base.with_suffix(".html").write_text(html, encoding="utf-8")
    except Exception:
        pass
    try:
        await page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
    except Exception:
        pass


async def save_debug_page(page, file_stem: str) -> None:
    """Диагностика для служебных шагов, где еще нет даты/возрастной группы."""
    base = DEBUG_DIR / slugify(file_stem)
    try:
        html = await page.content()
        base.with_suffix(".html").write_text(html, encoding="utf-8")
    except Exception:
        pass
    try:
        await page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
    except Exception:
        pass


async def click_first_visible(page, selectors: List[str], timeout_ms: int = SHORT_TIMEOUT_MS) -> bool:
    """Пробует кликнуть по первому видимому локатору из списка."""
    for selector in selectors:
        try:
            loc = page.locator(selector)
            count = await loc.count()
            for i in range(count):
                item = loc.nth(i)
                if await item.is_visible():
                    await item.click(timeout=timeout_ms)
                    return True
        except Exception:
            continue
    return False


async def get_visible_texts(page, selectors: List[str]) -> List[str]:
    texts: List[str] = []
    for selector in selectors:
        try:
            loc = page.locator(selector)
            count = await loc.count()
            for i in range(count):
                item = loc.nth(i)
                try:
                    if await item.is_visible():
                        t = normalize_text(await item.inner_text(timeout=1000))
                        if t:
                            texts.append(t)
                except Exception:
                    continue
        except Exception:
            continue
    return texts


async def find_field_handle_by_label(page, label_text: str):
    """
    Находит интерактивный элемент рядом с подписью поля.

    В первой версии был риск: при подъеме к широкому родительскому контейнеру функция
    могла находить первый input формы, а не поле рядом с нужной подписью.
    Здесь используется геометрический поиск: берем подпись и выбираем ближайший
    интерактивный элемент под ней или рядом с ней.
    """
    js = r"""
    (labelText) => {
        const norm = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
        const target = norm(labelText);

        function visible(el) {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style && style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 4 && rect.height > 4;
        }

        const labelCandidates = Array.from(document.querySelectorAll('label, div, span, p, small'))
            .filter(el => visible(el))
            .filter(el => {
                const t = norm(el.innerText || el.textContent);
                return t === target || t.replace(':', '') === target || t.includes(target);
            });

        const interactiveSelector = [
            'select',
            'input',
            'button',
            '[role="combobox"]',
            '.dropdown-toggle',
            '.p-dropdown',
            '.p-dropdown-label',
            '.multiselect',
            '.multiselect__select',
            '.select2-selection',
            '.v-select',
            '.vs__dropdown-toggle',
            '.el-select',
            '.el-input',
            '.form-control',
            '[class*="select"]'
        ].join(',');

        const candidates = Array.from(document.querySelectorAll(interactiveSelector))
            .filter(el => visible(el))
            .filter(el => !el.disabled);

        let best = null;
        let bestScore = Infinity;

        for (const label of labelCandidates) {
            const lr = label.getBoundingClientRect();
            const labelCx = (lr.left + lr.right) / 2;

            for (const cand of candidates) {
                const cr = cand.getBoundingClientRect();
                const candCx = (cr.left + cr.right) / 2;

                // Поле обычно находится под подписью. Допускаем небольшой заход по вертикали,
                // потому что в некоторых верстках label и control находятся в одном контейнере.
                const verticalGap = cr.top - lr.bottom;
                const sameColumn = Math.abs(cr.left - lr.left) < 260 || Math.abs(candCx - labelCx) < 260;
                const belowOrSameLine = verticalGap >= -20 && verticalGap < 110;
                const horizontalOverlap = Math.max(0, Math.min(lr.right, cr.right) - Math.max(lr.left, cr.left));

                if (!sameColumn || !belowOrSameLine) continue;

                let score = Math.abs(verticalGap) * 4 + Math.abs(cr.left - lr.left);
                if (horizontalOverlap <= 0) score += 150;
                if (cr.top < lr.top - 10) score += 500;

                // Не выбираем саму кнопку 'Показать' для полей фильтра.
                const candText = norm(cand.innerText || cand.value || cand.getAttribute('aria-label') || '');
                if (candText.includes('показать')) score += 10000;

                if (score < bestScore) {
                    bestScore = score;
                    best = cand;
                }
            }
        }

        return best;
    }
    """
    return await page.evaluate_handle(js, label_text)


async def open_dropdown_by_label(page, label_text: str) -> bool:
    """Открывает dropdown/combobox рядом с указанной подписью."""
    try:
        handle = await find_field_handle_by_label(page, label_text)
        if handle:
            element = handle.as_element()
            if element:
                await element.click(timeout=SHORT_TIMEOUT_MS)
                await safe_wait(page)
                return True
    except Exception:
        pass

    fallback_selectors = [
        f"xpath=//*[normalize-space(.)='{label_text}']/following::select[1]",
        f"xpath=//*[normalize-space(.)='{label_text}']/following::input[1]",
        f"xpath=//*[normalize-space(.)='{label_text}']/following::*[@role='combobox'][1]",
        f"xpath=//*[normalize-space(.)='{label_text}']/following::*[contains(@class,'dropdown')][1]",
        f"xpath=//*[contains(normalize-space(.), '{label_text}')]/following::select[1]",
        f"xpath=//*[contains(normalize-space(.), '{label_text}')]/following::input[1]",
        f"xpath=//*[contains(normalize-space(.), '{label_text}')]/following::*[@role='combobox'][1]",
    ]
    ok = await click_first_visible(page, fallback_selectors)
    if ok:
        await safe_wait(page)
    return ok


async def collect_open_dropdown_options(page) -> List[str]:
    """Собирает тексты видимых опций из открытого dropdown."""
    option_selectors = [
        "[role='option']",
        ".dropdown-menu.show li",
        ".dropdown-menu.show a",
        ".dropdown-menu.show button",
        ".dropdown-item",
        ".p-dropdown-items .p-dropdown-item",
        ".p-dropdown-panel .p-dropdown-item",
        ".multiselect__content-wrapper .multiselect__option",
        ".multiselect-option",
        ".select2-results__option",
        ".v-menu__content .v-list-item",
        ".vs__dropdown-option",
        ".el-select-dropdown__item",
        ".ng-option",
        "li",
        "option",
    ]
    texts = await get_visible_texts(page, option_selectors)

    # Дополнительный fallback: после открытия dropdown даты часто видны в body.innerText,
    # даже если у опций нестандартные классы.
    try:
        body_text = await page.locator("body").inner_text(timeout=SHORT_TIMEOUT_MS)
        texts.extend(re.findall(r"\d{2}\.\d{2}\.\d{4}", body_text))
    except Exception:
        pass

    cleaned: List[str] = []
    seen = set()
    for t in texts:
        t = normalize_text(t)
        if t and t not in seen:
            cleaned.append(t)
            seen.add(t)
    return cleaned


async def click_visible_option_by_text(page, value: str, exact: bool = True) -> bool:
    """Кликает по видимой опции dropdown по тексту."""
    value_norm = normalize_text(value).lower()

    try:
        option = page.get_by_role("option", name=value, exact=exact)
        count = await option.count()
        for i in range(count):
            item = option.nth(i)
            if await item.is_visible():
                await item.click(timeout=SHORT_TIMEOUT_MS)
                await safe_wait(page)
                return True
    except Exception:
        pass

    option_selectors = [
        "[role='option']",
        ".dropdown-menu.show li",
        ".dropdown-menu.show a",
        ".dropdown-menu.show button",
        ".dropdown-item",
        ".p-dropdown-items .p-dropdown-item",
        ".p-dropdown-panel .p-dropdown-item",
        ".multiselect__content-wrapper .multiselect__option",
        ".multiselect-option",
        ".select2-results__option",
        ".v-menu__content .v-list-item",
        ".vs__dropdown-option",
        ".el-select-dropdown__item",
        ".ng-option",
        "option",
        "li",
        "button",
        "a",
        "span",
        "div",
    ]

    for selector in option_selectors:
        try:
            loc = page.locator(selector)
            count = await loc.count()
            for i in range(count):
                item = loc.nth(i)
                if not await item.is_visible():
                    continue
                text = normalize_text(await item.inner_text(timeout=1000))
                text_norm = text.lower()
                ok = (text_norm == value_norm) if exact else (value_norm in text_norm)
                if ok:
                    await item.click(timeout=SHORT_TIMEOUT_MS)
                    await safe_wait(page)
                    return True
        except Exception:
            continue

    return False


async def set_native_select_by_label(page, label_text: str, value: str) -> bool:
    """Пробует установить значение обычного <select> рядом с подписью поля."""
    try:
        handle = await find_field_handle_by_label(page, label_text)
        if not handle:
            return False
        element = handle.as_element()
        if not element:
            return False

        tag_name = await element.evaluate("(el) => el.tagName.toLowerCase()")
        if tag_name != "select":
            return False

        options = await element.evaluate("""
            (el) => Array.from(el.options).map(o => ({value: o.value, text: (o.textContent || '').trim()}))
        """)
        value_norm = normalize_text(value).lower()
        chosen = None
        for opt in options:
            if normalize_text(opt["text"]).lower() == value_norm:
                chosen = opt["value"]
                break
        if chosen is None:
            for opt in options:
                if value_norm in normalize_text(opt["text"]).lower():
                    chosen = opt["value"]
                    break
        if chosen is None:
            return False

        await element.select_option(chosen)
        await safe_wait(page)
        return True
    except Exception:
        return False


async def set_dropdown_by_label(page, label_text: str, value: str, exact: bool = True) -> bool:
    """
    Универсально устанавливает значение поля по подписи.
    Работает с обычными select и с кастомными dropdown.
    """
    if await set_native_select_by_label(page, label_text, value):
        return True

    opened = await open_dropdown_by_label(page, label_text)
    if not opened:
        return False

    # Для searchable-select пробуем напечатать значение.
    try:
        await page.keyboard.press("Control+A")
        await page.keyboard.type(value)
        await safe_wait(page, 300)
    except Exception:
        pass

    clicked = await click_visible_option_by_text(page, value, exact=exact)

    if not clicked:
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass
    return clicked


# In[6]:


async def goto_ranking_page(page, url: str = BASE_URL, max_attempts: int = 3) -> None:
    """
    Надежный переход на страницу рейтинга.

    Важно: для сайта RTT не используем wait_until='networkidle', потому что страница
    может держать фоновые запросы/соединения, и Playwright будет ждать до timeout.
    Вместо этого ждем DOM + появление ключевых элементов формы.
    """
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Открываю страницу рейтинга, попытка {attempt}/{max_attempts}...")
            await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
            await safe_wait(page, 2_000)

            # Мягкая проверка: страница отрисовалась, если видим хотя бы одно ключевое слово формы.
            try:
                await page.wait_for_selector("text=Классификация игроков", timeout=15_000)
                return
            except Exception:
                pass

            # Fallback: иногда заголовок не доступен как text-selector; проверяем видимый текст DOM.
            body_text = normalize_text(await page.locator("body").inner_text(timeout=15_000))
            if (
                "Классификация игроков" in body_text
                or "Дата классификации" in body_text
                or "Возрастная группа" in body_text
            ):
                return

            raise RuntimeError("Страница открылась, но форма рейтинга не найдена в DOM.")

        except Exception as e:
            last_error = e
            try:
                await page.wait_for_load_state("load", timeout=10_000)
            except Exception:
                pass
            await safe_wait(page, 2_000)

    raise RuntimeError(f"Не удалось открыть страницу рейтинга после {max_attempts} попыток: {last_error}")


# ## 4. Сбор доступных дат классификации

# In[7]:


async def collect_dates_from_body_text(page) -> List[str]:
    """Собирает даты dd.mm.yyyy из видимого текста body."""
    try:
        body_text = await page.locator("body").inner_text(timeout=SHORT_TIMEOUT_MS)
    except Exception:
        return []
    return re.findall(r"\d{2}\.\d{2}\.\d{4}", body_text)


async def collect_dates_from_all_dom_options(page) -> List[str]:
    """Собирает даты из option/li/div/a/span, включая скрытые элементы."""
    js = r"""
    () => {
        const nodes = Array.from(document.querySelectorAll('option, li, a, span, div, button'));
        const out = [];
        for (const n of nodes) {
            const text = (n.innerText || n.textContent || '').replace(/\s+/g, ' ').trim();
            if (/\d{2}\.\d{2}\.\d{4}/.test(text)) out.push(text);
        }
        return out;
    }
    """
    try:
        texts = await page.evaluate(js)
    except Exception:
        return []
    dates: List[str] = []
    for text in texts:
        dates.extend(re.findall(r"\d{2}\.\d{2}\.\d{4}", normalize_text(text)))
    return dates


async def collect_ranking_dates_from_page(page) -> List[str]:
    """
    Считывает доступные даты классификации из dropdown 'Дата классификации'.

    Исправленная версия:
    - не полагается только на role='option';
    - открывает поле через геометрический поиск по label;
    - после открытия читает даты из body.innerText и из DOM;
    - сохраняет debug-html/png, если даты так и не найдены.
    """
    dates: List[str] = []

    # 1) Сначала проверяем DOM до открытия dropdown: иногда option уже есть в HTML.
    dates.extend(await collect_dates_from_all_dom_options(page))

    # 2) Открываем dropdown даты.
    opened = await open_dropdown_by_label(page, "Дата классификации")
    if opened:
        await safe_wait(page, 1000)
        dates.extend(await collect_open_dropdown_options(page))
        dates.extend(await collect_dates_from_body_text(page))
        dates.extend(await collect_dates_from_all_dom_options(page))

        # Иногда кастомный dropdown раскрывается только после стрелки вниз.
        try:
            await page.keyboard.press("ArrowDown")
            await safe_wait(page, 500)
            dates.extend(await collect_dates_from_body_text(page))
            dates.extend(await collect_dates_from_all_dom_options(page))
        except Exception:
            pass

        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass

    # 3) Последний fallback: видимый текст страницы.
    dates.extend(await collect_dates_from_body_text(page))

    only_dates = []
    for text in dates:
        for match in re.findall(r"\d{2}\.\d{2}\.\d{4}", normalize_text(text)):
            only_dates.append(match)

    unique_dates = sorted(set(only_dates), key=date_sort_key)

    if not unique_dates:
        await save_debug_page(page, "date_dropdown_not_found")
        raise RuntimeError(
            "Не удалось найти доступные даты классификации в dropdown. "
            "Сохранены debug-файлы в rtt_rankings_saved/debug/date_dropdown_not_found.*"
        )

    return unique_dates


def monthly_date_fallback(start_date: str) -> List[str]:
    start = pd.to_datetime(start_date, format="%d.%m.%Y")
    today = pd.Timestamp.today().normalize()
    dates = pd.date_range(start=start, end=today, freq="MS")
    return [d.strftime("%d.%m.%Y") for d in dates]


def filter_dates_from_start(available_dates: List[str], start_date: str) -> List[str]:
    start_ts = pd.to_datetime(start_date, format="%d.%m.%Y")
    out = []
    for d in available_dates:
        ts = parse_ru_date(d)
        if ts is not None and ts >= start_ts:
            out.append(d)
    return sorted(set(out), key=date_sort_key)


# ## 5. Нажатие кнопки `Показать`, page size 100 и пагинация

# In[8]:


async def click_show_button(page) -> bool:
    """Нажимает кнопку 'Показать'."""
    selectors = [
        "button:has-text('Показать')",
        "input[type='submit'][value*='Показать']",
        "xpath=//button[contains(normalize-space(.), 'Показать')]",
        "xpath=//*[self::button or self::a][contains(normalize-space(.), 'Показать')]",
    ]

    if await click_first_visible(page, selectors):
        await safe_wait(page, WAIT_AFTER_SEARCH_MS)
        return True

    try:
        button = page.get_by_role("button", name=re.compile("Показать", re.IGNORECASE))
        if await button.count() > 0:
            await button.first.click(timeout=SHORT_TIMEOUT_MS)
            await safe_wait(page, WAIT_AFTER_SEARCH_MS)
            return True
    except Exception:
        pass

    return False


async def wait_for_ranking_table(page) -> None:
    """Ждет появления таблицы рейтинга."""
    try:
        await page.wait_for_selector("table", timeout=PAGE_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        raise RuntimeError("Таблица рейтинга не появилась после нажатия 'Показать'.")
    await safe_wait(page, 1000)


async def try_set_page_size_100(page) -> bool:
    """
    Пытается установить '100 записей на странице'.

    Если не получится — это не критично: код все равно пройдет по всем страницам пагинации.
    """
    try:
        selects = page.locator("select")
        count = await selects.count()
        for i in range(count):
            sel = selects.nth(i)
            if not await sel.is_visible():
                continue
            options = await sel.evaluate("""
                (el) => Array.from(el.options).map(o => ({value: o.value, text: (o.textContent || '').trim()}))
            """)
            chosen = None
            for opt in options:
                if normalize_text(opt["text"]) == "100" or normalize_text(opt["value"]) == "100":
                    chosen = opt["value"]
                    break
            if chosen is not None:
                await sel.select_option(chosen)
                await safe_wait(page, WAIT_AFTER_SEARCH_MS)
                return True
    except Exception:
        pass

    candidate_texts = ["10", "20", "25", "50"]
    for current_text in candidate_texts:
        try:
            loc = page.get_by_text(current_text, exact=True)
            count = await loc.count()
            for i in range(count):
                item = loc.nth(i)
                if not await item.is_visible():
                    continue
                await item.click(timeout=1500)
                await safe_wait(page, 300)
                if await click_visible_option_by_text(page, "100", exact=True):
                    await safe_wait(page, WAIT_AFTER_SEARCH_MS)
                    return True
        except Exception:
            continue

    return False


async def click_next_page_if_possible(page) -> bool:
    """
    Кликает следующую страницу пагинации.
    Возвращает True, если переход выполнен. False — если следующей страницы нет.
    """
    next_selectors = [
        "button[aria-label*='Next']",
        "a[aria-label*='Next']",
        "button[aria-label*='След']",
        "a[aria-label*='След']",
        "button:has-text('След')",
        "a:has-text('След')",
        "button:has-text('›')",
        "a:has-text('›')",
        "button:has-text('»')",
        "a:has-text('»')",
        "button:has-text('>')",
        "a:has-text('>')",
        "xpath=//*[self::a or self::button][contains(normalize-space(.), 'След')]",
        "xpath=//*[self::a or self::button][normalize-space(.)='›' or normalize-space(.)='»' or normalize-space(.)='>']",
    ]

    for selector in next_selectors:
        try:
            loc = page.locator(selector)
            count = await loc.count()
            for i in reversed(range(count)):
                item = loc.nth(i)
                if not await item.is_visible():
                    continue

                disabled_attr = await item.get_attribute("disabled")
                aria_disabled = await item.get_attribute("aria-disabled")
                class_attr = await item.get_attribute("class") or ""
                parent_class = ""
                try:
                    parent_class = await item.evaluate("(el) => el.parentElement ? el.parentElement.className : ''")
                except Exception:
                    pass

                disabled = (
                    disabled_attr is not None
                    or str(aria_disabled).lower() == "true"
                    or "disabled" in class_attr.lower()
                    or "disabled" in str(parent_class).lower()
                )
                if disabled:
                    continue

                await item.click(timeout=SHORT_TIMEOUT_MS)
                await safe_wait(page, WAIT_AFTER_SEARCH_MS)
                return True
        except Exception:
            continue

    return False


# ## 6. Парсинг таблицы рейтинга

# In[9]:


EXPECTED_COLUMNS = {
    "место": "place",
    "фио": "fio",
    "пол игрока": "gender",
    "рни": "rni",
    "дата рождения": "birth_date",
    "город": "city",
    "всего турниров": "total_tournaments",
    "из них зачётных": "counting_tournaments",
    "из них зачетных": "counting_tournaments",
    "возрастная группа": "age_group_in_table",
    "очки": "points",
}


def normalize_header(text: str) -> str:
    text = normalize_text(text).lower().replace("ё", "е")
    return text


def find_ranking_table(soup: BeautifulSoup):
    """Находит таблицу рейтинга по заголовкам ФИО/РНИ."""
    tables = soup.find_all("table")
    best_table = None
    best_score = -1

    for table in tables:
        header_text = " ".join(
            normalize_text(th.get_text(" "))
            for th in table.find_all(["th", "td"])[:20]
        ).lower()
        score = 0
        for key in ["фио", "рни", "очки", "место"]:
            if key in header_text:
                score += 1
        if score > best_score:
            best_score = score
            best_table = table

    if best_table is None or best_score < 2:
        return None
    return best_table


def extract_table_headers(table) -> List[str]:
    """Извлекает заголовки таблицы."""
    header_row = None
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")

    if header_row is None:
        rows = table.find_all("tr")
        if rows:
            header_row = rows[0]

    if header_row is None:
        return []

    headers = [normalize_text(cell.get_text(" ")) for cell in header_row.find_all(["th", "td"])]
    return headers


def parse_ranking_table_from_html(html: str) -> List[Dict[str, Any]]:
    """Парсит текущую HTML-таблицу рейтинга."""
    soup = BeautifulSoup(html, "lxml")
    table = find_ranking_table(soup)
    if table is None:
        return []

    headers = extract_table_headers(table)
    if not headers:
        return []

    mapped_headers: List[str] = []
    for h in headers:
        h_norm = normalize_header(h)
        mapped = EXPECTED_COLUMNS.get(h_norm, slugify(h_norm))
        mapped_headers.append(mapped)

    rows: List[Dict[str, Any]] = []

    tbody = table.find("tbody")
    tr_list = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]

    for tr_idx, tr in enumerate(tr_list, start=1):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue

        cell_texts = [normalize_text(td.get_text(" ")) for td in cells]
        if not any(cell_texts):
            continue

        row: Dict[str, Any] = {}
        for idx, value in enumerate(cell_texts):
            key = mapped_headers[idx] if idx < len(mapped_headers) else f"extra_col_{idx}"
            row[key] = value

        fio_link = None
        player_href = None
        rni_from_href = None

        fio_col_idx = None
        for idx, key in enumerate(mapped_headers):
            if key == "fio":
                fio_col_idx = idx
                break

        link_search_cells = []
        if fio_col_idx is not None and fio_col_idx < len(cells):
            link_search_cells.append(cells[fio_col_idx])
        link_search_cells.extend(cells)

        for cell in link_search_cells:
            a = cell.find("a", href=True)
            if a:
                fio_link = normalize_text(a.get_text(" "))
                player_href = a.get("href")
                rni_from_href = extract_digits_from_href(player_href or "")
                break

        if fio_link:
            row["fio_from_link"] = fio_link
        row["player_href"] = player_href
        row["rni_from_href"] = rni_from_href

        rni_text = normalize_text(row.get("rni"))
        row["rni_final"] = rni_text if rni_text else rni_from_href

        row["_row_number_on_html_page"] = tr_idx
        rows.append(row)

    return rows


async def parse_current_ranking_page(page, ranking_date: str, age_group: str, page_number: int) -> List[Dict[str, Any]]:
    html = await page.content()
    rows = parse_ranking_table_from_html(html)

    for row in rows:
        row["ranking_date"] = ranking_date
        row["age_group_filter"] = age_group
        row["page_number"] = page_number
        row["source_url"] = BASE_URL

    return rows


# ## 7. Парсинг одной пары `дата × возрастная группа`

# In[10]:


async def select_age_group(page, age_group: str) -> bool:
    """Выбирает возрастную группу, учитывая возможные варианты названия."""
    aliases = AGE_GROUP_ALIASES.get(age_group, [age_group])
    for alias in aliases:
        if await set_dropdown_by_label(page, "Возрастная группа", alias, exact=True):
            return True
        if await set_dropdown_by_label(page, "Возрастная группа", alias, exact=False):
            return True
    return False


async def select_gender_all(page) -> bool:
    """Выбирает Пол игрока = Все. Если не получилось, не считаем это фатальной ошибкой."""
    try:
        return await set_dropdown_by_label(page, "Пол игрока", GENDER_FILTER_VALUE, exact=True)
    except Exception:
        return False


async def parse_one_date_age_group(page, ranking_date: str, age_group: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Парсит все страницы для одной пары дата/возрастная группа.
    """
    started_at = datetime.now()
    log: Dict[str, Any] = {
        "ranking_date": ranking_date,
        "age_group": age_group,
        "status": "started",
        "rows": 0,
        "pages": 0,
        "error": None,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": None,
    }

    all_rows: List[Dict[str, Any]] = []

    try:
        await goto_ranking_page(page)
        await safe_wait(page, 1000)

        await select_gender_all(page)

        ok_age = await select_age_group(page, age_group)
        if not ok_age:
            raise RuntimeError(f"Не удалось выбрать возрастную группу: {age_group}")

        ok_date = await set_dropdown_by_label(page, "Дата классификации", ranking_date, exact=True)
        if not ok_date:
            opened = await open_dropdown_by_label(page, "Дата классификации")
            if opened:
                try:
                    await page.keyboard.press("Control+A")
                    await page.keyboard.type(ranking_date)
                    await page.keyboard.press("Enter")
                    await safe_wait(page)
                    ok_date = True
                except Exception:
                    ok_date = False

        if not ok_date:
            raise RuntimeError(f"Не удалось выбрать дату классификации: {ranking_date}")

        ok_show = await click_show_button(page)
        if not ok_show:
            raise RuntimeError("Не удалось нажать кнопку 'Показать'.")

        await wait_for_ranking_table(page)

        page_size_set = await try_set_page_size_100(page)
        log["page_size_100_set"] = page_size_set

        previous_page_signature = None

        for page_number in range(1, MAX_PAGES_PER_COMBO + 1):
            await wait_for_ranking_table(page)
            rows = await parse_current_ranking_page(page, ranking_date, age_group, page_number=page_number)

            current_signature = json.dumps(rows[:3], ensure_ascii=False, sort_keys=True)
            if page_number > 1 and current_signature == previous_page_signature:
                # Защита от бесконечного цикла, если кнопка next нажалась, но таблица не сменилась.
                break
            previous_page_signature = current_signature

            if rows:
                all_rows.extend(rows)

            log["pages"] = page_number
            log["rows"] = len(all_rows)

            next_ok = await click_next_page_if_possible(page)
            if not next_ok:
                break

        df = pd.DataFrame(all_rows)

        if not df.empty:
            dedupe_cols = [
                c for c in [
                    "ranking_date",
                    "age_group_filter",
                    "place",
                    "fio",
                    "rni_final",
                    "birth_date",
                    "gender",
                    "city",
                    "points",
                ]
                if c in df.columns
            ]
            if dedupe_cols:
                df = df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)

        log["status"] = "ok"
        log["rows"] = len(df)
        log["finished_at"] = datetime.now().isoformat(timespec="seconds")
        return df, log

    except Exception as e:
        log["status"] = "error"
        log["error"] = repr(e)
        log["finished_at"] = datetime.now().isoformat(timespec="seconds")
        await save_debug_artifacts(page, ranking_date, age_group, "error")
        return pd.DataFrame(), log


# ## Важное исправление загрузки страницы
# 
# Для страницы RTT не используется `wait_until="networkidle"`, потому что сайт может держать фоновые сетевые запросы, из-за чего Playwright ждет до таймаута.  
# Вместо этого используется `goto_ranking_page()`: переход по `domcontentloaded` + проверка появления текста формы рейтинга.

# ## 8. Основной запуск парсера

# In[11]:


async def run_ranking_parser() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Основная функция:
    - получает список доступных дат;
    - парсит все date × age_group;
    - пишет checkpoint'ы;
    - собирает итоговые Excel/CSV.
    """
    all_result_frames: List[pd.DataFrame] = []
    log_rows: List[Dict[str, Any]] = []

    async with async_playwright() as playwright:
        browser_launcher = getattr(playwright, BROWSER_ENGINE)
        browser = await browser_launcher.launch(headless=HEADLESS)
        context = await browser.new_context(
            viewport={"width": 1600, "height": 2200},
            locale="ru-RU",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)

        try:
            print("Открываю страницу рейтинга...")
            await goto_ranking_page(page)
            await safe_wait(page, 1500)

            print("Собираю доступные даты классификации...")
            try:
                available_dates = await collect_ranking_dates_from_page(page)
            except Exception as e:
                if not ALLOW_MONTHLY_DATE_FALLBACK:
                    raise
                print(f"Не удалось собрать даты из dropdown: {e}")
                print("Использую fallback: первые числа месяцев.")
                available_dates = monthly_date_fallback(START_DATE)

            dates_to_parse = filter_dates_from_start(available_dates, START_DATE)

            if not dates_to_parse:
                raise RuntimeError(
                    f"Не найдено дат классификации >= {START_DATE}. "
                    f"Всего доступных дат: {len(available_dates)}"
                )

            date_options_df = pd.DataFrame({
                "available_ranking_date": available_dates,
                "parsed_date": [format_ru_date(date_sort_key(d)) for d in available_dates],
                "selected_for_parsing": [d in set(dates_to_parse) for d in available_dates],
            })
            date_options_df.to_excel(DATE_OPTIONS_PATH, index=False)

            print(f"Доступных дат всего: {len(available_dates)}")
            print(f"Дат к парсингу начиная с {START_DATE}: {len(dates_to_parse)}")
            print("Первая дата:", dates_to_parse[0], "| Последняя дата:", dates_to_parse[-1])
            print("Возрастные группы:", AGE_GROUPS)

            tasks = [(d, ag) for d in dates_to_parse for ag in AGE_GROUPS]

            for ranking_date, age_group in tqdm(tasks, desc="Парсинг рейтингов"):
                cp_path = checkpoint_path(ranking_date, age_group)

                if RESUME_FROM_CHECKPOINTS and cp_path.exists():
                    try:
                        cached_df = pd.read_csv(cp_path, dtype=str)
                        all_result_frames.append(cached_df)
                        log_rows.append({
                            "ranking_date": ranking_date,
                            "age_group": age_group,
                            "status": "loaded_from_checkpoint",
                            "rows": len(cached_df),
                            "pages": cached_df["page_number"].nunique() if "page_number" in cached_df.columns else None,
                            "error": None,
                            "started_at": None,
                            "finished_at": datetime.now().isoformat(timespec="seconds"),
                        })
                        continue
                    except Exception:
                        pass

                df_part, log = await parse_one_date_age_group(page, ranking_date, age_group)
                log_rows.append(log)

                if not df_part.empty:
                    df_part.to_csv(cp_path, index=False, encoding="utf-8-sig")
                    all_result_frames.append(df_part)

                await safe_wait(page, 500)

        finally:
            await context.close()
            await browser.close()

    if all_result_frames:
        result_df = pd.concat(all_result_frames, ignore_index=True)
    else:
        result_df = pd.DataFrame()

    log_df = pd.DataFrame(log_rows)

    if not result_df.empty:
        for col in ["place", "total_tournaments", "counting_tournaments", "points"]:
            if col in result_df.columns:
                result_df[col + "_num"] = (
                    result_df[col]
                    .astype(str)
                    .str.replace(" ", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                result_df[col + "_num"] = pd.to_numeric(result_df[col + "_num"], errors="coerce")

        if "ranking_date" in result_df.columns:
            result_df["ranking_date_dt"] = result_df["ranking_date"].apply(parse_ru_date)
        if "birth_date" in result_df.columns:
            result_df["birth_date_dt"] = result_df["birth_date"].apply(parse_ru_date)

        if "player_href" in result_df.columns:
            result_df["player_url"] = result_df["player_href"].apply(
                lambda h: None
                if not isinstance(h, str) or not h.strip()
                else (h if h.startswith("http") else "https://rtt.mytennis.online" + h)
            )

        dedupe_cols = [
            c for c in [
                "ranking_date",
                "age_group_filter",
                "place",
                "fio",
                "rni_final",
                "birth_date",
                "gender",
                "city",
                "points",
            ]
            if c in result_df.columns
        ]
        if dedupe_cols:
            result_df = result_df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)

    result_df.to_csv(FINAL_CSV_PATH, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(FINAL_XLSX_PATH, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="rankings_long", index=False)
        log_df.to_excel(writer, sheet_name="parse_log", index=False)

    log_df.to_excel(FINAL_LOG_PATH, index=False)

    print("Готово.")
    print(f"Итоговых строк: {len(result_df)}")
    print(f"Файл рейтингов Excel: {FINAL_XLSX_PATH}")
    print(f"Файл рейтингов CSV:   {FINAL_CSV_PATH}")
    print(f"Лог парсинга:         {FINAL_LOG_PATH}")

    return result_df, log_df


# ## 9. Запуск

# In[12]:


# Запуск парсера.
#
# В Jupyter Notebook достаточно выполнить:
# result_df, log_df = asyncio.run(run_ranking_parser())
#
# Если запускаешь как обычный .py-файл или в окружении без top-level await,
# используй:
# result_df, log_df = asyncio.run(run_ranking_parser())

result_df, log_df = asyncio.run(run_ranking_parser())

display(log_df.tail(10))
display(result_df.head())


# ## 10. Построение отдельного маппинга ФИО ↔ РНИ ↔ ссылка

# In[13]:


def build_rni_mapping(rankings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит отдельную таблицу маппинга:
    ФИО -> РНИ -> ссылка на карточку игрока.

    Один игрок может встречаться в нескольких возрастных группах и датах,
    поэтому дополнительно сохраняем первую/последнюю дату наблюдения и список групп.
    """
    df = rankings_df.copy()

    if df.empty:
        return pd.DataFrame()

    for col in ["fio", "fio_from_link", "rni", "rni_from_href", "rni_final", "player_href", "player_url"]:
        if col not in df.columns:
            df[col] = None

    df["fio_best"] = df["fio_from_link"].where(
        df["fio_from_link"].notna() & (df["fio_from_link"].astype(str).str.strip() != ""),
        df["fio"],
    )
    df["rni_best"] = df["rni_final"].where(
        df["rni_final"].notna() & (df["rni_final"].astype(str).str.strip() != ""),
        df["rni_from_href"],
    )

    agg_dict = {
        "rni": lambda x: sorted(set([normalize_text(v) for v in x if normalize_text(v)])),
        "rni_from_href": lambda x: sorted(set([normalize_text(v) for v in x if normalize_text(v)])),
        "player_url": lambda x: sorted(set([normalize_text(v) for v in x if normalize_text(v)])),
    }

    for optional_col in ["gender", "birth_date", "city", "age_group_filter"]:
        if optional_col in df.columns:
            agg_dict[optional_col] = lambda x: sorted(set([normalize_text(v) for v in x if normalize_text(v)]))

    if "ranking_date" in df.columns:
        agg_dict["ranking_date"] = ["min", "max", "nunique"]

    mapping = df.groupby(["rni_best", "fio_best"], dropna=False).agg(agg_dict)

    mapping.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in mapping.columns
    ]
    mapping = mapping.reset_index()

    for col in mapping.columns:
        if mapping[col].apply(lambda v: isinstance(v, list)).any():
            mapping[col] = mapping[col].apply(lambda v: "; ".join(v) if isinstance(v, list) else v)

    mapping = mapping.sort_values(["fio_best", "rni_best"], na_position="last").reset_index(drop=True)
    return mapping


mapping_df = build_rni_mapping(result_df)
mapping_df.to_excel(FINAL_MAPPING_PATH, index=False)

print(f"Маппинг сохранен: {FINAL_MAPPING_PATH}")
print(f"Уникальных строк маппинга: {len(mapping_df)}")
display(mapping_df.head())


# ## 11. Проверки качества выгрузки

# In[14]:


def show_basic_quality_report(rankings_df: pd.DataFrame, log_df: pd.DataFrame) -> None:
    print("=== LOG STATUS ===")
    if not log_df.empty and "status" in log_df.columns:
        display(log_df["status"].value_counts(dropna=False).to_frame("count"))

    print("=== ROWS BY DATE AND AGE GROUP ===")
    if not rankings_df.empty and {"ranking_date", "age_group_filter", "fio"}.issubset(rankings_df.columns):
        pivot = (
            rankings_df
            .pivot_table(
                index="ranking_date",
                columns="age_group_filter",
                values="fio",
                aggfunc="count",
                fill_value=0,
            )
            .sort_index(key=lambda s: s.map(date_sort_key))
        )
        display(pivot.tail(20))

    print("=== RNI COVERAGE ===")
    if not rankings_df.empty:
        total = len(rankings_df)
        empty = pd.Series(index=rankings_df.index, dtype=object)

        rni_col = rankings_df["rni"] if "rni" in rankings_df.columns else empty
        rni_href = rankings_df["rni_from_href"] if "rni_from_href" in rankings_df.columns else empty
        rni_final = rankings_df["rni_final"] if "rni_final" in rankings_df.columns else empty

        has_rni_col = rni_col.notna() & (rni_col.astype(str).str.strip() != "")
        has_rni_href = rni_href.notna() & (rni_href.astype(str).str.strip() != "")
        has_rni_final = rni_final.notna() & (rni_final.astype(str).str.strip() != "")

        coverage = pd.DataFrame([
            {"metric": "rows_total", "value": total, "share": 1.0},
            {"metric": "has_rni_column", "value": int(has_rni_col.sum()), "share": float(has_rni_col.mean())},
            {"metric": "has_rni_from_href", "value": int(has_rni_href.sum()), "share": float(has_rni_href.mean())},
            {"metric": "has_rni_final", "value": int(has_rni_final.sum()), "share": float(has_rni_final.mean())},
        ])
        display(coverage)

    print("=== ERROR LOG ===")
    if not log_df.empty and "status" in log_df.columns:
        errors = log_df[log_df["status"].astype(str).str.contains("error", case=False, na=False)]
        display(errors)


show_basic_quality_report(result_df, log_df)


# ## 12. Что делать, если сайт изменил верстку
# 
# 1. Открой папку `rtt_rankings_saved/debug`.
# 2. Найди HTML/PNG для упавшей комбинации `дата × возрастная группа`.
# 3. Проверь:
#    - видит ли код кнопку `Показать`;
#    - видит ли код таблицу;
#    - как называется dropdown взрослых;
#    - есть ли в таблице обычный `<table>`, или сайт перешел на `div-grid`.
# 
# Чаще всего достаточно поправить:
# 
# - `AGE_GROUP_ALIASES`
# - `click_show_button`
# - `try_set_page_size_100`
# - `click_next_page_if_possible`
# - `EXPECTED_COLUMNS`, если изменились названия колонок.
