"""Parsing and cache helpers for the isolated RTT tournament-analysis mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from .tournament_simulation import TournamentPlayer, draw_size_for


RTT_PUBLIC_ROOT = "https://rtt.mytennis.online/public/tours"
TOURNAMENT_ROUTES = ("dashboard", "requests/", "members", "grid", "matches")


def normalize_space(value: object) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


def normalize_name(value: object) -> str:
    return normalize_space(value).casefold().replace("ё", "е")


@dataclass(slots=True)
class ParsedPlayer:
    name: str
    source_id: str = ""
    rni: str = ""
    rank: float | None = None
    points: float | None = None
    seed: int | None = None
    affiliation: str = ""
    request_status: str = ""
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def player_id(self) -> str:
        if self.rni:
            return f"RNI:{self.rni}"
        if self.source_id:
            return f"RTT:{self.source_id}"
        return f"NAME:{normalize_name(self.name)}"

    def as_tournament_player(self) -> TournamentPlayer:
        return TournamentPlayer(
            player_id=self.player_id,
            name=self.name,
            rank=self.rank,
            points=self.points,
            seed=self.seed,
            affiliation=self.affiliation,
            metadata={"request_status": self.request_status, **self.raw},
        )


@dataclass(slots=True)
class ParsedMatch:
    player1: str
    player2: str
    winner: str = ""
    score: str = ""
    round_name: str = ""
    status: str = ""


@dataclass(slots=True)
class TournamentSnapshot:
    tour_id: str
    fetched_at: str
    title: str = ""
    status: str = ""
    category: str = ""
    age_group: str = ""
    gender: str = ""
    draw_system: str = ""
    start_date: str = ""
    end_date: str = ""
    main_draw_capacity: int | None = None
    players: list[ParsedPlayer] = field(default_factory=list)
    player_source: str = ""
    grid_slots: list[str | None] = field(default_factory=list)
    grid_rounds: list[list[str | None]] = field(default_factory=list)
    completed_matches: list[ParsedMatch] = field(default_factory=list)
    eligible: bool = False
    eligibility_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    page_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TournamentSnapshot":
        data = dict(payload)
        data["players"] = [ParsedPlayer(**dict(row)) for row in data.get("players", [])]
        data["completed_matches"] = [ParsedMatch(**dict(row)) for row in data.get("completed_matches", [])]
        return cls(**data)

    def tournament_players(self) -> list[TournamentPlayer]:
        return [player.as_tournament_player() for player in self.players]


@dataclass(slots=True)
class _Cell:
    text: str
    links: list[tuple[str, str]]


class _RenderedHTMLParser(HTMLParser):
    """Extract visible text, anchors and rendered table rows without bs4/lxml."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.visible_parts: list[str] = []
        self.anchors: list[tuple[str, str]] = []
        self.tables: list[list[list[_Cell]]] = []
        self._ignored_depth = 0
        self._anchor_href = ""
        self._anchor_parts: list[str] = []
        self._table_depth = 0
        self._table: list[list[_Cell]] | None = None
        self._row: list[_Cell] | None = None
        self._cell_parts: list[str] | None = None
        self._cell_links: list[tuple[str, str]] = []
        self.grid_first_round_cells: list[tuple[str, str]] = []
        self.grid_round_cells: dict[int, list[tuple[str, str]]] = {}
        self._div_classes: list[set[str]] = []
        self._grid_round0_depth: int | None = None
        self._grid_round_index: int | None = None
        self._grid_cell_depth: int | None = None
        self._grid_slot_depth: int | None = None
        self._grid_slot_side = ""
        self._grid_cell_parts: dict[str, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        if tag in {"script", "style", "svg", "noscript"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "div":
            classes = set(attrs_dict.get("class", "").split())
            self._div_classes.append(classes)
            depth = len(self._div_classes)
            round_class = next((value for value in classes if re.fullmatch(r"round\d+", value)), "")
            if "cell-wrapper" in classes and round_class:
                self._grid_round0_depth = depth
                self._grid_round_index = int(round_class.removeprefix("round"))
            elif (
                self._grid_round0_depth is not None
                and "TourGridCell" in classes
                and "cell-pointer" in classes
            ):
                self._grid_cell_depth = depth
                self._grid_cell_parts = {"top": [], "bottom": []}
            elif self._grid_cell_depth is not None and "cell-player" in classes:
                if "cell-top" in classes:
                    self._grid_slot_side = "top"
                    self._grid_slot_depth = depth
                elif "cell-bottom" in classes:
                    self._grid_slot_side = "bottom"
                    self._grid_slot_depth = depth
        if tag == "a":
            self._anchor_href = attrs_dict.get("href", "")
            self._anchor_parts = []
        elif tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._table = []
        elif tag == "tr" and self._table_depth == 1:
            self._row = []
        elif tag in {"th", "td"} and self._table_depth == 1:
            self._cell_parts = []
            self._cell_links = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth:
            return
        if tag == "div" and self._div_classes:
            depth = len(self._div_classes)
            if self._grid_slot_depth == depth:
                self._grid_slot_depth = None
                self._grid_slot_side = ""
            if self._grid_cell_depth == depth:
                if self._grid_cell_parts is not None:
                    cell = (
                        normalize_space(" ".join(self._grid_cell_parts["top"])),
                        normalize_space(" ".join(self._grid_cell_parts["bottom"])),
                    )
                    if self._grid_round_index is not None:
                        self.grid_round_cells.setdefault(self._grid_round_index, []).append(cell)
                    if self._grid_round_index == 0:
                        self.grid_first_round_cells.append(cell)
                self._grid_cell_depth = None
                self._grid_cell_parts = None
            if self._grid_round0_depth == depth:
                self._grid_round0_depth = None
                self._grid_round_index = None
            self._div_classes.pop()
        if tag == "a" and self._anchor_href:
            text = normalize_space(" ".join(self._anchor_parts))
            link = (self._anchor_href, text)
            self.anchors.append(link)
            if self._cell_parts is not None:
                self._cell_links.append(link)
            self._anchor_href = ""
            self._anchor_parts = []
        elif tag in {"th", "td"} and self._cell_parts is not None:
            if self._row is not None:
                self._row.append(_Cell(normalize_space(" ".join(self._cell_parts)), list(self._cell_links)))
            self._cell_parts = None
            self._cell_links = []
        elif tag == "tr" and self._row is not None:
            if self._table is not None and any(cell.text or cell.links for cell in self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table_depth:
            if self._table_depth == 1 and self._table:
                self.tables.append(self._table)
            self._table_depth -= 1
            if self._table_depth == 0:
                self._table = None

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = normalize_space(data)
        if not text:
            return
        self.visible_parts.append(text)
        if self._anchor_href:
            self._anchor_parts.append(text)
        if self._cell_parts is not None:
            self._cell_parts.append(text)
        if self._grid_cell_parts is not None and self._grid_slot_side:
            self._grid_cell_parts[self._grid_slot_side].append(text)

    @property
    def visible_text(self) -> str:
        return normalize_space(" ".join(self.visible_parts))


def parse_rendered_html(html: str) -> _RenderedHTMLParser:
    parser = _RenderedHTMLParser()
    parser.feed(html)
    parser.close()
    return parser


def _capture(text: str, start: str, following: Sequence[str]) -> str:
    stop = "|".join(re.escape(value) for value in following)
    pattern = rf"{re.escape(start)}\s*(.*?)\s*(?={stop}|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return normalize_space(match.group(1)) if match else ""


_KNOWN_STATUSES = (
    "Прием заявок",
    "Приём заявок",
    "Поздняя заявка",
    "Формирование состава",
    "Регистрация",
    "Жеребьевка",
    "Жеребьёвка",
    "Готов к проведению",
    "Проводится",
    "В процессе",
    "Идет",
    "Идёт",
    "Сдача отчета",
    "Сдача отчёта",
    "Завершен",
    "Завершён",
    "Отменен",
    "Отменён",
    "Не состоялся",
    "Аннулирован",
)


def parse_metadata(html: str, tour_id: str) -> dict[str, object]:
    parsed = parse_rendered_html(html)
    text = parsed.visible_text
    fields = (
        "Категория турнира:",
        "Разряд турнира:",
        "Пол игроков:",
        "Возрастная группа:",
        "Система проведения:",
        "Формат проведения:",
        "Кол-во участников:",
        "Место проведения:",
    )
    category = _capture(text, fields[0], fields[1:])
    gender = _capture(text, fields[2], fields[3:])
    age_group = _capture(text, fields[3], fields[4:])
    draw_system = _capture(text, fields[4], fields[5:])

    date_match = re.search(
        r"(?:Карточка турнира\s+)?(.{3,180}?)\s+(\d{2}\.\d{2}\.\d{4})\s*[-–—]\s*(\d{2}\.\d{2}\.\d{4})\s+Рег\.\s*номер\s+"
        + re.escape(str(tour_id)),
        text,
        flags=re.IGNORECASE,
    )
    title = start_date = end_date = ""
    if date_match:
        title = normalize_space(date_match.group(1))
        if "карточка турнира" in title.casefold():
            title = normalize_space(re.split(r"карточка турнира", title, flags=re.IGNORECASE)[-1])
        title_words = title.split()
        if len(title_words) >= 2 and title_words[0].casefold() == title_words[1].casefold():
            title = " ".join(title_words[1:])
        start_date = datetime.strptime(date_match.group(2), "%d.%m.%Y").date().isoformat()
        end_date = datetime.strptime(date_match.group(3), "%d.%m.%Y").date().isoformat()

    capacity_match = re.search(r"Кол-во участников:\s*ОТ:\s*(\d+)", text, flags=re.IGNORECASE)
    status = ""
    category_position = text.casefold().find("категория турнира:")
    status_area = text[max(0, category_position - 160):category_position] if category_position >= 0 else text
    for candidate in _KNOWN_STATUSES:
        if candidate.casefold() in status_area.casefold() or candidate.casefold() in text.casefold():
            status = candidate
            break

    return {
        "title": title,
        "status": status,
        "category": category,
        "age_group": age_group,
        "gender": gender,
        "draw_system": draw_system,
        "start_date": start_date,
        "end_date": end_date,
        "main_draw_capacity": int(capacity_match.group(1)) if capacity_match else None,
    }


def _number(value: object) -> float | None:
    text = normalize_space(value).replace(" ", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _column_index(headers: Sequence[str], markers: Iterable[str]) -> int | None:
    normalized = [normalize_name(value).replace(" ", "") for value in headers]
    markers = tuple(normalize_name(value).replace(" ", "") for value in markers)
    for index, header in enumerate(normalized):
        if any(marker in header for marker in markers):
            return index
    return None


def _player_link(cell: _Cell) -> tuple[str, str] | None:
    for href, text in cell.links:
        if any(
            marker in href.casefold()
            for marker in ("/players/", "/player/", "/sportsmen/", "/ranking/solo/")
        ):
            return href, text
    return None


def _source_id_from_href(href: str) -> str:
    match = re.search(r"/(?:players?|sportsmen|ranking/solo)/(\d+)", href, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def parse_players(html: str) -> list[ParsedPlayer]:
    parsed = parse_rendered_html(html)
    result: list[ParsedPlayer] = []
    for table in parsed.tables:
        header_row_index = next(
            (
                index
                for index, row in enumerate(table[:4])
                if any(
                    marker in normalize_name(cell.text)
                    for cell in row
                    for marker in ("игрок", "участник", "фио", "ф.и.о", "спортсмен", "рни")
                )
            ),
            None,
        )
        if header_row_index is None:
            continue
        headers = [cell.text for cell in table[header_row_index]]
        name_index = _column_index(headers, ("игрок", "участник", "фио", "ф.и.о", "спортсмен"))
        rni_index = _column_index(headers, ("рни",))
        rank_index = _column_index(headers, ("место", "рейтинг", "рейт."))
        points_index = _column_index(headers, ("очки",))
        seed_index = _column_index(headers, ("посев", "сеян"))
        affiliation_index = _column_index(headers, ("регион", "организац", "клуб", "город"))
        # RTT's current request table has both "Дата заявки" and
        # "Примечание".  Treating any "заявк" column as the status used to
        # capture the application timestamp instead of its actual note.
        status_index = _column_index(headers, ("статус", "список", "примечание"))

        for row in table[header_row_index + 1:]:
            if not row:
                continue
            links = [link for cell in row for link in cell.links]
            linked_player = next(
                ((href, text) for href, text in links if _source_id_from_href(href)),
                None,
            )
            name = ""
            source_id = ""
            if name_index is not None and name_index < len(row):
                name = row[name_index].text
                link = _player_link(row[name_index])
                if link:
                    source_id = _source_id_from_href(link[0])
                    name = link[1] or name
            elif linked_player:
                source_id = _source_id_from_href(linked_player[0])
                name = linked_player[1]
            if not name or normalize_name(name) in {"x", "bye", "свободен"}:
                continue

            values = [cell.text for cell in row]
            raw = {
                headers[index] if index < len(headers) else f"column_{index}": value
                for index, value in enumerate(values)
            }
            rni_text = values[rni_index] if rni_index is not None and rni_index < len(values) else ""
            rni_match = re.search(r"\d+", rni_text.replace(" ", ""))
            if rni_match is None:
                ranking_link = next((href for href, _ in links if "/ranking/solo/" in href.casefold()), "")
                linked_rni = _source_id_from_href(ranking_link)
                if linked_rni:
                    rni_match = re.search(r"\d+", linked_rni)
            rank = _number(values[rank_index]) if rank_index is not None and rank_index < len(values) else None
            points = _number(values[points_index]) if points_index is not None and points_index < len(values) else None
            seed_value = _number(values[seed_index]) if seed_index is not None and seed_index < len(values) else None
            result.append(
                ParsedPlayer(
                    name=normalize_space(name),
                    source_id=source_id,
                    rni=rni_match.group(0) if rni_match else "",
                    rank=rank,
                    points=points,
                    seed=int(seed_value) if seed_value is not None else None,
                    affiliation=(
                        values[affiliation_index]
                        if affiliation_index is not None and affiliation_index < len(values)
                        else ""
                    ),
                    request_status=(
                        values[status_index]
                        if status_index is not None and status_index < len(values)
                        else ""
                    ),
                    raw=raw,
                )
            )

    # Some RTT views use card rows rather than a table.  Player anchors are a
    # safe fallback, though they do not expose ranking columns.
    if not result:
        for href, text in parsed.anchors:
            source_id = _source_id_from_href(href)
            if source_id and text:
                result.append(ParsedPlayer(name=text, source_id=source_id))

    unique: dict[str, ParsedPlayer] = {}
    for player in result:
        key = player.rni or player.source_id or normalize_name(player.name)
        if key and key not in unique:
            unique[key] = player
    return list(unique.values())


def _score_winner_side(score: str) -> int | None:
    text = normalize_space(score)
    leading = re.match(r"^(\d+)\s*[-–—:]\s*(\d+)(?:\s|$)", text)
    if leading:
        left, right = map(int, leading.groups())
        if left != right:
            return 1 if left > right else 2
    set_scores = re.findall(r"(\d+)\s*[-–—:]\s*(\d+)", text)
    left_sets = sum(int(left) > int(right) for left, right in set_scores)
    right_sets = sum(int(right) > int(left) for left, right in set_scores)
    if left_sets != right_sets:
        return 1 if left_sets > right_sets else 2
    return None


def parse_matches(html: str) -> list[ParsedMatch]:
    parsed = parse_rendered_html(html)
    matches: list[ParsedMatch] = []
    for table in parsed.tables:
        if not table:
            continue
        headers = [cell.text for cell in table[0]]
        player1_index = _column_index(headers, ("участник1", "игрок1", "player1"))
        player2_index = _column_index(headers, ("участник2", "игрок2", "player2"))
        score_index = _column_index(headers, ("счет", "счёт", "результат", "score"))
        round_index = _column_index(headers, ("этап", "стадия", "round"))
        status_index = _column_index(headers, ("статус", "status"))
        winner_index = _column_index(headers, ("победитель", "winner"))
        if player1_index is None or player2_index is None:
            continue
        for row in table[1:]:
            if max(player1_index, player2_index) >= len(row):
                continue
            player1 = row[player1_index].text
            player2 = row[player2_index].text
            if not player1 or not player2:
                continue
            score = row[score_index].text if score_index is not None and score_index < len(row) else ""
            winner = row[winner_index].text if winner_index is not None and winner_index < len(row) else ""
            if not winner:
                side = _score_winner_side(score)
                winner = player1 if side == 1 else player2 if side == 2 else ""
            matches.append(
                ParsedMatch(
                    player1=player1,
                    player2=player2,
                    winner=winner,
                    score=score,
                    round_name=(row[round_index].text if round_index is not None and round_index < len(row) else ""),
                    status=(row[status_index].text if status_index is not None and status_index < len(row) else ""),
                )
            )
    return matches


def _player_id_from_grid_text(value: str, players: Sequence[ParsedPlayer]) -> str | None:
    tokens = re.findall(r"[0-9a-zа-я]+", normalize_name(value))
    if not tokens or any(token in {"x", "bye", "свободен"} for token in tokens):
        return None
    matches: list[str] = []
    for player in players:
        player_tokens = re.findall(r"[0-9a-zа-я]+", normalize_name(player.name))
        if len(player_tokens) < 2:
            continue
        surname = player_tokens[0]
        expected_initials = tuple(token[0] for token in player_tokens[1:3])
        for index, token in enumerate(tokens):
            following = tokens[index + 1:index + 1 + len(expected_initials)]
            if (
                token == surname
                and len(following) == len(expected_initials)
                and tuple(part[0] for part in following) == expected_initials
            ):
                matches.append(player.player_id)
                break
    unique_matches = list(dict.fromkeys(matches))
    return unique_matches[0] if len(unique_matches) == 1 else ""


def _parse_div_grid_rounds(
    parsed: _RenderedHTMLParser,
    players: Sequence[ParsedPlayer],
) -> list[list[str | None]]:
    if 0 not in parsed.grid_round_cells:
        return []
    player_ids = {player.player_id for player in players}
    rounds: list[list[str | None]] = []
    expected_size = 0
    for round_index in range(max(parsed.grid_round_cells) + 1):
        cells = parsed.grid_round_cells.get(round_index)
        if cells is None:
            break
        slots: list[str | None] = []
        ambiguous = False
        for top, bottom in cells:
            for value in (top, bottom):
                player_id = _player_id_from_grid_text(value, players)
                if player_id == "":
                    ambiguous = True
                    break
                slots.append(player_id)
            if ambiguous:
                break
        if ambiguous:
            break
        if round_index == 0:
            expected_size = len(slots)
            valid_size = expected_size in (4, 8, 16, 32)
        else:
            expected_size //= 2
            valid_size = len(slots) == expected_size
        present = [value for value in slots if value is not None]
        valid_players = (
            len(present) == len(set(present))
            and set(present).issubset(player_ids)
            and (round_index > 0 or set(present) == player_ids)
        )
        if not valid_size or not valid_players:
            break
        rounds.append(slots)
    return rounds


def parse_grid_rounds(html: str, players: Sequence[ParsedPlayer]) -> list[list[str | None]]:
    """Parse every published column of an RTT Olympic draw."""

    return _parse_div_grid_rounds(parse_rendered_html(html), players)


def parse_grid_slots(html: str, players: Sequence[ParsedPlayer]) -> list[str | None]:
    """Parse the initial slots of an RTT draw."""

    parsed = parse_rendered_html(html)
    by_name = {normalize_name(player.name): player.player_id for player in players}
    rounds = _parse_div_grid_rounds(parsed, players)
    if rounds:
        return rounds[0]

    for table in parsed.tables:
        rows: dict[int, str | None] = {}
        for row in table:
            row_text = " | ".join(cell.text for cell in row)
            row_number = re.match(r"^\s*(\d{1,2})(?:\s|\||$)", row_text)
            if not row_number:
                continue
            number = int(row_number.group(1))
            normalized_row = normalize_name(row_text)
            player_id = next(
                (player_id for name, player_id in by_name.items() if name and name in normalized_row),
                None,
            )
            if player_id is not None or re.search(r"\b(?:x|bye)\b", normalized_row):
                rows[number] = player_id
        if rows and max(rows) in (4, 8, 16, 32) and set(rows) == set(range(1, max(rows) + 1)):
            slots = [rows[index] for index in range(1, max(rows) + 1)]
            present = [value for value in slots if value is not None]
            if len(present) == len(set(present)) and set(present) == set(by_name.values()):
                return slots
    return []


def tournament_eligibility(
    status: object,
    start_date: object,
    end_date: object,
    *,
    today: date | None = None,
) -> tuple[bool, str]:
    today = today or date.today()
    normalized_status = normalize_name(status)
    terminal_markers = ("заверш", "сдача отч", "отмен", "не состоя", "аннулир")
    if any(marker in normalized_status for marker in terminal_markers):
        return False, f"Статус турнира «{normalize_space(status)}» уже не допускает прогноз мест."
    try:
        start = date.fromisoformat(str(start_date)) if start_date else None
        end = date.fromisoformat(str(end_date)) if end_date else None
    except ValueError:
        start = end = None
    if end and end < today:
        return False, "Турнир уже закончился по календарным датам."
    if start and start > today:
        return True, "Турнир ещё не начался."
    if start and (end is None or today <= end):
        return True, "Турнир находится в периоде проведения."
    pre_or_live_markers = (
        "заяв", "регистрац", "формирован", "жереб", "готов", "провод", "процесс", "идет", "идёт"
    )
    if any(marker in normalized_status for marker in pre_or_live_markers):
        return True, "Статус соответствует подготовке или проведению турнира."
    return False, "Не удалось подтвердить, что турнир ещё не начался или сейчас проводится."


def _request_priority(player: ParsedPlayer) -> tuple[int, float, float, str]:
    status = normalize_name(player.request_status)
    rejected = any(marker in status for marker in ("отказ", "снят", "отозв", "не допущ"))
    accepted = any(marker in status for marker in ("основ", "допущ", "принят", "участник"))
    rank = player.rank if player.rank is not None else float("inf")
    points = player.points if player.points is not None else float("-inf")
    return (2 if rejected else 0 if accepted else 1, rank, -points, normalize_name(player.name))


def project_main_draw(
    requests: Sequence[ParsedPlayer],
    capacity: int | None,
) -> tuple[list[ParsedPlayer], list[str]]:
    active = [
        player
        for player in requests
        if not any(
            marker in normalize_name(player.request_status)
            for marker in ("отказ", "снят", "отозв", "не допущ")
        )
    ]
    if capacity is None:
        capacity = min(32, len(active))
    capacity = max(0, min(32, int(capacity)))
    projected = sorted(active, key=_request_priority)[:capacity]
    notes = [
        f"Состав ОТ спрогнозирован из заявок: выбрано {len(projected)} из {len(active)} активных заявок."
    ]
    if len(active) > capacity:
        notes.append("Приоритет рассчитан по статусу заявки, месту рейтинга и очкам.")
    return projected, notes


def build_snapshot_from_pages(
    tour_id: str,
    pages: Mapping[str, str],
    *,
    fetched_at: str | None = None,
    page_paths: Mapping[str, str] | None = None,
    today: date | None = None,
) -> TournamentSnapshot:
    dashboard_html = pages.get("dashboard", "") or pages.get("matches", "")
    metadata = parse_metadata(dashboard_html, str(tour_id)) if dashboard_html else {}
    members = parse_players(pages.get("members", "")) if pages.get("members") else []
    requests = parse_players(pages.get("requests", "")) if pages.get("requests") else []
    warnings: list[str] = []

    if len(members) >= 4:
        selected = members
        player_source = "official_members"
    else:
        selected, projection_notes = project_main_draw(requests, metadata.get("main_draw_capacity"))
        warnings.extend(projection_notes)
        player_source = "projected_from_requests"

    if len(selected) > 32:
        selected = sorted(selected, key=_request_priority)[:32]
        warnings.append("Для ОТ оставлены первые 32 игрока; остальные относятся к отбору/ожиданию.")
    if selected and len(selected) < 4:
        warnings.append("Для моделирования олимпийской сетки нужно не менее четырёх игроков.")

    eligible, eligibility_reason = tournament_eligibility(
        metadata.get("status", ""),
        metadata.get("start_date", ""),
        metadata.get("end_date", ""),
        today=today,
    )
    grid_rounds = parse_grid_rounds(pages.get("grid", ""), selected) if pages.get("grid") and selected else []
    grid_slots = (
        grid_rounds[0]
        if grid_rounds
        else parse_grid_slots(pages.get("grid", ""), selected)
        if pages.get("grid") and selected
        else []
    )
    completed = parse_matches(pages.get("matches", "")) if pages.get("matches") else []
    if grid_slots:
        player_source += "+official_grid"

    return TournamentSnapshot(
        tour_id=str(tour_id),
        fetched_at=fetched_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        players=selected,
        player_source=player_source,
        grid_slots=grid_slots,
        grid_rounds=grid_rounds,
        completed_matches=completed,
        eligible=eligible,
        eligibility_reason=eligibility_reason,
        warnings=warnings,
        page_paths=dict(page_paths or {}),
        **metadata,
    )


def snapshot_path(cache_dir: Path | str, tour_id: str) -> Path:
    return Path(cache_dir) / str(tour_id) / "snapshot.json"


def save_snapshot(snapshot: TournamentSnapshot, cache_dir: Path | str) -> Path:
    path = snapshot_path(cache_dir, snapshot.tour_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)
    return path


def load_snapshot(cache_dir: Path | str, tour_id: str) -> TournamentSnapshot:
    path = snapshot_path(cache_dir, str(tour_id))
    if not path.exists():
        raise FileNotFoundError(f"Нет кэша нового режима для турнира {tour_id}: {path}")
    snapshot = TournamentSnapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if not snapshot.title and not snapshot.status and not snapshot.players:
        raise ValueError(
            f"Кэш турнира {tour_id} неполный: RTT вернул оболочку страницы без данных турнира."
        )
    return snapshot


def read_cached_pages(cache_dir: Path | str, tour_id: str) -> tuple[dict[str, str], dict[str, str]]:
    tour_dir = Path(cache_dir) / str(tour_id)
    pages: dict[str, str] = {}
    paths: dict[str, str] = {}
    for route in TOURNAMENT_ROUTES:
        key = route.rstrip("/")
        path = tour_dir / f"{key}.html"
        if path.exists():
            pages[key] = path.read_text(encoding="utf-8", errors="ignore")
            paths[key] = str(path)
    return pages, paths
