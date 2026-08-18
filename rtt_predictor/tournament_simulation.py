"""RTT Olympic-draw construction and Monte Carlo tournament simulation.

The module is deliberately independent from the data-refresh pipeline.  It accepts
already parsed players and a match-probability callback, so a notebook can use the
production model without changing its bundle or training data.

Draw placement follows sections 13.3.1--13.3.3 and table 6 of the 2026 RTT
regulations: 2/4/8 seeds for fields of 4--8/9--16/17--32 players, fixed sections
for seed groups, and byes placed next to seeds before the remaining seeded
sections are considered.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import math
import random
import re
from typing import Callable, Iterable, Mapping, Sequence


RTT_REGULATIONS_URL = (
    "https://tennis-russia.ru/upload/custom/9aa/"
    "583vovi2qi73yq2jxecj4xq1o515kyco.pdf"
)
RTT_POINTS_TABLES_URL = (
    "https://tennis-russia.ru/upload/custom/b7d/"
    "nnzb6lwb50eqia1dig5qmeehyhnsy58c.pdf"
)


@dataclass(frozen=True, slots=True)
class TournamentPlayer:
    """A player participating in a simulated RTT singles draw."""

    player_id: str
    name: str
    rank: float | None = None
    points: float | None = None
    seed: int | None = None
    affiliation: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True, slots=True)
class OutcomeBand:
    """One finish band in a standard Olympic draw."""

    index: int
    label: str
    lower_place: int
    upper_place: int


@dataclass(slots=True)
class SimulationResult:
    """Aggregated Monte Carlo output for one tournament."""

    iterations: int
    actual_player_count: int
    draw_size: int
    distributions: dict[str, dict[str, float]]
    outcome_points: dict[str, int | None]
    expected_points: dict[str, float | None]
    encounter_probabilities: dict[str, dict[str, float]]
    player_names: dict[str, str]
    draw_is_fixed: bool
    warnings: list[str] = field(default_factory=list)

    def player_distribution(self, player_id: str) -> list[dict[str, object]]:
        """Return a presentation-ready, correctly ordered distribution."""

        distribution = self.distributions.get(str(player_id), {})
        rows = []
        for label, probability in distribution.items():
            lower, upper = parse_place_label(label)
            rows.append(
                {
                    "place": label,
                    "lower_place": lower,
                    "upper_place": upper,
                    "probability": probability,
                    "points": self.outcome_points.get(label),
                    "expected_points_contribution": (
                        probability * self.outcome_points[label]
                        if self.outcome_points.get(label) is not None
                        else None
                    ),
                }
            )
        return sorted(rows, key=lambda row: (int(row["lower_place"]), int(row["upper_place"])))


ProbabilityProvider = Callable[[TournamentPlayer, TournamentPlayer], float]


def _finite_number(value: float | None, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _player_order_key(player: TournamentPlayer) -> tuple[float, float, str, str]:
    explicit_seed = _finite_number(player.seed, math.inf)
    rank = _finite_number(player.rank, math.inf)
    points = _finite_number(player.points, -math.inf)
    return explicit_seed, rank, -points, player.name.casefold()


def draw_size_for(player_count: int) -> int:
    """Return the RTT Olympic bracket size for 4--32 actual participants."""

    if not 4 <= int(player_count) <= 32:
        raise ValueError("RTT Olympic simulation supports 4 to 32 participants.")
    for size in (4, 8, 16, 32):
        if player_count <= size:
            return size
    raise AssertionError("unreachable")


def points_field_size_for(player_count: int) -> int:
    """Return the field-size row used by RTT points tables (8/16/24/32)."""

    if not 4 <= int(player_count) <= 32:
        raise ValueError("RTT points tables support 4 to 32 participants.")
    if player_count <= 8:
        return 8
    if player_count <= 16:
        return 16
    if player_count <= 24:
        return 24
    return 32


def seed_count_for(player_count: int) -> int:
    """Seed count from table 6 of the 2026 RTT regulations."""

    size = draw_size_for(player_count)
    return {4: 2, 8: 2, 16: 4, 32: 8}[size]


def seed_slot_groups(draw_size: int) -> tuple[tuple[int, ...], ...]:
    """Zero-based row groups for seeds 1--2, 3--4 and 5--8."""

    groups = {
        4: ((0, 3),),
        8: ((0, 7),),
        16: ((0, 15), (4, 11)),
        32: ((0, 31), (8, 23), (7, 15, 16, 24)),
    }
    try:
        return groups[int(draw_size)]
    except KeyError as exc:
        raise ValueError("Draw size must be one of 4, 8, 16 or 32.") from exc


def _virtual_seed_slots(draw_size: int, rng: random.Random) -> list[int]:
    """Assign all designated seed numbers to legal RTT rows."""

    result: list[int] = []
    for group_index, group in enumerate(seed_slot_groups(draw_size)):
        rows = list(group)
        if group_index > 0:
            rng.shuffle(rows)
        result.extend(rows)
    return result


def _opposite_slot(row: int) -> int:
    return row ^ 1


def _nearby_match_bye_slot(seed_row: int) -> int:
    """Return the nearest row in the adjacent first-round match."""

    match_index = seed_row // 2
    adjacent_match = match_index + 1 if match_index % 2 == 0 else match_index - 1
    adjacent_rows = (2 * adjacent_match, 2 * adjacent_match + 1)
    return min(adjacent_rows, key=lambda row: (abs(row - seed_row), row))


def _bye_rows(
    draw_size: int,
    bye_count: int,
    virtual_seed_rows: Sequence[int],
    actual_seed_count: int,
) -> list[int]:
    """Place X rows according to RTT section 13.3.2.

    Present seeds receive X first in seed order.  If the field has fewer rated
    seeds than table 6 provides for, their designated virtual sections are used
    next.  Remaining X go into matches adjacent to seeded sections in reverse
    seed order.
    """

    if bye_count <= 0:
        return []

    present = list(virtual_seed_rows[:actual_seed_count])
    missing = list(virtual_seed_rows[actual_seed_count:])
    first_pass = [_opposite_slot(row) for row in present]
    first_pass.extend(_opposite_slot(row) for row in reversed(missing))
    second_pass = [_nearby_match_bye_slot(row) for row in reversed(virtual_seed_rows)]

    rows: list[int] = []
    for row in [*first_pass, *second_pass]:
        if row not in rows:
            rows.append(row)
        if len(rows) == bye_count:
            return rows
    raise ValueError("Could not allocate all RTT byes.")


def build_rtt_draw(
    players: Sequence[TournamentPlayer],
    *,
    rng: random.Random | None = None,
) -> list[TournamentPlayer | None]:
    """Build one legal RTT Olympic draw, including seed separation and X rows."""

    rng = rng or random.Random()
    players = list(players)
    ids = [str(player.player_id) for player in players]
    if len(ids) != len(set(ids)):
        raise ValueError("player_id values must be unique within a tournament.")

    draw_size = draw_size_for(len(players))
    required_seed_count = seed_count_for(len(players))
    ordered = sorted(players, key=_player_order_key)

    # Explicit seed values take priority.  Otherwise positive RTT points/ranks
    # determine the projected seeds.  Unrated players remain in the common draw.
    explicitly_seeded = [player for player in ordered if player.seed is not None]
    if explicitly_seeded:
        seeded = explicitly_seeded[:required_seed_count]
    else:
        rated = [
            player
            for player in ordered
            if (
                _finite_number(player.points, 0.0) > 0
                or math.isfinite(_finite_number(player.rank, math.inf))
            )
        ]
        seeded = rated[:required_seed_count]

    virtual_rows = _virtual_seed_slots(draw_size, rng)
    draw: list[TournamentPlayer | None] = [None] * draw_size
    for player, row in zip(seeded, virtual_rows):
        draw[row] = player

    bye_count = draw_size - len(players)
    bye_rows = set(_bye_rows(draw_size, bye_count, virtual_rows, len(seeded)))
    occupied_seed_rows = {row for row, player in enumerate(draw) if player is not None}
    if bye_rows & occupied_seed_rows:
        raise AssertionError("A bye cannot replace a seeded player.")

    unseeded = [player for player in players if player not in set(seeded)]
    rng.shuffle(unseeded)
    free_rows = [
        row
        for row in range(draw_size)
        if row not in occupied_seed_rows and row not in bye_rows
    ]
    if len(free_rows) != len(unseeded):
        raise AssertionError("Draw allocation produced an invalid number of free rows.")
    for row, player in zip(free_rows, unseeded):
        draw[row] = player
    return draw


def _band_for_loss(draw_size: int, actual_count: int, round_index: int) -> OutcomeBand:
    bracket_upper = draw_size // (2**round_index)
    lower = draw_size // (2 ** (round_index + 1)) + 1
    upper = min(actual_count, bracket_upper)
    label = str(lower) if lower == upper else f"{lower}\u2013{upper}"
    return OutcomeBand(index=int(math.log2(draw_size)) - round_index, label=label, lower_place=lower, upper_place=upper)


def parse_place_label(label: str) -> tuple[int, int]:
    values = [int(value) for value in re.findall(r"\d+", str(label))]
    if not values:
        raise ValueError(f"Invalid place label: {label!r}")
    return values[0], values[-1]


def normalize_age_group(value: object) -> str:
    text = " ".join(str(value or "").replace("\xa0", " ").casefold().split())
    if "15" in text:
        return "до 15 лет"
    if "17" in text:
        return "до 17 лет"
    if "19" in text:
        return "до 19 лет"
    if "взрос" in text or "adult" in text:
        return "взрослые"
    return text


def normalize_tournament_category(value: object) -> str:
    text = " ".join(str(value or "").replace("\xa0", " ").upper().split())
    text = text.replace("A", "А").replace("B", "Б").replace("C", "В")
    text = text.replace("III", "3").replace("II", "2").replace("IV", "4").replace("VI", "6").replace("V", "5").replace("I", "1")
    text = text.replace(" ", "").replace("-", "")
    if text in {"ФТ", "FT"}:
        return "ФТ"
    match = re.search(r"([1-6])([АБВГС])", text)
    if not match:
        return text
    suffix = "В" if match.group(2) == "С" else match.group(2)
    return f"{match.group(1)}{suffix}"


# Winner points for the ordinary Olympic system in official RTT tables 4--7.
# Tuple order is the available field-size row: 32 only for high categories,
# otherwise 32/24/16 or 32/24/16/8.
_WINNER_POINTS: dict[str, dict[str, dict[int, int]]] = {
    "взрослые": {
        "ФТ": {32: 1500}, "1А": {32: 1000}, "1Б": {32: 900},
        "2А": {32: 800}, "2Б": {32: 700},
        "3А": {32: 600, 24: 450, 16: 300},
        "3Б": {32: 550, 24: 388, 16: 275},
        "3В": {32: 500, 24: 375, 16: 250},
        "4А": {32: 450, 24: 337, 16: 225, 8: 180},
        "4Б": {32: 400, 24: 300, 16: 200, 8: 140},
        "4В": {32: 350, 24: 262, 16: 175, 8: 123},
        "5А": {32: 300, 24: 225, 16: 150, 8: 105},
        "5Б": {32: 250, 24: 187, 16: 125, 8: 88},
        "5В": {32: 200, 24: 150, 16: 100, 8: 70},
    },
    "до 19 лет": {
        "ФТ": {32: 900}, "1А": {32: 700}, "1Б": {32: 650},
        "2А": {32: 550}, "2Б": {32: 500},
        "3А": {32: 400, 24: 300, 16: 200},
        "3Б": {32: 350, 24: 262, 16: 175},
        "3В": {32: 300, 24: 225, 16: 150},
        "4А": {32: 260, 24: 195, 16: 130, 8: 90},
        "4Б": {32: 230, 24: 172, 16: 115, 8: 80},
        "4В": {32: 210, 24: 157, 16: 105, 8: 75},
        "5А": {32: 180, 24: 135, 16: 90, 8: 64},
        "5Б": {32: 150, 24: 112, 16: 75, 8: 52},
        "5В": {32: 120, 24: 90, 16: 60, 8: 42},
    },
    "до 17 лет": {
        "ФТ": {32: 550}, "1А": {32: 450}, "1Б": {32: 430},
        "2А": {32: 400}, "2Б": {32: 350},
        "3А": {32: 300, 24: 225, 16: 150},
        "3Б": {32: 280, 24: 210, 16: 140},
        "3В": {32: 250, 24: 187, 16: 125},
        "4А": {32: 200, 24: 150, 16: 100, 8: 68},
        "4Б": {32: 180, 24: 135, 16: 90, 8: 64},
        "4В": {32: 150, 24: 112, 16: 75, 8: 52},
        "5А": {32: 120, 24: 90, 16: 60, 8: 42},
        "5Б": {32: 100, 24: 75, 16: 50, 8: 36},
        "5В": {32: 90, 24: 67, 16: 45, 8: 32},
    },
    "до 15 лет": {
        "ФТ": {32: 350}, "1А": {32: 250}, "1Б": {32: 230},
        "2А": {32: 200}, "2Б": {32: 180},
        "3А": {32: 150, 24: 112, 16: 75},
        "3Б": {32: 130, 24: 97, 16: 65},
        "3В": {32: 120, 24: 90, 16: 60},
        "4А": {32: 100, 24: 75, 16: 50, 8: 34},
        "4Б": {32: 90, 24: 67, 16: 45, 8: 32},
        "4В": {32: 85, 24: 64, 16: 43, 8: 30},
        "5А": {32: 75, 24: 56, 16: 38, 8: 26},
        "5Б": {32: 70, 24: 52, 16: 35, 8: 25},
        "5В": {32: 60, 24: 45, 16: 30, 8: 21},
    },
}


_OLYMPIC_STAGE_MULTIPLIERS: dict[int, tuple[Decimal, ...]] = {
    8: tuple(map(Decimal, ("1", "0.7", "0.5", "0.25"))),
    16: tuple(map(Decimal, ("1", "0.7", "0.5", "0.35", "0.225"))),
    # 0.1435 is the official-table average for places 17--24.
    24: tuple(map(Decimal, ("1", "0.7", "0.5", "0.35", "0.225", "0.1435"))),
    32: tuple(map(Decimal, ("1", "0.7", "0.5", "0.35", "0.225", "0.125"))),
}


def ordinary_olympic_points(
    age_group: object,
    tournament_category: object,
    actual_player_count: int,
) -> dict[int, int]:
    """Return ``outcome-band index -> points`` from official RTT tables 4--7.

    Band index 0 is champion, 1 finalist, 2 places 3--4, and so on.  The
    function covers the four age groups present in the predictor dataset.
    """

    age = normalize_age_group(age_group)
    category = normalize_tournament_category(tournament_category)
    table_size = points_field_size_for(actual_player_count)
    try:
        sizes = _WINNER_POINTS[age][category]
    except KeyError as exc:
        raise KeyError(
            f"No RTT points table for age={age!r}, category={category!r}."
        ) from exc
    if table_size not in sizes:
        # High categories are published for a 32-player field only.
        if len(sizes) == 1 and 32 in sizes:
            table_size = 32
        else:
            available = ", ".join(map(str, sorted(sizes)))
            raise KeyError(
                f"No {table_size}-player RTT points row for age={age!r}, "
                f"category={category!r}; available: {available}."
            )

    winner_points = Decimal(sizes[table_size])
    return {
        index: int((winner_points * multiplier).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        for index, multiplier in enumerate(_OLYMPIC_STAGE_MULTIPLIERS[table_size])
    }


def points_for_outcome_bands(
    age_group: object,
    tournament_category: object,
    actual_player_count: int,
    draw_size: int | None = None,
) -> dict[str, int]:
    """Return display-place labels and points for the simulated draw."""

    draw_size = draw_size or draw_size_for(actual_player_count)
    by_index = ordinary_olympic_points(age_group, tournament_category, actual_player_count)
    labels: dict[str, int] = {"1": by_index[0]}
    rounds = int(math.log2(draw_size))
    for round_index in reversed(range(rounds)):
        band = _band_for_loss(draw_size, actual_player_count, round_index)
        if band.index in by_index:
            labels[band.label] = by_index[band.index]
    return labels


def _validate_fixed_draw(
    fixed_draw: Sequence[TournamentPlayer | None],
    players_by_id: Mapping[str, TournamentPlayer],
) -> list[TournamentPlayer | None]:
    draw = list(fixed_draw)
    if len(draw) not in (4, 8, 16, 32):
        raise ValueError("fixed_draw must contain 4, 8, 16 or 32 rows.")
    seen: set[str] = set()
    normalized: list[TournamentPlayer | None] = []
    for player in draw:
        if player is None:
            normalized.append(None)
            continue
        player_id = str(player.player_id)
        if player_id not in players_by_id:
            raise ValueError(f"fixed_draw contains unknown player_id {player_id!r}.")
        if player_id in seen:
            raise ValueError(f"fixed_draw contains duplicate player_id {player_id!r}.")
        seen.add(player_id)
        normalized.append(players_by_id[player_id])
    if seen != set(players_by_id):
        missing = sorted(set(players_by_id) - seen)
        raise ValueError(f"fixed_draw is missing players: {missing!r}.")
    return normalized


def simulate_tournament(
    players: Sequence[TournamentPlayer],
    probability_provider: ProbabilityProvider,
    *,
    iterations: int = 20_000,
    random_seed: int | None = 2026,
    fixed_draw: Sequence[TournamentPlayer | None] | None = None,
    completed_winners: Mapping[frozenset[str], str] | None = None,
    target_player_ids: Iterable[str] | None = None,
    age_group: object | None = None,
    tournament_category: object | None = None,
) -> SimulationResult:
    """Simulate an RTT Olympic tournament and aggregate finish probabilities.

    ``completed_winners`` locks already played matches.  Its key is a frozenset
    with two player ids and its value is the winner id.  When ``fixed_draw`` is
    omitted, a fresh legal RTT draw is generated for every iteration, which is
    the intended mode before the official draw has been published.
    """

    if int(iterations) <= 0:
        raise ValueError("iterations must be positive.")
    players = list(players)
    players_by_id = {str(player.player_id): player for player in players}
    if len(players_by_id) != len(players):
        raise ValueError("player_id values must be unique within a tournament.")
    actual_count = len(players)
    expected_draw_size = draw_size_for(actual_count)
    rng = random.Random(random_seed)

    normalized_fixed_draw = None
    if fixed_draw is not None:
        normalized_fixed_draw = _validate_fixed_draw(fixed_draw, players_by_id)
        if len(normalized_fixed_draw) != expected_draw_size:
            raise ValueError(
                f"fixed_draw has {len(normalized_fixed_draw)} rows; "
                f"{actual_count} players require {expected_draw_size}."
            )

    targets = (
        {str(player_id) for player_id in target_player_ids}
        if target_player_ids is not None
        else set(players_by_id)
    )
    unknown_targets = targets - set(players_by_id)
    if unknown_targets:
        raise ValueError(f"Unknown target player ids: {sorted(unknown_targets)!r}.")

    locked = dict(completed_winners or {})
    for pair, winner_id in locked.items():
        if len(pair) != 2 or str(winner_id) not in pair:
            raise ValueError("Each completed_winners value must be one player from its pair key.")

    finish_counts: dict[str, Counter[str]] = {player_id: Counter() for player_id in targets}
    encounter_counts: dict[str, Counter[str]] = {player_id: Counter() for player_id in targets}

    for _ in range(int(iterations)):
        current = (
            list(normalized_fixed_draw)
            if normalized_fixed_draw is not None
            else build_rtt_draw(players, rng=rng)
        )
        round_index = 0
        while len(current) > 1:
            next_round: list[TournamentPlayer | None] = []
            for position in range(0, len(current), 2):
                player_a = current[position]
                player_b = current[position + 1]
                if player_a is None and player_b is None:
                    next_round.append(None)
                    continue
                if player_a is None or player_b is None:
                    next_round.append(player_b if player_a is None else player_a)
                    continue

                id_a = str(player_a.player_id)
                id_b = str(player_b.player_id)
                pair = frozenset((id_a, id_b))
                locked_winner = locked.get(pair)
                if locked_winner is not None:
                    winner = player_a if str(locked_winner) == id_a else player_b
                else:
                    # Completed matches are replayed only to advance their known
                    # winner through the bracket.  They are not possible future
                    # encounters and must not appear in the opponents forecast.
                    if id_a in targets:
                        encounter_counts[id_a][id_b] += 1
                    if id_b in targets:
                        encounter_counts[id_b][id_a] += 1
                    probability_a = float(probability_provider(player_a, player_b))
                    if not math.isfinite(probability_a) or not 0.0 <= probability_a <= 1.0:
                        raise ValueError(
                            f"Invalid win probability {probability_a!r} for {id_a!r} vs {id_b!r}."
                        )
                    winner = player_a if rng.random() < probability_a else player_b
                loser = player_b if winner is player_a else player_a
                loser_id = str(loser.player_id)
                if loser_id in targets:
                    band = _band_for_loss(expected_draw_size, actual_count, round_index)
                    finish_counts[loser_id][band.label] += 1
                next_round.append(winner)
            current = next_round
            round_index += 1

        champion = current[0]
        if champion is None:
            raise AssertionError("Simulation ended without a champion.")
        champion_id = str(champion.player_id)
        if champion_id in targets:
            finish_counts[champion_id]["1"] += 1

    warnings: list[str] = []
    points_by_label: dict[str, int | None] = {}
    if age_group is not None and tournament_category is not None:
        try:
            points_by_label.update(
                points_for_outcome_bands(
                    age_group,
                    tournament_category,
                    actual_count,
                    expected_draw_size,
                )
            )
        except KeyError as exc:
            warnings.append(str(exc))

    possible_labels = {"1"}
    possible_labels.update(
        _band_for_loss(expected_draw_size, actual_count, round_index).label
        for round_index in range(int(math.log2(expected_draw_size)))
    )
    for label in possible_labels:
        points_by_label.setdefault(label, None)

    distributions = {
        player_id: {
            label: counter.get(label, 0) / int(iterations)
            for label in possible_labels
        }
        for player_id, counter in finish_counts.items()
    }
    expected_points: dict[str, float | None] = {}
    for player_id, distribution in distributions.items():
        if not distribution or any(points_by_label.get(label) is None for label in distribution):
            expected_points[player_id] = None
        else:
            expected_points[player_id] = sum(
                probability * int(points_by_label[label])
                for label, probability in distribution.items()
            )

    encounter_probabilities = {
        player_id: {
            opponent_id: count / int(iterations)
            for opponent_id, count in counter.items()
        }
        for player_id, counter in encounter_counts.items()
    }
    return SimulationResult(
        iterations=int(iterations),
        actual_player_count=actual_count,
        draw_size=expected_draw_size,
        distributions=distributions,
        outcome_points=points_by_label,
        expected_points=expected_points,
        encounter_probabilities=encounter_probabilities,
        player_names={player_id: player.name for player_id, player in players_by_id.items()},
        draw_is_fixed=normalized_fixed_draw is not None,
        warnings=warnings,
    )


class PairProbabilityMatrix:
    """Fast symmetric probability provider for precomputed H2H forecasts."""

    def __init__(self, probabilities: Mapping[tuple[str, str], float]):
        self._probabilities = {
            (str(player_a), str(player_b)): float(probability)
            for (player_a, player_b), probability in probabilities.items()
        }

    def __call__(self, player_a: TournamentPlayer, player_b: TournamentPlayer) -> float:
        id_a = str(player_a.player_id)
        id_b = str(player_b.player_id)
        if (id_a, id_b) in self._probabilities:
            return self._probabilities[(id_a, id_b)]
        if (id_b, id_a) in self._probabilities:
            return 1.0 - self._probabilities[(id_b, id_a)]
        raise KeyError(f"No precomputed probability for {id_a!r} vs {id_b!r}.")
