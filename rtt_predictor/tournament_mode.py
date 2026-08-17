"""High-level orchestration for player and mega tournament analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from itertools import combinations
from pathlib import Path
import re
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from scripts import prediction_runtime as prediction

from .tournament_data import (
    ParsedPlayer,
    TournamentSnapshot,
    load_snapshot,
    normalize_name,
    project_main_draw,
)
from .tournament_simulation import (
    PairProbabilityMatrix,
    SimulationResult,
    TournamentPlayer,
    normalize_age_group,
    simulate_tournament,
)


ProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class TournamentAnalysis:
    snapshot: TournamentSnapshot
    target_player: TournamentPlayer
    players: list[TournamentPlayer]
    simulation: SimulationResult
    probability_matrix: dict[tuple[str, str], float]
    prediction_date: pd.Timestamp
    warnings: list[str]

    def distribution_table(self) -> pd.DataFrame:
        table = pd.DataFrame(self.simulation.player_distribution(self.target_player.player_id))
        if table.empty:
            return table
        table["probability_pct"] = 100.0 * table["probability"]
        return table[
            [
                "place",
                "probability",
                "probability_pct",
                "points",
                "expected_points_contribution",
            ]
        ]

    def opponents_table(self) -> pd.DataFrame:
        probabilities = self.simulation.encounter_probabilities.get(self.target_player.player_id, {})
        rows = []
        for opponent_id, encounter_probability in probabilities.items():
            opponent_name = self.simulation.player_names.get(opponent_id, opponent_id)
            h2h_probability = self.probability_matrix.get((self.target_player.player_id, opponent_id))
            if h2h_probability is None and (opponent_id, self.target_player.player_id) in self.probability_matrix:
                h2h_probability = 1.0 - self.probability_matrix[(opponent_id, self.target_player.player_id)]
            rows.append(
                {
                    "opponent_id": opponent_id,
                    "opponent": opponent_name,
                    "encounter_probability": encounter_probability,
                    "encounter_probability_pct": 100.0 * encounter_probability,
                    "target_h2h_win_probability": h2h_probability,
                    "target_h2h_win_probability_pct": (
                        100.0 * h2h_probability if h2h_probability is not None else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values("encounter_probability", ascending=False).reset_index(drop=True)


def _model_directory(bundle: dict) -> tuple[set[str], dict[str, list[tuple[pd.Timestamp, str, str]]]]:
    long_feat = bundle["long_feat"]
    known_ids = set(long_feat["player_id"].astype(str))
    by_name: dict[str, list[tuple[pd.Timestamp, str, str]]] = {}
    for row in long_feat[["player_id", "player_name", "match_date"]].dropna(subset=["player_id"]).itertuples(index=False):
        key = prediction.normalize_player_name(row.player_name)
        by_name.setdefault(key, []).append((pd.Timestamp(row.match_date), str(row.player_id), str(row.player_name)))
    for values in by_name.values():
        values.sort(key=lambda item: item[0])
    return known_ids, by_name


def resolve_players_for_model(
    bundle: dict,
    players: Sequence[TournamentPlayer],
) -> tuple[list[TournamentPlayer], dict[str, str], list[str]]:
    """Map RTT page identities to bundle ids while retaining never-seen players."""

    known_ids, by_name = _model_directory(bundle)
    resolved: list[TournamentPlayer] = []
    source_to_model: dict[str, str] = {}
    warnings: list[str] = []
    used_ids: set[str] = set()

    for player in players:
        source_id = str(player.player_id)
        model_id = source_id if source_id in known_ids else ""
        canonical_name = player.name
        if not model_id:
            exact = by_name.get(prediction.normalize_player_name(player.name), [])
            if exact:
                _, model_id, canonical_name = exact[-1]
        if not model_id:
            # A new player can still be forecast: ranking inputs come from the
            # current application page; unavailable historical features stay neutral.
            model_id = source_id
            warnings.append(
                f"{player.name}: нет истории матчей в bundle, использованы текущий рейтинг и нейтральные исторические признаки."
            )
        if model_id in used_ids:
            raise ValueError(f"Два участника сопоставились одному игроку модели: {player.name} ({model_id}).")
        used_ids.add(model_id)
        source_to_model[source_id] = model_id
        metadata = dict(player.metadata)
        metadata["source_player_id"] = source_id
        metadata["source_player_name"] = player.name
        resolved.append(
            replace(
                player,
                player_id=model_id,
                name=canonical_name or player.name,
                metadata=metadata,
            )
        )
    return resolved, source_to_model, warnings


def prediction_date_for(snapshot: TournamentSnapshot, today: date | None = None) -> pd.Timestamp:
    today = today or date.today()
    try:
        start = date.fromisoformat(snapshot.start_date)
    except (TypeError, ValueError):
        start = today
    return pd.Timestamp(max(today, start))


def build_probability_matrix(
    bundle: dict,
    players: Sequence[TournamentPlayer],
    prediction_date: str | pd.Timestamp,
    *,
    age_group: str,
    tournament_name: str,
    progress: ProgressCallback | None = None,
) -> tuple[dict[tuple[str, str], float], dict]:
    """Vectorize all possible H2H predictions, sharing expensive date state."""

    player_overrides = {
        str(player.player_id): {
            "name": player.name,
            "rank": player.rank,
            "points": player.points,
        }
        for player in players
    }
    state = prediction.prepare_prediction_state(
        bundle,
        prediction_date,
        player_overrides=player_overrides,
    )
    context = {
        "tournament_age_category": age_group or "__UNKNOWN_AGE__",
        "draw_type": "Олимпийская",
        "tournament_name": tournament_name or "__TOURNAMENT_SIMULATION__",
        "tournament_city": "__UNKNOWN_CITY__",
    }
    pairs = list(combinations(players, 2))
    frames = []
    total = len(pairs)
    for index, (player_a, player_b) in enumerate(pairs, start=1):
        rows = prediction.build_prediction_rows(
            bundle,
            str(player_a.player_id),
            str(player_b.player_id),
            pd.Timestamp(prediction_date),
            context=context,
            prediction_state=state,
        )
        rows["_pair_index"] = index - 1
        frames.append(rows)
        if progress and (index == total or index == 1 or index % 25 == 0):
            progress(index, total, f"Подготовка H2H {index}/{total}")

    if not frames:
        raise ValueError("Для матрицы вероятностей нужно минимум два игрока.")
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["_p_model"] = bundle["model"].predict_proba(all_rows[list(bundle["features"])])[:, 1]

    matrix: dict[tuple[str, str], float] = {}
    for pair_index, (player_a, player_b) in enumerate(pairs):
        pair_rows = all_rows[all_rows["_pair_index"].eq(pair_index)]
        summary = prediction.symmetrize_pair_probs(pair_rows, "_p_model")
        matrix[(str(player_a.player_id), str(player_b.player_id))] = summary["p_player1_sym"]
    if progress:
        progress(total, total, "Матрица H2H готова")
    return matrix, state


def _player_name_aliases(player: TournamentPlayer) -> tuple[str, ...]:
    aliases = [player.name, str(player.metadata.get("source_player_name", ""))]
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _name_tokens(value: object) -> tuple[str, ...]:
    return tuple(re.findall(r"[0-9a-zа-я]+", normalize_name(value)))


def _surname_initials(value: object) -> tuple[str, tuple[str, ...]]:
    tokens = _name_tokens(value)
    if not tokens:
        return "", ()
    return tokens[0], tuple(token[0] for token in tokens[1:3])


def _find_target(players: Sequence[TournamentPlayer], target_name: str) -> TournamentPlayer:
    target_norm = normalize_name(target_name)
    exact = [
        player
        for player in players
        if any(normalize_name(alias) == target_norm for alias in _player_name_aliases(player))
    ]
    if len(exact) == 1:
        return exact[0]

    target_surname, target_initials = _surname_initials(target_name)
    initials_matches: list[TournamentPlayer] = []
    if target_surname and target_initials:
        initials_matches = [
            player
            for player in players
            if any(
                surname == target_surname and initials[: len(target_initials)] == target_initials
                for surname, initials in map(_surname_initials, _player_name_aliases(player))
            )
        ]
        if len(initials_matches) == 1:
            return initials_matches[0]

    partial = [
        player
        for player in players
        if any(target_norm in normalize_name(alias) for alias in _player_name_aliases(player))
    ]
    if len(partial) == 1:
        return partial[0]
    candidates_found = exact or initials_matches or partial
    if not candidates_found:
        raise KeyError(f"Игрок «{target_name}» не найден среди участников/заявок турнира.")
    candidates = ", ".join(player.name for player in candidates_found)
    raise KeyError(f"Имя «{target_name}» неоднозначно: {candidates}.")


def _optional_target(
    players: Sequence[TournamentPlayer],
    target_name: str,
) -> TournamentPlayer | None:
    try:
        return _find_target(players, target_name)
    except KeyError as exc:
        if "не найден" in str(exc):
            return None
        raise


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def registration_status_allows_application(value: object) -> bool:
    """Return whether a player could still submit an RTT application now."""

    status = normalize_name(value)
    return "заяв" in status


def registration_status_needs_verification(value: object) -> bool:
    """Return whether the calendar status is missing and needs a fresh RTT card."""

    return normalize_name(value) in {"", "unknown", "все"}


def prepare_registration_scenario(
    bundle: dict,
    snapshot: TournamentSnapshot,
    target_name: str,
) -> TournamentSnapshot:
    """Keep a real entry or add the player as a prospective late application.

    Published draws and played matches are immutable. For an application list,
    the candidate competes for the advertised main-draw capacity by the same
    status/rank/points ordering used when the cached snapshot is built.
    """

    if _optional_target(snapshot.tournament_players(), target_name) is not None:
        return snapshot

    if snapshot.grid_slots or snapshot.completed_matches:
        raise ValueError(
            "игрок не заявлен, а официальная сетка уже опубликована или матчи уже начались"
        )
    if not registration_status_allows_application(snapshot.status):
        raise ValueError(
            f"игрок не заявлен, а статус «{snapshot.status}» не допускает новую заявку"
        )

    lookup = prediction.resolve_player_id_by_name(bundle["long_feat"], target_name)
    if not lookup.get("found"):
        raise KeyError(f"Игрок «{target_name}» не найден в модели.")

    model_player_id = str(lookup["player_id"])
    canonical_name = str(lookup.get("player_name") or target_name)
    rating = prediction.rating_snapshot(
        bundle,
        model_player_id,
        pd.Timestamp(date.today()),
        snapshot.age_group,
    )
    candidate = ParsedPlayer(
        name=canonical_name,
        rni=model_player_id.removeprefix("RNI:") if model_player_id.startswith("RNI:") else "",
        source_id=("" if model_player_id.startswith("RNI:") else f"virtual:{model_player_id}"),
        rank=_optional_float(rating.get("rank")),
        points=_optional_float(rating.get("points")),
        request_status="Поздняя заявка (виртуальный сценарий)",
        raw={
            "registration_scenario": "virtual",
            "model_player_id": model_player_id,
        },
    )

    capacity = snapshot.main_draw_capacity
    capacity = max(0, min(32, int(capacity))) if capacity is not None else 32
    if capacity < 4:
        raise ValueError("в карточке не задан допустимый состав ОТ минимум из 4 игроков")

    candidates = [*snapshot.players, candidate]
    if snapshot.player_source.startswith("official_members"):
        if len(snapshot.players) >= capacity:
            raise ValueError(f"состав ОТ уже заполнен ({len(snapshot.players)}/{capacity})")
        projected = candidates
        projection_notes: list[str] = []
    else:
        projected, projection_notes = project_main_draw(candidates, capacity)
        if candidate not in projected:
            raise ValueError(
                f"виртуальная заявка не проходит в ОТ по текущему рейтингу "
                f"(мест {capacity}, активных заявок {len(candidates)})"
            )

    if len(projected) < 4:
        opponents = max(0, len(projected) - 1)
        raise ValueError(
            f"после виртуальной заявки доступно только {opponents} соперников; "
            "для прогноза нужны минимум 4 участника"
        )

    rating_note = (
        f"место {candidate.rank:g}, очки {candidate.points:g}"
        if candidate.rank is not None and candidate.points is not None
        else f"место {candidate.rank:g}"
        if candidate.rank is not None
        else "рейтинг для этой возрастной группы не найден"
    )
    scenario_warning = (
        f"{canonical_name} отсутствует в заявках RTT и добавлен только для сценария "
        f"«зарегистрироваться сейчас» ({rating_note})."
    )
    return replace(
        snapshot,
        players=projected,
        player_source=(snapshot.player_source or "application_list") + "+virtual_registration",
        warnings=list(dict.fromkeys([*snapshot.warnings, *projection_notes, scenario_warning])),
    )


def _fixed_draw(
    snapshot: TournamentSnapshot,
    players: Sequence[TournamentPlayer],
    source_to_model: dict[str, str],
) -> list[TournamentPlayer | None] | None:
    if not snapshot.grid_slots:
        return None
    by_id = {str(player.player_id): player for player in players}
    draw: list[TournamentPlayer | None] = []
    for source_id in snapshot.grid_slots:
        if source_id is None:
            draw.append(None)
            continue
        model_id = source_to_model.get(str(source_id), str(source_id))
        if model_id not in by_id:
            return None
        draw.append(by_id[model_id])
    return draw


def _completed_winners(
    snapshot: TournamentSnapshot,
    players: Sequence[TournamentPlayer],
) -> tuple[dict[frozenset[str], str], list[str]]:
    def player_id_for(match_name: str) -> str | None:
        try:
            return str(_find_target(players, match_name).player_id)
        except KeyError:
            return None

    completed: dict[frozenset[str], str] = {}
    warnings: list[str] = []

    def lock_winner(player1_id: str, player2_id: str, winner_id: str, source: str) -> None:
        pair = frozenset((player1_id, player2_id))
        previous = completed.get(pair)
        if previous is not None and previous != winner_id:
            warnings.append(f"Противоречивый победитель матча {source}.")
            return
        completed[pair] = winner_id

    for match in snapshot.completed_matches:
        player1_id = player_id_for(match.player1)
        player2_id = player_id_for(match.player2)
        winner_id = player_id_for(match.winner)
        if player1_id and player2_id and winner_id in {player1_id, player2_id}:
            lock_winner(player1_id, player2_id, winner_id, f"{match.player1} — {match.player2}")
        elif match.winner:
            warnings.append(
                f"Не удалось зафиксировать сыгранный матч: {match.player1} — {match.player2}, победитель {match.winner}."
            )

    source_to_model = {
        str(player.metadata.get("source_player_id", player.player_id)): str(player.player_id)
        for player in players
    }
    for current_round, next_round in zip(snapshot.grid_rounds, snapshot.grid_rounds[1:]):
        for match_index, advanced_source_id in enumerate(next_round):
            if advanced_source_id is None:
                continue
            player1_source_id = current_round[2 * match_index]
            player2_source_id = current_round[2 * match_index + 1]
            if player1_source_id is None or player2_source_id is None:
                continue
            if advanced_source_id not in {player1_source_id, player2_source_id}:
                warnings.append("Опубликованная сетка содержит несогласованное продвижение игрока.")
                continue
            player1_id = source_to_model.get(str(player1_source_id))
            player2_id = source_to_model.get(str(player2_source_id))
            winner_id = source_to_model.get(str(advanced_source_id))
            if player1_id and player2_id and winner_id:
                lock_winner(player1_id, player2_id, winner_id, "в опубликованной сетке")
    return completed, warnings


def analyze_tournament(
    bundle: dict,
    snapshot: TournamentSnapshot,
    target_name: str,
    *,
    iterations: int = 20_000,
    random_seed: int = 2026,
    progress: ProgressCallback | None = None,
) -> TournamentAnalysis:
    """Run the complete single-player tournament analysis."""

    if not snapshot.eligible:
        raise ValueError(snapshot.eligibility_reason or "Турнир недоступен для прогноза мест.")
    if "олимп" not in normalize_name(snapshot.draw_system):
        raise NotImplementedError(
            f"Сейчас моделируется олимпийская система; в карточке указано «{snapshot.draw_system}»."
        )
    source_players = snapshot.tournament_players()
    if not 4 <= len(source_players) <= 32:
        raise ValueError(f"Для олимпийской сетки нужно 4–32 игрока, сейчас найдено {len(source_players)}.")

    players, source_to_model, resolve_warnings = resolve_players_for_model(bundle, source_players)
    target = _find_target(players, target_name)
    prediction_date = prediction_date_for(snapshot)
    matrix, _ = build_probability_matrix(
        bundle,
        players,
        prediction_date,
        age_group=snapshot.age_group,
        tournament_name=snapshot.title,
        progress=progress,
    )
    fixed_draw = _fixed_draw(snapshot, players, source_to_model)
    completed, completed_warnings = _completed_winners(snapshot, players)
    if completed and fixed_draw is None:
        raise ValueError(
            "Турнир уже идёт, но официальную сетку не удалось разобрать. "
            "Расчёт остановлен, чтобы не пересэмплировать уже состоявшуюся жеребьёвку."
        )
    simulation = simulate_tournament(
        players,
        PairProbabilityMatrix(matrix),
        iterations=iterations,
        random_seed=random_seed,
        fixed_draw=fixed_draw,
        completed_winners=completed,
        target_player_ids=[target.player_id],
        age_group=snapshot.age_group,
        tournament_category=snapshot.category,
    )
    warnings = [
        *snapshot.warnings,
        *resolve_warnings,
        *completed_warnings,
        *simulation.warnings,
    ]
    if snapshot.grid_slots and fixed_draw is None:
        warnings.append("Опубликованную сетку не удалось однозначно сопоставить; сетка пересэмплирована по правилам РТТ.")
    return TournamentAnalysis(
        snapshot=snapshot,
        target_player=target,
        players=players,
        simulation=simulation,
        probability_matrix=matrix,
        prediction_date=prediction_date,
        warnings=list(dict.fromkeys(warnings)),
    )


def mega_summary(analyses: Iterable[TournamentAnalysis]) -> pd.DataFrame:
    """Rank eligible tournaments by expected RTT points for the selected player."""

    rows = []
    for analysis in analyses:
        target_id = analysis.target_player.player_id
        win_probability = analysis.simulation.distributions.get(target_id, {}).get("1", 0.0)
        rows.append(
            {
                "tour_id": analysis.snapshot.tour_id,
                "tournament": analysis.snapshot.title,
                "start_date": analysis.snapshot.start_date,
                "age_group": analysis.snapshot.age_group,
                "category": analysis.snapshot.category,
                "players": len(analysis.players),
                "entry_scenario": (
                    "Виртуальная заявка"
                    if "virtual_registration" in analysis.snapshot.player_source
                    else "Фактическая заявка"
                ),
                "list_source": analysis.snapshot.player_source,
                "official_grid": analysis.simulation.draw_is_fixed,
                "win_probability": win_probability,
                "win_probability_pct": 100.0 * win_probability,
                "expected_points": analysis.simulation.expected_points.get(target_id),
                "warnings": " | ".join(analysis.warnings),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["expected_points", "win_probability"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )


def current_rating_age_groups(
    bundle: dict,
    target_name: str,
    *,
    today: date | None = None,
) -> list[str]:
    """Return every age group in the player's latest available classification."""

    lookup = prediction.resolve_player_id_by_name(bundle["long_feat"], target_name)
    if not lookup.get("found"):
        return []
    history = prediction.player_rating_history(bundle, str(lookup["player_id"]))
    if history.empty or "classification_date" not in history or "age_group" not in history:
        return []
    cutoff = pd.Timestamp(today or date.today())
    dates = pd.to_datetime(history["classification_date"], errors="coerce")
    available = history.loc[dates.le(cutoff) & dates.notna()].copy()
    if available.empty:
        return []
    available_dates = pd.to_datetime(available["classification_date"], errors="coerce")
    latest = available.loc[available_dates.eq(available_dates.max()), "age_group"].dropna()
    return list(dict.fromkeys(str(value) for value in latest if normalize_age_group(value)))


def cached_registered_tour_ids(
    cache_dir: Path | str,
    target_name: str,
) -> list[str]:
    """Find still-eligible cached tournaments containing the selected player."""

    rows: list[tuple[pd.Timestamp, str]] = []
    for snapshot_file in Path(cache_dir).glob("*/snapshot.json"):
        tour_id = snapshot_file.parent.name
        try:
            snapshot = load_snapshot(cache_dir, tour_id)
            registered = _optional_target(snapshot.tournament_players(), target_name) is not None
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not snapshot.eligible or not registered:
            continue
        start = pd.to_datetime(snapshot.start_date, errors="coerce")
        sort_date = start if pd.notna(start) else pd.Timestamp.max
        rows.append((sort_date, str(snapshot.tour_id)))
    rows.sort(key=lambda row: (row[0], row[1]))
    return list(dict.fromkeys(tour_id for _, tour_id in rows))


def eligible_tour_ids_from_master(
    master_path: Path | str,
    *,
    today: date | None = None,
    age_group: str | Sequence[str] | None = None,
) -> list[str]:
    """Read-only discovery helper for the notebook's mega mode."""

    today = today or date.today()
    frame = pd.read_excel(master_path)
    if "tour_id" not in frame.columns:
        return []
    start = pd.to_datetime(frame.get("start_date"), errors="coerce")
    status = frame.get("status", pd.Series("", index=frame.index)).fillna("").astype(str).map(normalize_name)
    terminal = status.str.contains("заверш|сдача отч|отмен|не состоя|аннулир", regex=True)
    # The master contributes only tournaments where a new application may still
    # be submitted. Already-entered active tournaments are merged from cache by
    # ``cached_registered_tour_ids``.
    future_or_today = start.ge(pd.Timestamp(today))
    application_candidate = status.map(
        lambda value: registration_status_allows_application(value)
        or registration_status_needs_verification(value)
    )
    mask = future_or_today & ~terminal & application_candidate
    requested_values = [age_group] if isinstance(age_group, str) else list(age_group or [])
    requested_ages = {normalize_age_group(value) for value in requested_values if normalize_age_group(value)}
    if requested_ages and "age_category" in frame.columns:
        mask &= frame["age_category"].map(normalize_age_group).isin(requested_ages)
    selected = frame.loc[mask, ["tour_id"]].copy()
    selected["_start_date"] = start.loc[mask]
    selected = selected.sort_values("_start_date", kind="stable")
    ids = selected["tour_id"].dropna().astype(str).str.replace(r"\.0$", "", regex=True)
    return ids.drop_duplicates().tolist()
