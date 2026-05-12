from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ELO_BASE_RATING = 1500.0
ELO_K = 32.0
ELO_MARGIN_COEF = 0.08
ELO_RECENCY_HALF_LIFE_DAYS = 45.0
COMMON_OPP_HALF_LIFE_DAYS = 180.0
SCHEDULE_STRENGTH_WINDOW = 10
OPP_ELO_LAST_WINDOW = 5

COMMON_OPP_FEATURES = [
    "common_opp_count",
    "common_opp_available",
    "common_opp_player_matches_sum",
    "common_opp_opponent_matches_sum",
    "common_opp_player_wins_sum",
    "common_opp_opponent_wins_sum",
    "common_opp_player_winrate_mean",
    "common_opp_opponent_winrate_mean",
    "common_opp_winrate_edge_mean",
    "common_opp_weighted_edge",
    "common_opp_weight_sum",
    "common_opp_min_pair_matches_sum",
]


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    raise FileNotFoundError("Could not find project root.")


PROJECT_ROOT = find_project_root()
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "assembled_predictor" / "prediction_bundle.joblib"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "assembled_predictor" / "predictor_model_dataset_from_parsers.xlsx"


def load_prediction_bundle(path: Path | str | None = None) -> dict[str, Any]:
    bundle_path = Path(path) if path is not None else DEFAULT_BUNDLE_PATH
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Prediction bundle not found: {bundle_path}. Run scripts/train_model.py or the full pipeline first."
        )
    bundle = joblib.load(bundle_path)
    bundle["bundle_path"] = str(bundle_path)
    bundle["long_feat"] = prepare_long_feat(bundle["long_feat"])
    if "rating_history" not in bundle or not isinstance(bundle.get("rating_history"), pd.DataFrame):
        data_path = Path(bundle.get("data_path", DEFAULT_DATASET_PATH))
        if data_path.exists():
            bundle["rating_history"] = pd.read_excel(data_path, sheet_name="rating_history")
        else:
            bundle["rating_history"] = pd.DataFrame()
    return bundle


def prepare_long_feat(long_feat: pd.DataFrame) -> pd.DataFrame:
    df = long_feat.copy()
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    for col in ["player_id", "opponent_id", "player_name", "opponent_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def normalize_player_name(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    text = text.replace("\xa0", " ").lower().replace("ё", "е")
    return " ".join(text.split())


def normalize_age_group(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    text = text.replace("\xa0", " ").lower().replace("ё", "е")
    text = " ".join(text.split())
    if text in {"", "__unknown_age__", "unknown"}:
        return ""
    if "15" in text:
        return "до 15 лет"
    if "17" in text:
        return "до 17 лет"
    if "19" in text:
        return "до 19 лет"
    if "взрос" in text or "adult" in text:
        return "взрослые"
    return text


def player_options(bundle: dict[str, Any]) -> list[str]:
    long_feat = bundle["long_feat"]
    directory = (
        long_feat[["player_id", "player_name", "match_date"]]
        .dropna(subset=["player_id", "player_name"])
        .sort_values(["player_name", "match_date"])
        .drop_duplicates(subset=["player_id"], keep="last")
    )
    return sorted(directory["player_name"].astype(str).unique().tolist())


def build_player_directory(long_feat: pd.DataFrame) -> pd.DataFrame:
    directory = long_feat[["player_id", "player_name", "match_date"]].dropna(subset=["player_id"]).copy()
    directory["player_id"] = directory["player_id"].astype(str)
    directory["player_name_norm"] = directory["player_name"].apply(normalize_player_name)
    return directory.sort_values(["player_name_norm", "match_date"])


def resolve_player_id_by_name(long_feat: pd.DataFrame, player_name: str) -> dict[str, Any]:
    directory = build_player_directory(long_feat)
    name_norm = normalize_player_name(player_name)

    exact = directory[directory["player_name_norm"] == name_norm].copy()
    if not exact.empty:
        latest = exact.sort_values("match_date").iloc[-1]
        return {
            "found": True,
            "player_id": str(latest["player_id"]),
            "player_name": latest["player_name"],
            "candidates": exact[["player_id", "player_name", "match_date"]].drop_duplicates().tail(10),
        }

    partial = directory[directory["player_name_norm"].str.contains(name_norm, na=False, regex=False)].copy()
    if not partial.empty:
        latest = partial.sort_values("match_date").iloc[-1]
        return {
            "found": True,
            "player_id": str(latest["player_id"]),
            "player_name": latest["player_name"],
            "candidates": partial[["player_id", "player_name", "match_date"]].drop_duplicates().tail(10),
        }

    return {
        "found": False,
        "player_id": None,
        "player_name": None,
        "candidates": pd.DataFrame(columns=["player_id", "player_name", "match_date"]),
    }


def build_match_level_base(long_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["match_id", "match_date", "player1_id", "player2_id", "winner_player1", "games_diff"]
    cols = [col for col in cols if col in long_df.columns]
    return (
        long_df[cols]
        .drop_duplicates(subset=["match_id"])
        .sort_values(["match_date", "match_id"])
        .reset_index(drop=True)
        .copy()
    )


def add_adjusted_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    specs = [
        ("player_points_pre", "player_rated_counting_tournaments_pre", "player_points_per_counting_tournament_pre"),
        ("opponent_points_pre", "opponent_rated_counting_tournaments_pre", "opponent_points_per_counting_tournament_pre"),
    ]
    for points_col, count_col, out_col in specs:
        if points_col not in out.columns or count_col not in out.columns:
            continue
        points = pd.to_numeric(out[points_col], errors="coerce")
        count = pd.to_numeric(out[count_col], errors="coerce")
        out[out_col] = np.where(count > 0, points / count, np.nan)
    if {
        "player_points_per_counting_tournament_pre",
        "opponent_points_per_counting_tournament_pre",
    }.issubset(out.columns):
        out["diff_points_per_counting_tournament_pre"] = (
            out["player_points_per_counting_tournament_pre"]
            - out["opponent_points_per_counting_tournament_pre"]
        )
    return out


def add_relative_diff_to_min_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    specs = [
        ("player_points_pre", "opponent_points_pre", "points"),
        (
            "player_points_per_counting_tournament_pre",
            "opponent_points_per_counting_tournament_pre",
            "points_per_counting_tournament",
        ),
    ]
    for player_col, opponent_col, stem in specs:
        if player_col not in out.columns or opponent_col not in out.columns:
            continue
        player_value = pd.to_numeric(out[player_col], errors="coerce")
        opponent_value = pd.to_numeric(out[opponent_col], errors="coerce")
        denominator = np.minimum(player_value, opponent_value)
        valid = player_value.notna() & opponent_value.notna() & (denominator > 0)
        out[f"rel_diff_{stem}_pct_min_pre"] = np.where(
            valid,
            100.0 * (player_value - opponent_value) / denominator,
            np.nan,
        )
    return out


def add_observed_only_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    specs = [
        ("player_rank_pre", "opponent_rank_pre", "rank"),
        ("player_points_pre", "opponent_points_pre", "points"),
        (
            "player_points_per_counting_tournament_pre",
            "opponent_points_per_counting_tournament_pre",
            "points_per_counting_tournament",
        ),
    ]
    for player_col, opponent_col, stem in specs:
        if player_col not in out.columns or opponent_col not in out.columns:
            continue
        player_value = pd.to_numeric(out[player_col], errors="coerce")
        opponent_value = pd.to_numeric(out[opponent_col], errors="coerce")
        both_observed = player_value.notna() & opponent_value.notna()
        out[f"both_{stem}_observed_pre"] = both_observed.astype(int)
        out[f"diff_{stem}_pre_observed_only"] = np.where(both_observed, player_value - opponent_value, np.nan)
        if stem != "rank":
            denominator = np.minimum(player_value, opponent_value)
            valid = both_observed & (denominator > 0)
            out[f"rel_diff_{stem}_pct_min_pre_observed_only"] = np.where(
                valid,
                100.0 * (player_value - opponent_value) / denominator,
                np.nan,
            )
    return out


def fill_feature_nans(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["h2h_player_winrate_before"]:
        if col in out.columns:
            out[col] = out[col].fillna(0.5)
    for col in ["h2h_matches_before", "h2h_player_wins_before", "player_matches_pre", "opponent_matches_pre"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    if "schedule_strength" in out.columns and "elo_pre" in out.columns:
        out["schedule_strength"] = out["schedule_strength"].fillna(out["elo_pre"])
    if "opp_elo_last_5" in out.columns and "elo_opp_pre" in out.columns:
        out["opp_elo_last_5"] = out["opp_elo_last_5"].fillna(out["elo_opp_pre"])
    return out


def compute_elo_state_until(long_feat: pd.DataFrame, prediction_date: pd.Timestamp) -> dict[str, float]:
    match_base = build_match_level_base(long_feat)
    match_base = match_base[match_base["match_date"] < prediction_date].copy()
    elo_state: dict[str, float] = {}
    last_match_date: dict[str, pd.Timestamp] = {}

    for match in match_base.itertuples(index=False):
        player1_id = str(match.player1_id)
        player2_id = str(match.player2_id)
        match_date = pd.Timestamp(match.match_date)
        rating1 = float(elo_state.get(player1_id, ELO_BASE_RATING))
        rating2 = float(elo_state.get(player2_id, ELO_BASE_RATING))
        expected_player1 = 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))
        games_diff = getattr(match, "games_diff", np.nan)
        margin_multiplier = 1.0 if pd.isna(games_diff) else 1.0 + ELO_MARGIN_COEF * np.log1p(abs(float(games_diff)))

        def recency_weight(player_id: str) -> float:
            prev_date = last_match_date.get(player_id)
            if prev_date is None:
                return 1.0
            days = max((match_date - prev_date).days, 0)
            return 1.0 + (1.0 - np.exp(-days / ELO_RECENCY_HALF_LIFE_DAYS))

        winner_player1 = float(match.winner_player1)
        winner_player2 = 1.0 - winner_player1
        k1 = ELO_K * margin_multiplier * recency_weight(player1_id)
        k2 = ELO_K * margin_multiplier * recency_weight(player2_id)
        elo_state[player1_id] = rating1 + k1 * (winner_player1 - expected_player1)
        elo_state[player2_id] = rating2 + k2 * (winner_player2 - (1.0 - expected_player1))
        last_match_date[player1_id] = match_date
        last_match_date[player2_id] = match_date

    return elo_state


def player_history_rows(long_feat: pd.DataFrame, player_id: str, prediction_date: pd.Timestamp) -> pd.DataFrame:
    return (
        long_feat[
            (long_feat["player_id"].astype(str) == str(player_id))
            & (long_feat["match_date"] < prediction_date)
        ]
        .sort_values(["match_date", "match_id"])
        .copy()
    )


def get_last_player_snapshot(long_feat: pd.DataFrame, player_id: str, prediction_date: pd.Timestamp) -> pd.Series | None:
    history = player_history_rows(long_feat, player_id, prediction_date)
    if history.empty:
        return None
    return history.iloc[-1]


def h2h_stats_until(long_feat: pd.DataFrame, player_id: str, opponent_id: str, prediction_date: pd.Timestamp) -> dict[str, Any]:
    history = (
        long_feat[
            (long_feat["player_id"].astype(str) == str(player_id))
            & (long_feat["opponent_id"].astype(str) == str(opponent_id))
            & (long_feat["match_date"] < prediction_date)
        ]
        .sort_values(["match_date", "match_id"])
        .copy()
    )
    n_matches = int(history["match_id"].nunique()) if not history.empty else 0
    wins = int(history.drop_duplicates("match_id")["win"].sum()) if n_matches > 0 else 0
    return {
        "h2h_matches_before": n_matches,
        "h2h_player_wins_before": wins,
        "h2h_player_winrate_before": wins / n_matches if n_matches > 0 else np.nan,
    }


def empty_common_opponent_feature_row(match_id: int, perspective: str) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "perspective": perspective,
        "common_opp_count": 0,
        "common_opp_available": 0,
        "common_opp_player_matches_sum": 0,
        "common_opp_opponent_matches_sum": 0,
        "common_opp_player_wins_sum": 0,
        "common_opp_opponent_wins_sum": 0,
        "common_opp_player_winrate_mean": np.nan,
        "common_opp_opponent_winrate_mean": np.nan,
        "common_opp_winrate_edge_mean": np.nan,
        "common_opp_weighted_edge": np.nan,
        "common_opp_weight_sum": 0.0,
        "common_opp_min_pair_matches_sum": 0,
    }


def common_opponent_features_for_pair(
    stats_by_player: dict[str, dict[str, dict[str, Any]]],
    player_id: str,
    opponent_id: str,
    match_id: int,
    perspective: str,
    match_date: pd.Timestamp,
) -> dict[str, Any]:
    row = empty_common_opponent_feature_row(match_id, perspective)
    common_opponents = set(stats_by_player.get(player_id, {})) & set(stats_by_player.get(opponent_id, {}))
    common_opponents.discard(player_id)
    common_opponents.discard(opponent_id)
    if not common_opponents:
        return row

    edges = []
    weighted_edges = []
    weights = []
    player_matches_sum = opponent_matches_sum = player_wins_sum = opponent_wins_sum = min_pair_matches_sum = 0
    player_winrates = []
    opponent_winrates = []

    for common_id in common_opponents:
        player_stats = stats_by_player[player_id].get(common_id)
        opponent_stats = stats_by_player[opponent_id].get(common_id)
        if not player_stats or not opponent_stats:
            continue
        player_n = int(player_stats["n"])
        opponent_n = int(opponent_stats["n"])
        if player_n <= 0 or opponent_n <= 0:
            continue
        player_wins = int(player_stats["wins"])
        opponent_wins = int(opponent_stats["wins"])
        player_winrate = player_wins / player_n
        opponent_winrate = opponent_wins / opponent_n
        edge = player_winrate - opponent_winrate
        last_date = max(pd.Timestamp(player_stats["last_date"]), pd.Timestamp(opponent_stats["last_date"]))
        days_ago = max((pd.Timestamp(match_date) - last_date).days, 0)
        weight = float(np.exp(-days_ago / COMMON_OPP_HALF_LIFE_DAYS) * np.sqrt(min(player_n, opponent_n)))

        edges.append(edge)
        weights.append(weight)
        weighted_edges.append(weight * edge)
        player_matches_sum += player_n
        opponent_matches_sum += opponent_n
        player_wins_sum += player_wins
        opponent_wins_sum += opponent_wins
        min_pair_matches_sum += min(player_n, opponent_n)
        player_winrates.append(player_winrate)
        opponent_winrates.append(opponent_winrate)

    if not edges:
        return row

    weight_sum = float(np.sum(weights))
    row.update({
        "common_opp_count": int(len(edges)),
        "common_opp_available": 1,
        "common_opp_player_matches_sum": int(player_matches_sum),
        "common_opp_opponent_matches_sum": int(opponent_matches_sum),
        "common_opp_player_wins_sum": int(player_wins_sum),
        "common_opp_opponent_wins_sum": int(opponent_wins_sum),
        "common_opp_player_winrate_mean": float(np.mean(player_winrates)),
        "common_opp_opponent_winrate_mean": float(np.mean(opponent_winrates)),
        "common_opp_winrate_edge_mean": float(np.mean(edges)),
        "common_opp_weighted_edge": float(np.sum(weighted_edges) / weight_sum) if weight_sum > 0 else np.nan,
        "common_opp_weight_sum": weight_sum,
        "common_opp_min_pair_matches_sum": int(min_pair_matches_sum),
    })
    return row


def build_common_opponent_stats_until(long_feat: pd.DataFrame, prediction_date: pd.Timestamp) -> dict[str, dict[str, dict[str, Any]]]:
    match_base = build_match_level_base(long_feat)
    match_base = match_base[match_base["match_date"] < prediction_date].copy()
    stats_by_player: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    def update_stats(player_id: str, opponent_id: str, win: int, match_date: pd.Timestamp) -> None:
        current = stats_by_player[player_id].get(opponent_id)
        if current is None:
            current = {"n": 0, "wins": 0, "last_date": pd.Timestamp(match_date)}
        current["n"] = int(current["n"]) + 1
        current["wins"] = int(current["wins"]) + int(win)
        current["last_date"] = pd.Timestamp(match_date)
        stats_by_player[player_id][opponent_id] = current

    for match in match_base.itertuples(index=False):
        player1_id = str(match.player1_id)
        player2_id = str(match.player2_id)
        player1_win = int(match.winner_player1)
        match_date = pd.Timestamp(match.match_date)
        update_stats(player1_id, player2_id, player1_win, match_date)
        update_stats(player2_id, player1_id, 1 - player1_win, match_date)

    return stats_by_player


def build_common_opponent_prediction_features(
    long_feat: pd.DataFrame,
    player_a_id: str,
    player_b_id: str,
    prediction_date: pd.Timestamp,
) -> pd.DataFrame:
    stats_by_player = build_common_opponent_stats_until(long_feat, prediction_date)
    rows = [
        common_opponent_features_for_pair(stats_by_player, player_a_id, player_b_id, -1, "player1", prediction_date),
        common_opponent_features_for_pair(stats_by_player, player_b_id, player_a_id, -1, "player2", prediction_date),
    ]
    return pd.DataFrame(rows).drop(columns=["match_id"], errors="ignore")


def build_single_prediction_row(
    bundle: dict[str, Any],
    player_id: str,
    opponent_id: str,
    prediction_date: pd.Timestamp,
    perspective: str,
    context: dict[str, Any],
    elo_state: dict[str, float],
) -> dict[str, Any]:
    long_feat = bundle["long_feat"]
    player_last = get_last_player_snapshot(long_feat, player_id, prediction_date)
    opponent_last = get_last_player_snapshot(long_feat, opponent_id, prediction_date)
    player_history = player_history_rows(long_feat, player_id, prediction_date)
    opponent_history = player_history_rows(long_feat, opponent_id, prediction_date)
    player_elo = float(elo_state.get(str(player_id), ELO_BASE_RATING))
    opponent_elo = float(elo_state.get(str(opponent_id), ELO_BASE_RATING))
    expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))

    def from_last(last_row: pd.Series | None, col: str, default=np.nan):
        if last_row is None or col not in long_feat.columns:
            return default
        return last_row.get(col, default)

    def days_since_last_match(history: pd.DataFrame) -> float:
        if history.empty:
            return np.nan
        return (prediction_date - pd.Timestamp(history["match_date"].iloc[-1])).days

    player_days_since = days_since_last_match(player_history)
    opponent_days_since = days_since_last_match(opponent_history)
    h2h = h2h_stats_until(long_feat, player_id, opponent_id, prediction_date)
    player_rating = rating_snapshot(
        bundle,
        player_id,
        prediction_date,
        context.get("tournament_age_category"),
    )
    opponent_rating = rating_snapshot(
        bundle,
        opponent_id,
        prediction_date,
        context.get("tournament_age_category"),
    )

    return {
        "match_id": -1,
        "match_date": prediction_date,
        "perspective": perspective,
        "player_id": str(player_id),
        "opponent_id": str(opponent_id),
        "player_name": from_last(player_last, "player_name", str(player_id)),
        "opponent_name": from_last(opponent_last, "player_name", str(opponent_id)),
        "tournament_name": context.get("tournament_name", "__UNKNOWN_TOURNAMENT__"),
        "tournament_city": context.get("tournament_city", "__UNKNOWN_CITY__"),
        "tournament_age_category": context.get("tournament_age_category", "__UNKNOWN_AGE__"),
        "draw_type": context.get("draw_type", "__UNKNOWN_DRAW__"),
        "elo_pre": player_elo,
        "elo_opp_pre": opponent_elo,
        "elo_diff": player_elo - opponent_elo,
        "expected_win_prob_elo": expected,
        "player_rank_pre": player_rating.get("rank", from_last(player_last, "player_rank_pre")),
        "opponent_rank_pre": opponent_rating.get("rank", from_last(opponent_last, "player_rank_pre")),
        "player_points_pre": player_rating.get("points", from_last(player_last, "player_points_pre")),
        "opponent_points_pre": opponent_rating.get("points", from_last(opponent_last, "player_points_pre")),
        "player_rating_date_pre": player_rating.get("classification_date", from_last(player_last, "player_rating_date_pre")),
        "opponent_rating_date_pre": opponent_rating.get("classification_date", from_last(opponent_last, "player_rating_date_pre")),
        "player_rating_age_group_pre": player_rating.get("age_group", from_last(player_last, "player_rating_age_group_pre")),
        "opponent_rating_age_group_pre": opponent_rating.get("age_group", from_last(opponent_last, "player_rating_age_group_pre")),
        "player_rated_tournaments_pre": player_rating.get("rated_tournaments", from_last(player_last, "player_rated_tournaments_pre")),
        "opponent_rated_tournaments_pre": opponent_rating.get("rated_tournaments", from_last(opponent_last, "player_rated_tournaments_pre")),
        "player_rated_counting_tournaments_pre": player_rating.get("counting_tournaments", from_last(player_last, "player_rated_counting_tournaments_pre")),
        "opponent_rated_counting_tournaments_pre": opponent_rating.get("counting_tournaments", from_last(opponent_last, "player_rated_counting_tournaments_pre")),
        "player_matches_pre": len(player_history),
        "opponent_matches_pre": len(opponent_history),
        "experience_diff": len(player_history) - len(opponent_history),
        "player_winrate_all": float(player_history["win"].mean()) if not player_history.empty else np.nan,
        "days_since_prev_match": player_days_since,
        "rest_diff_days": (
            player_days_since - opponent_days_since
            if pd.notna(player_days_since) and pd.notna(opponent_days_since)
            else np.nan
        ),
        "schedule_strength": (
            float(player_history["elo_opp_pre"].tail(SCHEDULE_STRENGTH_WINDOW).mean())
            if not player_history.empty and "elo_opp_pre" in player_history.columns
            else player_elo
        ),
        "opp_elo_last_5": (
            float(player_history["elo_opp_pre"].tail(OPP_ELO_LAST_WINDOW).mean())
            if not player_history.empty and "elo_opp_pre" in player_history.columns
            else opponent_elo
        ),
        **h2h,
    }


def build_prediction_rows(
    bundle: dict[str, Any],
    player_a_id: str,
    player_b_id: str,
    prediction_date: pd.Timestamp,
    context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    long_feat = bundle["long_feat"]
    features = list(bundle["features"])
    context = dict(context or {})
    prediction_date = pd.Timestamp(prediction_date)
    elo_state = compute_elo_state_until(long_feat, prediction_date)

    pred_df = pd.DataFrame([
        build_single_prediction_row(bundle, player_a_id, player_b_id, prediction_date, "player1", context, elo_state),
        build_single_prediction_row(bundle, player_b_id, player_a_id, prediction_date, "player2", context, elo_state),
    ])
    pred_df = add_adjusted_rating_features(pred_df)
    pred_df = add_relative_diff_to_min_features(pred_df)
    pred_df = add_observed_only_features(pred_df)
    common_pred = build_common_opponent_prediction_features(long_feat, player_a_id, player_b_id, prediction_date)
    pred_df = pred_df.drop(columns=[col for col in COMMON_OPP_FEATURES if col in pred_df.columns], errors="ignore")
    pred_df = pred_df.merge(common_pred, on="perspective", how="left", validate="one_to_one")
    pred_df = fill_feature_nans(pred_df)

    medians = numeric_feature_medians(bundle)
    for col in features:
        if col not in pred_df.columns:
            pred_df[col] = medians.get(col, np.nan)
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
        if col in medians:
            pred_df[col] = pred_df[col].fillna(medians[col])
    return pred_df


def numeric_feature_medians(bundle: dict[str, Any]) -> dict[str, float]:
    long_feat = bundle["long_feat"]
    medians: dict[str, float] = {}
    for feature in bundle["features"]:
        if feature in long_feat.columns:
            value = pd.to_numeric(long_feat[feature], errors="coerce").median()
            if pd.notna(value):
                medians[feature] = float(value)
    return medians


def symmetrize_pair_probs(pred_rows: pd.DataFrame, prob_col: str = "p_model_raw") -> dict[str, float]:
    p_ab = float(pred_rows.iloc[0][prob_col])
    p_ba = float(pred_rows.iloc[1][prob_col])
    p_player1 = float(np.clip(0.5 * (p_ab + (1.0 - p_ba)), 0.0, 1.0))
    return {
        "p_player1_sym": p_player1,
        "p_player2_sym": 1.0 - p_player1,
        "p_ab": p_ab,
        "p_ba": p_ba,
        "symmetry_gap_abs": abs(p_ab - (1.0 - p_ba)),
    }


def predict_match_by_names(
    bundle: dict[str, Any],
    player1_name: str,
    player2_name: str,
    prediction_date: str | pd.Timestamp,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    long_feat = bundle["long_feat"]
    model = bundle["model"]
    features = list(bundle["features"])
    prediction_date = pd.Timestamp(prediction_date)
    player1_info = resolve_player_id_by_name(long_feat, player1_name)
    player2_info = resolve_player_id_by_name(long_feat, player2_name)
    if not player1_info["found"] or not player2_info["found"]:
        return {
            "ok": False,
            "message": "Не удалось найти одного или обоих игроков.",
            "player1_lookup": player1_info,
            "player2_lookup": player2_info,
        }

    pred_rows = build_prediction_rows(
        bundle=bundle,
        player_a_id=player1_info["player_id"],
        player_b_id=player2_info["player_id"],
        prediction_date=prediction_date,
        context=context,
    )
    x_pred = pred_rows[features].copy()
    pred_rows["p_model_raw"] = model.predict_proba(x_pred)[:, 1]
    pair_summary = symmetrize_pair_probs(pred_rows, "p_model_raw")
    pred_rows["p_final_player1"] = pair_summary["p_player1_sym"]
    pred_rows["p_final_player2"] = pair_summary["p_player2_sym"]
    pred_rows["symmetry_gap_abs"] = pair_summary["symmetry_gap_abs"]

    return {
        "ok": True,
        "prediction_rows": pred_rows,
        "p_player1_win": pair_summary["p_player1_sym"],
        "p_player2_win": pair_summary["p_player2_sym"],
        "player1_lookup": player1_info,
        "player2_lookup": player2_info,
        "model_name": bundle.get("model_name", "production_model"),
        "factor_contributions": factor_contribution_table(bundle, pred_rows),
        "player1_profile": player_profile(bundle, player1_info["player_id"], prediction_date, player2_info["player_id"]),
        "player2_profile": player_profile(bundle, player2_info["player_id"], prediction_date, player1_info["player_id"]),
    }


def player_profile(
    bundle: dict[str, Any],
    player_id: str,
    prediction_date: pd.Timestamp,
    opponent_id: str | None = None,
) -> dict[str, Any]:
    long_feat = bundle["long_feat"]
    history = player_history_rows(long_feat, player_id, prediction_date)
    last = history.iloc[-1] if not history.empty else None
    current_rating = rating_snapshot(bundle, player_id, prediction_date)
    elo_state = compute_elo_state_until(long_feat, prediction_date)
    last_date = pd.Timestamp(history["match_date"].iloc[-1]) if not history.empty else pd.NaT
    form5 = float(history["win"].tail(5).mean()) if not history.empty else np.nan
    form10 = float(history["win"].tail(10).mean()) if not history.empty else np.nan
    h2h = h2h_stats_until(long_feat, player_id, opponent_id, prediction_date) if opponent_id else {}

    def val(col: str, default=np.nan):
        if last is None or col not in history.columns:
            return default
        return last.get(col, default)

    return {
        "player_id": str(player_id),
        "player_name": val("player_name", str(player_id)),
        "rank": current_rating.get("rank", val("player_rank_pre")),
        "points": current_rating.get("points", val("player_points_pre")),
        "rating_date": current_rating.get("classification_date", val("player_rating_date_pre")),
        "rating_age_group": current_rating.get("age_group", val("player_rating_age_group_pre", "")),
        "elo": float(elo_state.get(str(player_id), ELO_BASE_RATING)),
        "matches": int(history["match_id"].nunique()) if not history.empty else 0,
        "wins": int(history.drop_duplicates("match_id")["win"].sum()) if not history.empty else 0,
        "winrate": float(history.drop_duplicates("match_id")["win"].mean()) if not history.empty else np.nan,
        "form5": form5,
        "form10": form10,
        "last_match_date": last_date,
        "days_rest": int((prediction_date - last_date).days) if pd.notna(last_date) else np.nan,
        "avg_opp_elo_last5": (
            float(history["elo_opp_pre"].tail(5).mean())
            if not history.empty and "elo_opp_pre" in history.columns
            else np.nan
        ),
        **h2h,
    }


def profiles_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key in ["player1_profile", "player2_profile"]:
        profile = result[key]
        rows.append({
            "Player": profile["player_name"],
            "Rank": profile["rank"],
            "Points": profile["points"],
            "ELO": round(profile["elo"], 1),
            "Matches": profile["matches"],
            "Wins": profile["wins"],
            "Win rate": profile["winrate"],
            "Form 5": profile["form5"],
            "Form 10": profile["form10"],
            "Last match": profile["last_match_date"],
            "Rest days": profile["days_rest"],
            "H2H matches": profile.get("h2h_matches_before", 0),
            "H2H wins": profile.get("h2h_player_wins_before", 0),
        })
    return pd.DataFrame(rows)


def player_history(bundle: dict[str, Any], player_id: str, prediction_date: pd.Timestamp) -> pd.DataFrame:
    history = player_history_rows(bundle["long_feat"], player_id, pd.Timestamp(prediction_date))
    cols = [
        "match_date",
        "player_name",
        "opponent_name",
        "win",
        "player_rank_pre",
        "player_points_pre",
        "elo_pre",
        "elo_opp_pre",
        "tournament_name",
    ]
    cols = [col for col in cols if col in history.columns]
    return history[cols].drop_duplicates().copy()


def player_rating_history(bundle: dict[str, Any], player_id: str) -> pd.DataFrame:
    rating_history = bundle.get("rating_history", pd.DataFrame()).copy()
    if rating_history.empty:
        return pd.DataFrame()

    rni = str(player_id).replace("RNI:", "").strip()
    if not rni:
        return pd.DataFrame()

    required = ["РНИ", "Дата классификации", "Возрастная группа", "Очки", "Место"]
    missing = [col for col in required if col not in rating_history.columns]
    if missing:
        return pd.DataFrame()

    rating_history["rni_norm"] = (
        rating_history["РНИ"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    result = rating_history[rating_history["rni_norm"].eq(rni)].copy()
    if result.empty:
        return pd.DataFrame()

    result = result.rename(
        columns={
            "Дата классификации": "classification_date",
            "Возрастная группа": "age_group",
            "Очки": "points",
            "Место": "rank",
            "Всего турниров": "rated_tournaments",
            "Из них зачетных": "counting_tournaments",
        }
    )
    result["classification_date"] = pd.to_datetime(result["classification_date"], errors="coerce")
    for col in ["points", "rank", "rated_tournaments", "counting_tournaments"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    keep_cols = [
        "classification_date",
        "age_group",
        "rank",
        "points",
        "rated_tournaments",
        "counting_tournaments",
    ]
    keep_cols = [col for col in keep_cols if col in result.columns]
    return (
        result[keep_cols]
        .dropna(subset=["classification_date"])
        .drop_duplicates()
        .sort_values(["classification_date", "age_group"])
        .reset_index(drop=True)
    )


def rating_snapshot(
    bundle: dict[str, Any],
    player_id: str,
    prediction_date: pd.Timestamp,
    age_group: object = None,
) -> dict[str, Any]:
    history = player_rating_history(bundle, player_id)
    if history.empty:
        return {}

    prediction_date = pd.Timestamp(prediction_date)
    history = history[history["classification_date"] <= prediction_date].copy()
    if history.empty:
        return {}

    requested_age = normalize_age_group(age_group)
    if requested_age:
        exact = history[history["age_group"].map(normalize_age_group).eq(requested_age)].copy()
        if not exact.empty:
            history = exact

    latest = history.sort_values(["classification_date", "age_group"]).iloc[-1]
    result: dict[str, Any] = {
        "classification_date": latest.get("classification_date"),
        "age_group": latest.get("age_group"),
        "rank": latest.get("rank"),
        "points": latest.get("points"),
    }
    for col in ["rated_tournaments", "counting_tournaments"]:
        if col in history.columns:
            result[col] = latest.get(col)
    return result


def probability_timeline(
    bundle: dict[str, Any],
    player1_name: str,
    player2_name: str,
    end_date: str | pd.Timestamp,
    periods: int = 12,
) -> pd.DataFrame:
    long_feat = bundle["long_feat"]
    end_date = pd.Timestamp(end_date)
    p1 = resolve_player_id_by_name(long_feat, player1_name)
    p2 = resolve_player_id_by_name(long_feat, player2_name)
    if not p1["found"] or not p2["found"]:
        return pd.DataFrame()

    p1_dates = long_feat.loc[long_feat["player_id"].astype(str).eq(p1["player_id"]), "match_date"]
    p2_dates = long_feat.loc[long_feat["player_id"].astype(str).eq(p2["player_id"]), "match_date"]
    if p1_dates.empty or p2_dates.empty:
        return pd.DataFrame()

    start_date = max(pd.Timestamp(p1_dates.min()), pd.Timestamp(p2_dates.min()))
    if start_date >= end_date:
        dates = [end_date]
    else:
        dates = list(pd.date_range(start_date, end_date, periods=periods))
        dates.append(end_date)

    rows = []
    for date_value in sorted(set(pd.Timestamp(d).normalize() for d in dates)):
        pred = predict_match_by_names(bundle, player1_name, player2_name, date_value)
        if pred.get("ok"):
            rows.append({
                "date": date_value,
                "p_player1_win": pred["p_player1_win"],
                "p_player2_win": pred["p_player2_win"],
            })
    return pd.DataFrame(rows)


def _model_feature_importances(bundle: dict[str, Any]) -> pd.DataFrame:
    model = bundle["model"]
    features = list(bundle["features"])
    fitted = getattr(model, "named_steps", {}).get("model") if hasattr(model, "named_steps") else model
    importances = getattr(fitted, "feature_importances_", None)
    if importances is None and hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()
    if importances is None:
        return pd.DataFrame({"feature": features, "importance": np.ones(len(features))})
    return pd.DataFrame({"feature": features, "importance": importances})


def factor_contribution_table(bundle: dict[str, Any], pred_rows: pd.DataFrame, top_n: int = 14) -> pd.DataFrame:
    features = list(bundle["features"])
    model = bundle["model"]
    base_prob = symmetrize_pair_probs(pred_rows, "p_model_raw")["p_player1_sym"]
    medians = numeric_feature_medians(bundle)
    importance = _model_feature_importances(bundle).sort_values("importance", ascending=False).head(top_n * 2)

    rows = []
    for feature in importance["feature"].tolist():
        neutral_rows = pred_rows.copy()
        neutral_value = medians.get(feature, np.nan)
        if pd.isna(neutral_value):
            continue
        neutral_rows[feature] = neutral_value
        neutral_rows["p_neutral"] = model.predict_proba(neutral_rows[features])[:, 1]
        neutral_prob = symmetrize_pair_probs(neutral_rows, "p_neutral")["p_player1_sym"]
        rows.append({
            "feature": feature,
            "value_player1_row": pred_rows.iloc[0][feature],
            "value_player2_row": pred_rows.iloc[1][feature],
            "neutral_value": neutral_value,
            "probability_effect": base_prob - neutral_prob,
            "abs_probability_effect": abs(base_prob - neutral_prob),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    catalog = bundle.get("feature_catalog", pd.DataFrame())
    if not catalog.empty and {"feature", "group", "meaning"}.issubset(catalog.columns):
        result = result.merge(catalog[["feature", "group", "meaning"]], on="feature", how="left")
    return result.sort_values("abs_probability_effect", ascending=False).head(top_n).reset_index(drop=True)
