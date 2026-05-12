from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import prediction_runtime as pr


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value)


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    if limit is not None:
        out = out.head(limit)
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    out = out.replace({np.nan: None})
    return out.to_dict(orient="records")


def _write_json(payload: dict[str, Any]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, default=_json_default))


def players_command(_: argparse.Namespace) -> None:
    bundle = pr.load_prediction_bundle()
    long_feat = bundle["long_feat"]
    payload = {
        "ok": True,
        "bundle_path": bundle.get("bundle_path"),
        "model_name": bundle.get("model_name"),
        "max_match_date": pd.Timestamp(long_feat["match_date"].max()).date().isoformat(),
        "players": pr.player_options(bundle),
    }
    _write_json(payload)


def predict_command(args: argparse.Namespace) -> None:
    bundle = pr.load_prediction_bundle()
    result = pr.predict_match_by_names(
        bundle,
        args.player1,
        args.player2,
        args.date,
        context={
            "tournament_age_category": args.age,
            "draw_type": args.draw_type,
            "tournament_name": "__USER_PREDICTION__",
            "tournament_city": "__UNKNOWN_CITY__",
        },
    )

    if not result.get("ok"):
        _write_json({
            "ok": False,
            "message": result.get("message", "Prediction failed."),
            "player1_candidates": _records(result.get("player1_lookup", {}).get("candidates", pd.DataFrame())),
            "player2_candidates": _records(result.get("player2_lookup", {}).get("candidates", pd.DataFrame())),
        })
        return

    player1_id = result["player1_lookup"]["player_id"]
    player2_id = result["player2_lookup"]["player_id"]
    date_value = pd.Timestamp(args.date)
    timeline = pr.probability_timeline(bundle, args.player1, args.player2, date_value, periods=args.periods)
    player1_history = pr.player_history(bundle, player1_id, pd.Timestamp.max)
    player2_history = pr.player_history(bundle, player2_id, pd.Timestamp.max)

    payload = {
        "ok": True,
        "model_name": result.get("model_name"),
        "player1_name": result["player1_lookup"]["player_name"],
        "player2_name": result["player2_lookup"]["player_name"],
        "player1_id": player1_id,
        "player2_id": player2_id,
        "p_player1_win": result["p_player1_win"],
        "p_player2_win": result["p_player2_win"],
        "profiles": _records(pr.profiles_table(result)),
        "factor_contributions": _records(result.get("factor_contributions", pd.DataFrame())),
        "prediction_rows": _records(result.get("prediction_rows", pd.DataFrame())),
        "timeline": _records(timeline),
        "player1_history": _records(player1_history),
        "player2_history": _records(player2_history),
    }
    _write_json(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prediction CLI for the RTT control panel.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    players = subparsers.add_parser("players", help="List player names from the prediction bundle.")
    players.set_defaults(func=players_command)

    predict = subparsers.add_parser("predict", help="Predict a match and return analytics as JSON.")
    predict.add_argument("--player1", required=True)
    predict.add_argument("--player2", required=True)
    predict.add_argument("--date", required=True)
    predict.add_argument("--age", default="__UNKNOWN_AGE__")
    predict.add_argument("--draw-type", default="__UNKNOWN_DRAW__")
    predict.add_argument("--periods", type=int, default=28)
    predict.set_defaults(func=predict_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
