from __future__ import annotations

import unittest
from unittest.mock import patch
from datetime import date

import pandas as pd

from rtt_predictor.tournament_data import ParsedMatch, ParsedPlayer, TournamentSnapshot
from rtt_predictor.tournament_mode import (
    _completed_winners,
    _find_target,
    cached_registered_tour_ids,
    current_rating_age_groups,
    eligible_tour_ids_from_master,
    prepare_registration_scenario,
)
from rtt_predictor.tournament_simulation import TournamentPlayer


class TargetPlayerLookupTests(unittest.TestCase):
    def test_abbreviated_initials_match_full_rtt_name(self) -> None:
        players = [
            TournamentPlayer("1", "Рассудова Мария Павловна"),
            TournamentPlayer("2", "Рыжикова Анна Антоновна"),
        ]
        target = _find_target(players, "Рыжикова А.А.")
        self.assertEqual(target.player_id, "2")

    def test_source_name_is_used_after_model_name_resolution(self) -> None:
        players = [
            TournamentPlayer(
                "2",
                "Анна Рыжикова",
                metadata={"source_player_name": "Рыжикова Анна Антоновна"},
            )
        ]
        target = _find_target(players, "Рыжикова А.А.")
        self.assertEqual(target.player_id, "2")

    def test_same_surname_and_initials_remain_ambiguous(self) -> None:
        players = [
            TournamentPlayer("1", "Рыжикова Анна Антоновна"),
            TournamentPlayer("2", "Рыжикова Алина Андреевна"),
        ]
        with self.assertRaisesRegex(KeyError, "неоднозначно"):
            _find_target(players, "Рыжикова А.А.")


class CompletedMatchLookupTests(unittest.TestCase):
    def test_abbreviated_match_names_lock_winner_with_full_roster_names(self) -> None:
        players = [
            TournamentPlayer("RNI:44228", "Новикова Ангелина Сергеевна"),
            TournamentPlayer("RNI:49799", "Рыжикова Анна Антоновна"),
        ]
        snapshot = TournamentSnapshot(
            tour_id="306306",
            fetched_at="2026-08-17T08:00:00+03:00",
            completed_matches=[
                ParsedMatch(
                    player1="Новикова А.С.",
                    player2="Рыжикова А.А.",
                    winner="Рыжикова А.А.",
                    score="0 - 2 0-6,4-6",
                )
            ],
        )

        completed, warnings = _completed_winners(snapshot, players)

        self.assertEqual(
            completed,
            {frozenset(("RNI:44228", "RNI:49799")): "RNI:49799"},
        )
        self.assertEqual(warnings, [])

    def test_later_grid_round_infers_walkover_winner(self) -> None:
        players = [
            TournamentPlayer("RNI:51572", "Лисица Анна Сергеевна"),
            TournamentPlayer("RNI:49372", "Воронова Ирина Алексеевна"),
            TournamentPlayer("RNI:44228", "Новикова Ангелина Сергеевна"),
            TournamentPlayer("RNI:49799", "Рыжикова Анна Антоновна"),
        ]
        snapshot = TournamentSnapshot(
            tour_id="306306",
            fetched_at="2026-08-17T20:55:09+03:00",
            grid_rounds=[
                ["RNI:51572", "RNI:49372", "RNI:44228", "RNI:49799"],
                ["RNI:51572", "RNI:49799"],
            ],
            completed_matches=[
                ParsedMatch(
                    player1="Новикова А.С.",
                    player2="Рыжикова А.А.",
                    winner="Рыжикова А.А.",
                    score="0 - 2 0-6,4-6",
                )
            ],
        )

        completed, warnings = _completed_winners(snapshot, players)

        self.assertEqual(
            completed,
            {
                frozenset(("RNI:44228", "RNI:49799")): "RNI:49799",
                frozenset(("RNI:51572", "RNI:49372")): "RNI:51572",
            },
        )
        self.assertEqual(warnings, [])


class RegistrationScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = {
            "long_feat": pd.DataFrame(
                {
                    "player_id": ["RNI:99"],
                    "player_name": ["Новая Анна Андреевна"],
                    "match_date": [pd.Timestamp("2026-01-01")],
                }
            )
        }

    @staticmethod
    def snapshot(*, grid: bool = False, players: int = 4, capacity: int = 8) -> TournamentSnapshot:
        roster = [
            ParsedPlayer(name=f"Игрок {index}", rni=str(index), rank=float(index))
            for index in range(1, players + 1)
        ]
        return TournamentSnapshot(
            tour_id="1",
            fetched_at="2026-08-16T12:00:00+03:00",
            title="Тест",
            status="Подача заявок",
            age_group="до 15 лет",
            draw_system="Олимпийская",
            start_date="2026-08-20",
            main_draw_capacity=capacity,
            players=roster,
            player_source="projected_from_requests",
            grid_slots=[player.player_id for player in roster] if grid else [],
            eligible=True,
        )

    def test_existing_entry_is_kept_even_with_published_draw(self) -> None:
        snapshot = self.snapshot(grid=True)
        result = prepare_registration_scenario(self.bundle, snapshot, "Игрок 1")
        self.assertIs(result, snapshot)

    @patch("rtt_predictor.tournament_mode.prediction.rating_snapshot")
    def test_missing_player_is_added_with_current_rating(self, rating_snapshot) -> None:
        rating_snapshot.return_value = {"rank": 7, "points": 321}
        result = prepare_registration_scenario(self.bundle, self.snapshot(), "Новая Анна Андреевна")
        added = next(player for player in result.players if player.name == "Новая Анна Андреевна")
        self.assertEqual((added.rank, added.points), (7.0, 321.0))
        self.assertIn("virtual_registration", result.player_source)

    def test_missing_player_cannot_be_added_after_draw_publication(self) -> None:
        with self.assertRaisesRegex(ValueError, "сетка уже опубликована"):
            prepare_registration_scenario(self.bundle, self.snapshot(grid=True), "Новая Анна Андреевна")

    def test_missing_player_cannot_be_added_after_applications_close(self) -> None:
        snapshot = self.snapshot()
        snapshot.status = "Формирование состава"
        with self.assertRaisesRegex(ValueError, "не допускает новую заявку"):
            prepare_registration_scenario(self.bundle, snapshot, "Новая Анна Андреевна")

    def test_missing_player_cannot_be_added_with_unknown_application_status(self) -> None:
        snapshot = self.snapshot()
        snapshot.status = "unknown"
        with self.assertRaisesRegex(ValueError, "не допускает новую заявку"):
            prepare_registration_scenario(self.bundle, snapshot, "Новая Анна Андреевна")

    @patch("rtt_predictor.tournament_mode.prediction.rating_snapshot")
    def test_virtual_entry_must_qualify_for_full_application_list(self, rating_snapshot) -> None:
        rating_snapshot.return_value = {"rank": 9999, "points": 0}
        with self.assertRaisesRegex(ValueError, "не проходит в ОТ"):
            prepare_registration_scenario(
                self.bundle,
                self.snapshot(players=4, capacity=4),
                "Новая Анна Андреевна",
            )


class MegaTournamentDiscoveryTests(unittest.TestCase):
    @patch("rtt_predictor.tournament_mode.load_snapshot")
    @patch("rtt_predictor.tournament_mode.Path.glob")
    def test_registered_cached_tournament_is_discovered_outside_master(self, glob, load) -> None:
        snapshot_path = type("SnapshotPath", (), {})()
        snapshot_path.parent = type("Parent", (), {"name": "306306"})()
        glob.return_value = [snapshot_path]
        load.return_value = TournamentSnapshot(
            tour_id="306306",
            fetched_at="2026-08-16T12:00:00+03:00",
            start_date="2026-08-17",
            players=[ParsedPlayer(name="Рыжикова Анна Антоновна", rni="49799")],
            eligible=True,
        )
        self.assertEqual(
            cached_registered_tour_ids("unused", "Рыжикова А.А."),
            ["306306"],
        )

    @patch("rtt_predictor.tournament_mode.prediction.player_rating_history")
    @patch("rtt_predictor.tournament_mode.prediction.resolve_player_id_by_name")
    def test_all_groups_from_latest_classification_are_returned(self, resolve, history) -> None:
        resolve.return_value = {"found": True, "player_id": "RNI:99"}
        history.return_value = pd.DataFrame(
            {
                "classification_date": pd.to_datetime(
                    ["2026-07-01", "2026-08-01", "2026-08-01", "2026-08-01"]
                ),
                "age_group": ["до 15 лет", "до 17 лет", "до 19 лет", "взрослые"],
            }
        )
        self.assertEqual(
            current_rating_age_groups(
                {"long_feat": pd.DataFrame()},
                "Новая А.А.",
                today=date(2026, 8, 16),
            ),
            ["до 17 лет", "до 19 лет", "взрослые"],
        )

    @patch("rtt_predictor.tournament_mode.pd.read_excel")
    def test_master_filter_accepts_multiple_age_groups(self, read_excel) -> None:
        read_excel.return_value = pd.DataFrame(
            {
                "tour_id": [306306, 306999, 306888, 307500, 307600, 307700],
                "start_date": [
                    "2026-08-17",
                    "2026-08-18",
                    "2026-08-19",
                    "2026-12-20",
                    "2026-09-01",
                    "2026-09-02",
                ],
                "status": [
                    "Прием заявок",
                    "Подача поздних заявок",
                    "Прием заявок",
                    "unknown",
                    "Формирование состава",
                    "В процессе проведения",
                ],
                "age_category": [
                    "до 17 лет",
                    "до 19 лет",
                    "до 15 лет",
                    "до 17 лет",
                    "до 17 лет",
                    "до 17 лет",
                ],
            }
        )
        result = eligible_tour_ids_from_master(
            "unused.xlsx",
            today=date(2026, 8, 16),
            age_group=["до 17 лет", "до 19 лет"],
        )
        self.assertEqual(result, ["306306", "306999", "307500"])


if __name__ == "__main__":
    unittest.main()
