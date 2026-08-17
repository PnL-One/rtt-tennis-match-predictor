from __future__ import annotations

import random
import unittest

from rtt_predictor.tournament_simulation import (
    PairProbabilityMatrix,
    TournamentPlayer,
    build_rtt_draw,
    ordinary_olympic_points,
    points_for_outcome_bands,
    seed_count_for,
    simulate_tournament,
)


def players(count: int) -> list[TournamentPlayer]:
    return [
        TournamentPlayer(
            player_id=str(index),
            name=f"Player {index:02d}",
            rank=index,
            points=1000 - index,
        )
        for index in range(1, count + 1)
    ]


class DrawRulesTests(unittest.TestCase):
    def test_seed_counts_follow_rtt_table_6(self) -> None:
        self.assertEqual(seed_count_for(4), 2)
        self.assertEqual(seed_count_for(8), 2)
        self.assertEqual(seed_count_for(9), 4)
        self.assertEqual(seed_count_for(16), 4)
        self.assertEqual(seed_count_for(17), 8)
        self.assertEqual(seed_count_for(32), 8)

    def test_seeded_players_are_separated_in_32_draw(self) -> None:
        draw = build_rtt_draw(players(24), rng=random.Random(7))
        positions = {
            int(player.player_id): row
            for row, player in enumerate(draw)
            if player is not None and int(player.player_id) <= 8
        }
        self.assertEqual(positions[1], 0)
        self.assertEqual(positions[2], 31)
        self.assertEqual({positions[3], positions[4]}, {8, 23})
        self.assertEqual({positions[index] for index in range(5, 9)}, {7, 15, 16, 24})
        for seed in range(1, 9):
            opponent_row = positions[seed] ^ 1
            self.assertIsNone(draw[opponent_row])

    def test_17_player_draw_allocates_all_15_byes(self) -> None:
        draw = build_rtt_draw(players(17), rng=random.Random(11))
        self.assertEqual(len(draw), 32)
        self.assertEqual(sum(player is None for player in draw), 15)
        self.assertEqual(sum(player is not None for player in draw), 17)


class PointsTests(unittest.TestCase):
    def test_known_under_17_iii_b_24_player_points(self) -> None:
        self.assertEqual(
            ordinary_olympic_points("до 17 лет", "III Б", 21),
            {0: 210, 1: 147, 2: 105, 3: 74, 4: 47, 5: 30},
        )
        self.assertEqual(
            points_for_outcome_bands("до 17 лет", "III Б", 21),
            {"1": 210, "2": 147, "3–4": 105, "5–8": 74, "9–16": 47, "17–21": 30},
        )


class SimulationTests(unittest.TestCase):
    def test_deterministic_stronger_player_wins(self) -> None:
        field = players(8)

        def stronger(a: TournamentPlayer, b: TournamentPlayer) -> float:
            return 1.0 if int(a.player_id) < int(b.player_id) else 0.0

        result = simulate_tournament(
            field,
            stronger,
            iterations=50,
            random_seed=4,
            target_player_ids=["1"],
            age_group="до 17 лет",
            tournament_category="IV Б",
        )
        self.assertEqual(result.distributions["1"]["1"], 1.0)
        self.assertEqual(set(result.distributions["1"]), {"1", "2", "3–4", "5–8"})
        self.assertEqual(result.expected_points["1"], 64.0)
        self.assertAlmostEqual(sum(result.distributions["1"].values()), 1.0)

    def test_fixed_completed_match_overrides_probability(self) -> None:
        field = players(4)
        fixed_draw = [field[0], field[1], field[2], field[3]]
        matrix = PairProbabilityMatrix(
            {
                ("1", "2"): 1.0,
                ("3", "4"): 1.0,
                ("1", "3"): 1.0,
                ("2", "3"): 1.0,
            }
        )
        result = simulate_tournament(
            field,
            matrix,
            iterations=10,
            fixed_draw=fixed_draw,
            completed_winners={frozenset(("1", "2")): "2"},
            target_player_ids=["2"],
        )
        self.assertEqual(result.distributions["2"]["1"], 1.0)
        self.assertEqual(result.encounter_probabilities["2"], {"3": 1.0})


if __name__ == "__main__":
    unittest.main()
