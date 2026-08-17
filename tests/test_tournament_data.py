from __future__ import annotations

from datetime import date
import json
import unittest
from unittest.mock import MagicMock, patch

from rtt_predictor.tournament_data import (
    build_snapshot_from_pages,
    load_snapshot,
    parse_metadata,
    parse_grid_slots,
    parse_players,
    tournament_eligibility,
)


class TournamentDataTests(unittest.TestCase):
    def test_metadata_and_terminal_status(self) -> None:
        html = """
        <html><body><h1>Кубок Тестовый турнир</h1>
        <div>Тестовый турнир 03.08.2026 - 09.08.2026 Рег. номер 305991</div>
        <span>Сдача отчета</span><div>Категория турнира: III Б</div>
        <div>Разряд турнира: Одиночный</div><div>Пол игроков: Женский</div>
        <div>Возрастная группа: до 17 лет</div><div>Система проведения: Олимпийская</div>
        <div>Формат проведения: Недельный</div><div>Кол-во участников: ОТ: 24</div>
        </body></html>
        """
        metadata = parse_metadata(html, "305991")
        self.assertEqual(metadata["category"], "III Б")
        self.assertEqual(metadata["age_group"], "до 17 лет")
        self.assertEqual(metadata["main_draw_capacity"], 24)
        self.assertEqual(metadata["status"], "Сдача отчета")
        eligible, _ = tournament_eligibility(
            metadata["status"], metadata["start_date"], metadata["end_date"], today=date(2026, 8, 8)
        )
        self.assertFalse(eligible)

    def test_player_table(self) -> None:
        html = """
        <table><thead><tr><th>Игрок</th><th>РНИ</th><th>Место</th><th>Очки</th><th>Статус заявки</th></tr></thead>
        <tbody>
        <tr><td><a href="/public/players/91">Иванова И.И.</a></td><td>12345</td><td>7</td><td>880</td><td>Основной состав</td></tr>
        <tr><td><a href="/public/players/92">Петрова П.П.</a></td><td>12346</td><td>11</td><td>700</td><td>Основной состав</td></tr>
        </tbody></table>
        """
        result = parse_players(html)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].player_id, "RNI:12345")
        self.assertEqual(result[0].rank, 7)

    def test_current_rtt_request_table_uses_note_as_status(self) -> None:
        html = """
        <table><thead><tr>
        <th>№</th><th>РНИ</th><th>Участник, город, дата рождения</th>
        <th>Заяв. рейтинг (турниров)</th><th>Рейтинг рассеивания</th>
        <th>Дата заявки</th><th>Примечание</th>
        </tr></thead><tbody><tr>
        <td>1</td><td>42738</td>
        <td><a href="/public/ranking/solo/92124">Суркова Майя Сергеевна</a> Москва, 01.06.2010</td>
        <td>195 (22)</td><td>186</td><td>15.07.2026 10:58</td><td>Поздняя заявка</td>
        </tr></tbody></table>
        """
        result = parse_players(html)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "Суркова Майя Сергеевна")
        self.assertEqual(result[0].source_id, "92124")
        self.assertEqual(result[0].rni, "42738")
        self.assertEqual(result[0].rank, 195)
        self.assertEqual(result[0].request_status, "Поздняя заявка")

    def test_current_div_grid_matches_abbreviated_players(self) -> None:
        players = parse_players("""
        <table><tr><th>Участник</th><th>РНИ</th></tr>
        <tr><td>Новикова Ангелина Сергеевна</td><td>44228</td></tr>
        <tr><td>Рыжикова Анна Антоновна</td><td>49799</td></tr>
        <tr><td>Лисица Анна Сергеевна</td><td>51572</td></tr>
        </table>
        """)
        html = """
        <div class="cell-wrapper round0">
          <div class="TourGridCell cell-pointer">
            <div class="cell-player cell-top">8 Новикова А.С. BLR</div>
            <div class="cell-player cell-bottom">6 Рыжикова А.А. Москва</div>
          </div>
          <div class="TourGridCell cell-pointer-print">
            <div class="cell-player cell-top">8 Новикова А.С. BLR</div>
            <div class="cell-player cell-bottom">6 Рыжикова А.А. Москва</div>
          </div>
        </div>
        <div class="cell-wrapper round0">
          <div class="TourGridCell cell-pointer">
            <div class="cell-player cell-top">4 СИ Лисица А.С. BLR</div>
            <div class="cell-player cell-bottom">X</div>
          </div>
        </div>
        """
        self.assertEqual(
            parse_grid_slots(html, players),
            ["RNI:44228", "RNI:49799", "RNI:51572", None],
        )

    def test_snapshot_projects_requests_when_members_absent(self) -> None:
        dashboard = """
        <div>Тест 20.08.2026 - 25.08.2026 Рег. номер 300001</div>
        <span>Прием заявок</span><div>Категория турнира: IV Б</div><div>Разряд турнира: Одиночный</div>
        <div>Пол игроков: Женский</div><div>Возрастная группа: до 17 лет</div>
        <div>Система проведения: Олимпийская</div><div>Формат проведения: Недельный</div>
        <div>Кол-во участников: ОТ: 4</div>
        """
        requests = """<table><tr><th>Игрок</th><th>РНИ</th><th>Место</th></tr>
        <tr><td>А А.А.</td><td>1</td><td>1</td></tr><tr><td>Б Б.Б.</td><td>2</td><td>2</td></tr>
        <tr><td>В В.В.</td><td>3</td><td>3</td></tr><tr><td>Г Г.Г.</td><td>4</td><td>4</td></tr>
        <tr><td>Д Д.Д.</td><td>5</td><td>5</td></tr></table>"""
        snapshot = build_snapshot_from_pages(
            "300001", {"dashboard": dashboard, "requests": requests}, today=date(2026, 8, 16)
        )
        self.assertTrue(snapshot.eligible)
        self.assertEqual(snapshot.player_source, "projected_from_requests")
        self.assertEqual(len(snapshot.players), 4)

    def test_empty_application_shell_snapshot_is_rejected(self) -> None:
        path = MagicMock()
        path.exists.return_value = True
        path.read_text.return_value = json.dumps(
            {"tour_id": "306306", "fetched_at": "2026-08-16T16:00:00+03:00"}
        )
        with patch("rtt_predictor.tournament_data.snapshot_path", return_value=path):
            with self.assertRaisesRegex(ValueError, "оболочку страницы"):
                load_snapshot("unused", "306306")


if __name__ == "__main__":
    unittest.main()
