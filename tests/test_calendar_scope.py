from __future__ import annotations

from datetime import date
import unittest

from scripts.parse_rtt_calendar import default_calendar_date_to


class CalendarScopeTests(unittest.TestCase):
    def test_default_calendar_scope_keeps_announced_future_tournaments(self) -> None:
        self.assertEqual(
            default_calendar_date_to(date(2026, 8, 17)),
            date(2027, 2, 13),
        )

    def test_future_horizon_can_be_overridden(self) -> None:
        self.assertEqual(
            default_calendar_date_to(date(2026, 8, 17), future_days=30),
            date(2026, 9, 16),
        )

    def test_future_horizon_cannot_be_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            default_calendar_date_to(date(2026, 8, 17), future_days=-1)


if __name__ == "__main__":
    unittest.main()
