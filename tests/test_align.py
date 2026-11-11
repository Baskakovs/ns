"""Tests for the panel shaping: the row grid, the re-dating, the PIT mask and the wide frames."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from novetly_score.align import entry_day, pit_members, tradable, trading_days, wide


def _prices(rows: list[tuple]) -> pl.DataFrame:
    """A long price frame from ``(ric, day, close_price)`` rows."""
    return pl.DataFrame(rows, schema=["ric", "day", "close_price"], orient="row").with_columns(
        pl.col("day").str.to_date()
    )


def _counts(rows: list[tuple]) -> pl.DataFrame:
    """A long counts frame from ``(day, ric, positive, neutral, negative)`` rows."""
    return pl.DataFrame(rows, schema=["day", "ric", "positive", "neutral", "negative"], orient="row").with_columns(
        pl.col("day").str.to_date()
    )


def _days(*days: str) -> pl.Series:
    """A trading-day grid from ISO date strings."""
    return pl.Series("day", list(days)).str.to_date()


class TestTradingDays:
    """The row grid comes from the price feed, not the calendar."""

    def test_distinct_sorted_and_bounded(self):
        """Duplicates collapse, order is ascending, and both ends are inclusive."""
        frame = _prices(
            [
                ("B.DE", "2025-01-03", 2.0),
                ("A.DE", "2025-01-02", 1.0),
                ("B.DE", "2025-01-02", 2.0),
                ("A.DE", "2025-01-06", 1.0),
            ]
        )

        days = trading_days(frame, date(2025, 1, 2), date(2025, 1, 3))

        assert days.to_list() == _days("2025-01-02", "2025-01-03").to_list()


class TestEntryDay:
    """News prices into the next trading day, never its own."""

    def test_weekend_news_lands_on_monday_summed(self):
        """Friday, Saturday and Sunday all feed Monday, and their counts add."""
        days = _days("2025-03-14", "2025-03-17", "2025-03-18")  # Fri, Mon, Tue
        frame = _counts(
            [
                ("2025-03-14", "A.DE", 1, 0, 0),  # Friday -> Monday
                ("2025-03-15", "A.DE", 2, 0, 0),  # Saturday -> Monday
                ("2025-03-16", "A.DE", 4, 0, 0),  # Sunday -> Monday
                ("2025-03-17", "A.DE", 8, 0, 0),  # Monday -> Tuesday
            ]
        )

        out = entry_day(frame, days)

        assert out.filter(pl.col("day") == pl.date(2025, 3, 17)).item(0, "positive") == 7
        assert out.filter(pl.col("day") == pl.date(2025, 3, 18)).item(0, "positive") == 8

    def test_every_calendar_day_lands_on_the_next_trading_day(self):
        """The defining property, checked day by day: news prices into the first close after it."""
        days = _days("2025-03-14", "2025-03-17", "2025-03-18", "2025-03-19")  # Fri, Mon, Tue, Wed
        frame = _counts(
            [(d, "A.DE", 1, 0, 0) for d in ("2025-03-14", "2025-03-15", "2025-03-16", "2025-03-17", "2025-03-18")]
        )

        out = entry_day(frame, days)

        assert out.select("day", "positive").rows() == [
            (date(2025, 3, 17), 3),  # Fri + Sat + Sun
            (date(2025, 3, 18), 1),  # Mon
            (date(2025, 3, 19), 1),  # Tue
        ]
        assert out.get_column("positive").sum() == frame.get_column("positive").sum(), "nothing lost or doubled"

    def test_news_after_the_last_trading_day_is_dropped(self):
        """It has no close to price into, so it is absent rather than folded backwards."""
        days = _days("2025-03-17", "2025-03-18")
        frame = _counts([("2025-03-18", "A.DE", 5, 0, 0)])

        assert entry_day(frame, days).height == 0

    def test_companies_stay_separate(self):
        """The sum is per company, not per day."""
        days = _days("2025-03-14", "2025-03-17")
        frame = _counts([("2025-03-14", "A.DE", 1, 0, 0), ("2025-03-14", "B.DE", 9, 0, 0)])

        out = entry_day(frame, days).sort("ric")

        assert out.get_column("positive").to_list() == [1, 9]


class TestPitMembers:
    """Membership is the review in force, in that review's own symbols."""

    def _membership(self) -> pl.DataFrame:
        """Two reviews, a company that changed RIC between them, and a non-member."""
        return pl.DataFrame(
            {
                "effective_date": ["2025-01-02", "2025-01-02", "2025-03-03", "2025-03-03"],
                "ric": ["OLD.VX", "B.DE", "NEW.S", "C.DE"],
                "ric_current": ["NEW.S", "B.DE", "NEW.S", "C.DE"],
                "is_member": [True, True, True, True],
                "member_rank": [1, 2, 1, 2],
            }
        ).with_columns(pl.col("effective_date").str.to_date())

    def test_a_day_takes_the_review_in_force(self):
        """Before the March review, January's composition and January's symbols."""
        out = pit_members(self._membership(), _days("2025-02-01", "2025-03-10"))

        assert set(out.filter(pl.col("day") == pl.date(2025, 2, 1)).get_column("ric")) == {"OLD.VX", "B.DE"}
        assert set(out.filter(pl.col("day") == pl.date(2025, 3, 10)).get_column("ric")) == {"NEW.S", "C.DE"}

    def test_the_point_in_time_symbol_is_used_not_the_current_one(self):
        """A February day carries OLD.VX even though the company trades as NEW.S today."""
        out = pit_members(self._membership(), _days("2025-02-01"))

        assert "OLD.VX" in out.get_column("ric").to_list()
        assert "NEW.S" not in out.get_column("ric").to_list()

    def test_days_before_the_first_review_are_dropped(self):
        """There is no composition to assert, so the day is absent rather than guessed."""
        assert pit_members(self._membership(), _days("2024-12-31")).height == 0

    def test_top_takes_the_rank_in_force(self):
        """``top`` cuts on the review's own ranking."""
        out = pit_members(self._membership(), _days("2025-02-01"), top=1)

        assert out.get_column("ric").to_list() == ["OLD.VX"]

    def test_non_members_are_excluded(self):
        """A row flagged is_member=False names no membership."""
        frame = self._membership().with_columns(is_member=pl.lit(False))

        assert pit_members(frame, _days("2025-02-01")).height == 0


class TestWide:
    """One column per company, and a missing cell that stays missing."""

    def test_shape_and_order_follow_the_arguments(self):
        """The grid is exactly the days and RICs asked for, in that order."""
        frame = _counts([("2025-01-02", "A.DE", 1, 0, 0)])

        out = wide(frame, "positive", _days("2025-01-02", "2025-01-03"), ["B.DE", "A.DE"])

        assert out.columns == ["date", "B.DE", "A.DE"], "basanos requires the key column be 'date'"
        assert out.height == 2

    def test_an_absent_cell_is_null_not_zero(self):
        """The distinction the whole measurement rests on: unobserved is not zero."""
        frame = _counts([("2025-01-02", "A.DE", 0, 0, 0)])

        out = wide(frame, "positive", _days("2025-01-02", "2025-01-03"), ["A.DE", "B.DE"])

        assert out.item(0, "A.DE") == 0.0, "an observed zero count is a zero"
        assert out.item(1, "A.DE") is None, "a day with no row is null"
        assert out.item(0, "B.DE") is None, "a company with no rows at all is null"

    def test_values_are_floats(self):
        """Integer counts become floats, so a null reads as NaN downstream."""
        frame = _counts([("2025-01-02", "A.DE", 3, 0, 0)])

        assert wide(frame, "positive", _days("2025-01-02"), ["A.DE"]).schema["A.DE"] == pl.Float64


class TestTradable:
    """Which price columns are usable as return series."""

    def _wide(self, **columns: list[float | None]) -> pl.DataFrame:
        """A wide price frame from named columns."""
        days = _days(*[f"2025-01-{d:02d}" for d in range(1, len(next(iter(columns.values()))) + 1)])
        return pl.DataFrame({"date": days, **columns}).with_columns(pl.col(list(columns)).cast(pl.Float64))

    def test_keeps_a_normal_series(self):
        """Positive, varying and present."""
        assert tradable(self._wide(A=[10.0, 11.0, 10.5, 11.2]), ["A"]) == ["A"]

    @pytest.mark.parametrize(
        ("name", "values"),
        [
            ("non_positive", [10.0, 0.0, 11.0, 12.0]),
            ("monotone_up", [1.0, 2.0, 3.0, 4.0]),
            ("monotone_down", [4.0, 3.0, 2.0, 1.0]),
            ("constant", [5.0, 5.0, 5.0, 5.0]),
            ("too_short", [10.0, None, None, 11.0]),
        ],
    )
    def test_rejects(self, name, values):
        """Each rejection is one a basanos engine would otherwise raise on."""
        assert tradable(self._wide(**{name: values}), [name]) == []

    def test_rejects_a_column_too_sparse(self):
        """More than max_null missing leaves nothing to measure against."""
        frame = self._wide(A=[10.0, None, None, 11.5, 10.8])  # 3 quotes of 5 rows, 40% null

        assert tradable(frame, ["A"], max_null=0.9) == ["A"]
        assert tradable(frame, ["A"], max_null=0.3) == []

    def test_order_is_preserved(self):
        """The caller's column order survives, so it can index the wide frames."""
        frame = self._wide(A=[10.0, 11.0, 10.5, 11.2], B=[20.0, 19.0, 21.0, 20.5])

        assert tradable(frame, ["B", "A"]) == ["B", "A"]
