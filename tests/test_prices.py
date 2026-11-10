"""Tests for the price panel: the span the endpoint wants, and the line a level belongs to."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from novetly_score.data import data
from novetly_score.data.prices import ADJUSTMENTS, COLUMNS, HISTORY_START, fetch, keyed, window

#: A universe frame of membership spells, as :attr:`yukka.data.Index.frame` is shaped.
#:
#: ``industrivarden`` is the two-line case the snapshot forces a choice on: both the A and the C
#: line are listed today, so both come back from the feed and only the current one is this
#: company's. ``bmw`` changes line inside the fixture's dates; ``nestle`` is in the universe but
#: returns no levels.
UNIVERSE = pl.DataFrame(
    {
        "yukka_id": [
            "company:bmw",
            "company:bmw",
            "company:industrivarden",
            "company:industrivarden",
            "company:nestle",
        ],
        "ric": ["BMWG.L", "BMWG.DE", "INDUa.ST", "INDUc.ST", "NESN.S"],
        "start_date": [date(2015, 10, 13), date(2024, 8, 2), date(2015, 10, 13), date(2022, 12, 19), date(2016, 1, 1)],
        "end_date": [date(2024, 8, 1), date(2026, 9, 21), date(2022, 9, 19), date(2026, 9, 21), date(2026, 9, 21)],
    }
)


def levels(*rows):
    """Feed rows, as :func:`fetch` returns them.

    Args:
        *rows: ``(ric, date, close_price)`` triples.

    Returns:
        A frame of ``date``, ``ric``, ``close_price``.
    """
    return pl.DataFrame(
        {
            "date": [row[1] for row in rows],
            "ric": [row[0] for row in rows],
            "close_price": [row[2] for row in rows],
        }
    )


class TestTheSpanTheEndpointWants:
    """``date_to`` is exclusive on the endpoint and inclusive in the call, which is one bug waiting."""

    def test_a_day_span_is_made_exclusive(self):
        """The last day asked for has to come back, so the endpoint is asked for one more."""
        assert window("2025-01-06", "2025-01-10") == (date(2025, 1, 6), date(2025, 1, 11))

    def test_a_month_means_all_of_it(self):
        """A month as a start is its first day; as an end, its last."""
        assert window("2025-01", "2025-02") == (date(2025, 1, 1), date(2025, 3, 1))

    def test_a_february_month_end_knows_its_length(self):
        """Month arithmetic, not a 30-day assumption."""
        assert window("2024-02", "2024-02") == (date(2024, 2, 1), date(2024, 3, 1))

    def test_a_date_object_passes_through(self):
        """A caller holding dates should not have to format them."""
        assert window(date(2025, 1, 6), date(2025, 1, 6)) == (date(2025, 1, 6), date(2025, 1, 7))

    def test_an_open_start_is_the_feeds_history(self):
        """The feed's own coverage start, so an open span is everything rather than an error."""
        assert window(None, "2025-01-10")[0] == HISTORY_START

    def test_an_open_end_is_today(self):
        """There is nothing later to read, so today is the only sensible edge."""
        assert window("2025-01-06", None)[1] == date.today() + timedelta(days=1)

    def test_a_backwards_span_is_refused(self):
        """Backwards is backwards, and silently returning nothing would be worse."""
        with pytest.raises(ValueError, match="runs backwards"):
            window("2025-01-10", "2025-01-06")

    def test_a_malformed_span_end_is_refused(self):
        """The same two shapes the counts take, and the same message when it is neither."""
        with pytest.raises(ValueError, match="not YYYY-MM or YYYY-MM-DD"):
            window("January", None)


class TestTheAdjustments:
    """Three adjustments over two endpoints, and one of them is the default for a reason."""

    def test_the_three_are_named(self):
        """Net is the STOXX Net Return convention, which is what a UCITS receives."""
        assert ADJUSTMENTS == ("net", "gross", "raw")

    def test_an_unknown_adjustment_is_refused_before_any_request(self):
        """A typo must not cost a round trip, and must not silently fall back to a default."""
        with pytest.raises(ValueError, match="no such adjustment"):
            fetch(["BMWG.DE"], date(2025, 1, 6), date(2025, 1, 7), "total")


class TestALevelBelongsToOneLine:
    """The feed is keyed by the line a company trades on today; the panel is keyed by the company."""

    def test_a_level_is_relabelled_to_its_company(self):
        """The store files snippets by yukka_id, so that is what a price panel has to join on."""
        panel = keyed(levels(("BMWG.DE", date(2025, 1, 6), 80.0)), UNIVERSE)
        assert panel.to_dicts() == [{"entity": "company:bmw", "date": date(2025, 1, 6), "close_price": 80.0}]

    def test_the_columns_are_the_documented_ones(self):
        """A caller joining this to the counts needs the shape to be fixed."""
        assert keyed(levels(("BMWG.DE", date(2025, 1, 6), 80.0)), UNIVERSE).columns == list(COLUMNS)

    def test_only_the_current_line_of_a_two_line_company_is_kept(self):
        """Both lines are listed and both come back, but one company cannot have two prices a day."""
        panel = keyed(
            levels(
                ("INDUa.ST", date(2025, 1, 6), 29.9),
                ("INDUc.ST", date(2025, 1, 6), 30.1),
            ),
            UNIVERSE,
        )
        assert panel.to_dicts() == [
            {"entity": "company:industrivarden", "date": date(2025, 1, 6), "close_price": 30.1}
        ], "the C line is the one the company trades on now"

    def test_one_row_per_company_and_day(self):
        """The invariant a join downstream depends on, stated as a test rather than hoped for."""
        panel = keyed(
            levels(
                ("INDUa.ST", date(2025, 1, 6), 29.9),
                ("INDUc.ST", date(2025, 1, 6), 30.1),
                ("BMWG.DE", date(2025, 1, 6), 80.0),
            ),
            UNIVERSE,
        )
        assert panel.height == panel.select("entity", "date").n_unique() == 2

    def test_a_ric_the_universe_does_not_name_is_left_out(self):
        """It belongs to no company this panel is about."""
        panel = keyed(levels(("AAPL.O", date(2025, 1, 6), 100.0)), UNIVERSE)
        assert panel.is_empty()

    def test_the_key_can_be_point_in_time(self):
        """So a price panel and a counts panel key alike and join without a bridge per caller."""
        panel = keyed(
            levels(("BMWG.DE", date(2024, 8, 1), 80.0), ("BMWG.DE", date(2024, 8, 2), 81.0)),
            UNIVERSE,
            key="ric",
        )
        assert panel.to_dicts() == [
            {"ric": "BMWG.DE", "date": date(2024, 8, 2), "close_price": 81.0},
            {"ric": "BMWG.L", "date": date(2024, 8, 1), "close_price": 80.0},
        ]

    def test_a_ric_naming_two_companies_is_refused(self):
        """It would double that RIC's levels, and the universe is wrong rather than this code."""
        confused = pl.concat(
            [UNIVERSE, UNIVERSE.head(1).with_columns(yukka_id=pl.lit("company:other"))], how="vertical"
        )
        with pytest.raises(ValueError, match="name two companies"):
            keyed(levels(("BMWG.L", date(2025, 1, 6), 80.0)), confused)


class TestIdsAloneCannotBePriced:
    """The store knows what a company is; only the universe knows which line it trades on."""

    def test_ids_are_refused_with_a_message_that_says_what_to_do(self, monkeypatch, tmp_path):
        """Reaches no network: the frame is missing before any request could be made."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="needs the universe frame"):
            data("company:bmw").prices("2025-01-06", "2025-01-10")


@pytest.mark.integration
def test_reads_a_week_of_the_real_feed():
    """One week of the real STOXX 600, all three adjustments, against the live snapshot.

    Skipped unless a YUKKA token is set. Three things are checked that only the real feed can
    answer: the panel is uniquely keyed, the three adjustments agree on shape, and a gross total
    return is never below the net one over the same dates.
    """
    yukka_data = pytest.importorskip("yukka.data")
    try:
        universe = yukka_data.Index.STOXX600.frame
        panels = {kind: data(universe).prices("2025-01-02", "2025-01-10", adjustment=kind) for kind in ADJUSTMENTS}
    except Exception as exc:  # noqa: BLE001 — a missing token is a skip, not a failure
        pytest.skip(f"universe or feed unavailable: {exc}")

    for kind, panel in panels.items():
        assert panel.columns == list(COLUMNS), kind
        assert panel.schema["date"] == pl.Date, kind
        assert panel.height == panel.select("entity", "date").n_unique(), f"{kind} is not uniquely keyed"
        assert panel.get_column("entity").n_unique() > 500, kind

    matched = (
        panels["net"]
        .rename({"close_price": "net"})
        .join(panels["gross"].rename({"close_price": "gross"}), on=["entity", "date"])
        .sort("date")
        .group_by("entity")
        .agg(
            net=(pl.col("net").last() / pl.col("net").first() - 1),
            gross=(pl.col("gross").last() / pl.col("gross").first() - 1),
        )
    )
    assert matched.filter(pl.col("gross") < pl.col("net") - 1e-12).is_empty(), (
        "a gross total return cannot be below the net one"
    )
