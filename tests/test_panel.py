"""Tests for the RIC-keyed panel.

One month of the real store, which the panel is a thin wrapper over, and the one expression it
owns rather than delegates.
"""

from __future__ import annotations

import polars as pl
import pytest

from novetly_score.panel import COLUMNS, SENTIMENT_COLUMNS, counts, sentiment


def _counts(rows: list[tuple]) -> pl.DataFrame:
    """A counts frame as the data layer returns one, from ``(ric, day, pos, neu, neg)`` rows."""
    return pl.DataFrame(rows, schema=["ric", "day", "positive", "neutral", "negative"], orient="row").with_columns(
        pl.col("day").str.to_date()
    )


class TestSentiment:
    """`S = (P + 1) / (P + N + 2)` over a counts frame."""

    def test_columns_and_dtype(self):
        """The frame is ``ric, day, sentiment``, the ratio a float."""
        frame = sentiment(_counts([("BMWG.DE", "2025-01-02", 3, 5, 1)]))

        assert frame.columns == list(SENTIMENT_COLUMNS)
        assert frame.schema["sentiment"] == pl.Float64

    @pytest.mark.parametrize(
        ("positive", "negative", "expected"),
        [
            (0, 0, 0.5),  # no signed article at all: the prior, untouched
            (1, 0, 2 / 3),  # one positive is not certainty
            (0, 1, 1 / 3),  # and one negative is its mirror
            (1, 1, 0.5),  # balanced, so back to the prior
            (3, 1, 4 / 6),
            (99, 0, 100 / 101),  # deep coverage approaches 1 without reaching it
        ],
    )
    def test_the_ratio(self, positive, negative, expected):
        """The smoothing is Beta(1, 1): the score is in (0, 1) and never at an extreme."""
        frame = sentiment(_counts([("BMWG.DE", "2025-01-02", positive, 0, negative)]))

        assert frame.item(0, "sentiment") == pytest.approx(expected)
        assert 0.0 < frame.item(0, "sentiment") < 1.0

    def test_neutral_is_not_in_the_ratio(self):
        """``neutral`` is excluded by definition, so neutral volume cannot move the score."""
        quiet = sentiment(_counts([("BMWG.DE", "2025-01-02", 1, 0, 1)]))
        loud = sentiment(_counts([("BMWG.DE", "2025-01-02", 1, 500, 1)]))

        assert quiet.item(0, "sentiment") == loud.item(0, "sentiment") == 0.5

    def test_coverage_is_carried_not_filled(self):
        """One row out per row in: a missing company-day stays missing rather than scoring 0.5."""
        frame = _counts([("BMWG.DE", "2025-01-02", 1, 0, 0), ("SAPG.DE", "2025-01-03", 0, 0, 2)])

        assert sentiment(frame).height == frame.height

    def test_nulls_propagate(self):
        """A null count scores null — unknown is not 0.5."""
        frame = sentiment(
            pl.DataFrame(
                {
                    "ric": ["BMWG.DE"],
                    "day": ["2025-01-02"],
                    "positive": [None],
                    "neutral": [0],
                    "negative": [1],
                },
                schema={
                    "ric": pl.String,
                    "day": pl.String,
                    "positive": pl.Int64,
                    "neutral": pl.Int64,
                    "negative": pl.Int64,
                },
            )
        )

        assert frame.item(0, "sentiment") is None

    def test_input_order_is_preserved(self):
        """No re-sorting: the layer's order is kept rather than paid for again on 472k rows."""
        frame = _counts(
            [
                ("SAPG.DE", "2025-01-03", 1, 0, 0),
                ("BMWG.DE", "2025-01-02", 1, 0, 0),
                ("SAPG.DE", "2025-01-02", 1, 0, 0),
            ]
        )

        assert sentiment(frame).get_column("ric").to_list() == ["SAPG.DE", "BMWG.DE", "SAPG.DE"]

    def test_extra_columns_are_ignored(self):
        """A frame carrying more than the counts is read for what it needs."""
        frame = _counts([("BMWG.DE", "2025-01-02", 1, 0, 0)]).with_columns(uncharacterised=pl.lit(7))

        assert sentiment(frame).columns == list(SENTIMENT_COLUMNS)

    @pytest.mark.parametrize("dropped", ["ric", "day", "positive", "negative"])
    def test_a_missing_column_is_an_error(self, dropped):
        """Naming what is absent beats a column-not-found from inside polars."""
        frame = _counts([("BMWG.DE", "2025-01-02", 1, 0, 0)]).drop(dropped)

        with pytest.raises(ValueError, match=f"missing {dropped}"):
            sentiment(frame)


@pytest.mark.integration
def test_reads_a_month_of_the_real_store():
    """One month of the real STOXX 600, through the real store: columns, keys and dtypes.

    Skipped unless the S3 variables and a YUKKA token are set, so the suite stays runnable
    without either. The logic under it — the counting and the RIC bridge — is covered against
    the fixture store in ``test_snippets.py``; what this adds is that the three sources line up.
    """
    from novetly_score.data import ConnectionFailedError

    pytest.importorskip("yukka.data")
    try:
        frame = counts("2025-01", "2025-01")
    except ConnectionFailedError as exc:
        if "not set" in str(exc):
            pytest.skip(str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 — a missing token is a skip, not a failure
        pytest.skip(f"universe or store unavailable: {exc}")

    assert frame.columns == list(COLUMNS)
    assert frame.schema["day"] == pl.Date
    assert frame.height == frame.select("ric", "day").n_unique(), "one row per RIC and day"
    assert frame.get_column("ric").n_unique() > 500
    assert frame.select(pl.col("positive", "neutral", "negative").min()).to_numpy().min() >= 0
