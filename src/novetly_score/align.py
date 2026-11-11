"""Shaping a RIC-keyed panel into the matched wide frames a cross-sectional IC needs.

    from novetly_score.align import entry_day, pit_members, tradable, trading_days, wide

    days = trading_days(levels, first, last)
    panel = entry_day(tone, days).join(pit_members(membership, days), on=["ric", "day"])
    mu = wide(panel, "positive", days, assets)

Long frames here are keyed ``(ric, day)`` and carry the store's own column names — ``positive``,
``neutral``, ``negative`` — so nothing is renamed on the way through. The one exception is
:func:`wide`, which must emit a column called ``date`` because that is the name ``basanos``
requires on the frames it is handed; the rename is a single column on an already-pivoted frame,
and it happens at the library boundary rather than in a caller.

Four decisions live here rather than in each notebook, because getting any of them wrong is
silent rather than loud:

* **The row grid is trading days.** An IC horizon is a row offset, so a calendar grid would make
  ``h=5`` mean five calendar days for some rows and five trading days for others.
* **News is re-dated to the next trading day.** A UTC day of articles is complete only at
  midnight, which is after the European close, so pairing day *t*'s articles with day *t*'s close
  reads news the close could not have seen. :func:`entry_day` moves each calendar day's articles
  onto the first trading day strictly after it.
* **Membership is point in time, in the day's own symbols.** :func:`pit_members` reads the RIC a
  company traded under at the review in force, not the one it trades under today.
* **A missing cell stays missing.** :func:`wide` leaves a company-day with no articles null, so
  the IC drops it from that day's cross-section instead of scoring it as a zero that was never
  observed.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: Count columns :func:`entry_day` sums when none are named — the store's, unrenamed.
COUNT_COLUMNS = ("positive", "neutral", "negative")


def trading_days(prices: pl.DataFrame, first: date, last: date) -> pl.Series:
    """The days the price feed quotes in ``[first, last]``, which become the panel's rows.

    Args:
        prices: A long price frame with a ``day`` column, as
            :meth:`novetly_score.data.Universe.prices` returns.
        first: Earliest day to keep, included.
        last: Latest day to keep, included.

    Returns:
        The distinct days, sorted. One row of every wide frame per entry, so that a horizon of
        ``h`` rows is ``h`` trading days.
    """
    return prices.filter(pl.col("day").is_between(first, last)).get_column("day").unique().sort()


def entry_day(counts: pl.DataFrame, days: pl.Series, columns: Sequence[str] = COUNT_COLUMNS) -> pl.DataFrame:
    """Re-date each company-day of news to the first trading day strictly after it.

    Articles published on calendar day *d* are complete by the end of *d*, so the earliest close
    that postdates all of them is the next trading day's. Friday, Saturday and Sunday news
    therefore all land on Monday, summed. Every calendar day is used exactly once and none is
    used before it was published; the last day of the span is dropped, having no trading day
    after it to price into.

    Args:
        counts: A long counts frame: ``ric``, ``day`` and *columns*.
        days: The trading-day grid, from :func:`trading_days`.
        columns: The count columns to sum. Defaults to :data:`COUNT_COLUMNS`.

    Returns:
        ``ric``, ``day`` and *columns*, where ``day`` is now the trading day the news prices into
        rather than the day it was published.
    """
    calendar = pl.DataFrame({"published": pl.date_range(days.min(), days.max(), eager=True)})
    mapping = (
        calendar.with_columns(after=pl.col("published").dt.offset_by("1d"))
        .sort("after")
        .join_asof(
            pl.DataFrame({"day": days}).sort("day"),
            left_on="after",
            right_on="day",
            strategy="forward",
        )
        .select("published", "day")
        .drop_nulls("day")
    )
    return (
        counts.join(mapping, left_on="day", right_on="published", suffix="_entry")
        .group_by("ric", "day_entry")
        .agg(pl.col(column).sum() for column in columns)
        .rename({"day_entry": "day"})
        .sort("ric", "day")
    )


def pit_members(membership: pl.DataFrame, days: pl.Series, *, top: int | None = None) -> pl.DataFrame:
    """The ``(ric, day)`` pairs the index actually held, in each day's own symbols.

    Composition changes at quarterly reviews, so a day takes the membership of the latest review
    on or before it. The RIC is the one that review recorded — ``ric``, not ``ric_current``:
    150 of the STOXX 600's companies trade under a different RIC today from the one they held at
    some earlier review, so the current symbol would be the wrong key for an earlier day.

    Args:
        membership: ``Index.STOXX600.membership_history``, which needs ``effective_date``,
            ``ric``, ``is_member`` and, for *top*, ``member_rank``.
        days: The trading-day grid, from :func:`trading_days`.
        top: Keep only members ranked at or above this, so ``top=200`` is the largest 200 of the
            index as it stood that day. ``None`` keeps the whole index.

    Returns:
        ``ric``, ``day`` — one row per company and trading day it was a member of.
    """
    members = membership.filter(pl.col("is_member"))
    if top is not None:
        members = members.filter(pl.col("member_rank") <= top)
    reviews = members.select("effective_date", "ric").unique()
    return (
        pl.DataFrame({"day": days})
        .sort("day")
        .join_asof(
            reviews.select("effective_date").unique().sort("effective_date"),
            left_on="day",
            right_on="effective_date",
            strategy="backward",
        )
        .drop_nulls("effective_date")
        .join(reviews, on="effective_date")
        .select("ric", "day")
        .sort("ric", "day")
    )


def wide(frame: pl.DataFrame, value: str, days: pl.Series, rics: Iterable[str]) -> pl.DataFrame:
    """``date`` plus one float column per RIC, on exactly *days* and *rics*.

    The shape a cross-sectional IC reads: row is time, column is company. The key column is
    called ``date`` rather than ``day`` only because ``basanos`` requires that name on the
    ``prices`` and ``mu`` frames it is given, and this is the boundary where its contract starts.

    A pair *frame* has no row for is **null**, not zero — that is what keeps an unobserved
    company-day out of the cross-section rather than asserting a value for it.

    Args:
        frame: A long frame with ``ric``, ``day`` and *value*.
        value: The column to spread across the RICs.
        days: The trading-day grid, from :func:`trading_days`.
        rics: The columns to produce, in order. One absent from *frame* is all null.

    Returns:
        ``date`` followed by one ``Float64`` column per entry of *rics*.
    """
    columns = list(rics)
    pivot = frame.select("day", "ric", value).pivot(on="ric", index="day", values=value)
    absent = [ric for ric in columns if ric not in pivot.columns]
    return (
        pl.DataFrame({"day": days})
        .join(pivot, on="day", how="left")
        .with_columns([pl.lit(None, dtype=pl.Float64).alias(ric) for ric in absent])
        .select(["day", *columns])
        .with_columns(pl.col(columns).cast(pl.Float64))
        .rename({"day": "date"})
    )


def tradable(prices: pl.DataFrame, rics: Iterable[str], *, max_null: float = 0.5) -> list[str]:
    """The RICs whose price column is usable as a return series.

    Three rejections, which are also the three a ``basanos`` engine makes on its ``prices``
    frame: a non-positive level, a series that only ever moves one way (constant included, which
    yields no returns at all), and a column too sparse to measure anything against.

    Args:
        prices: A wide price frame, as :func:`wide` returns.
        rics: The columns to consider.
        max_null: Reject a column more than this fraction null. ``basanos`` itself rejects above
            0.9; half a window of quotes is the stricter bar here.

    Returns:
        The acceptable RICs, in the order given. Report the ones dropped rather than letting a
        narrowed universe pass for the whole one.
    """
    keep = []
    for ric in rics:
        column = prices.get_column(ric)
        quoted = column.drop_nulls()
        if quoted.len() < 3 or quoted.min() <= 0 or column.null_count() / prices.height > max_null:
            continue
        steps = quoted.diff().drop_nulls()
        if (steps >= 0).all() or (steps <= 0).all():
            continue
        keep.append(ric)
    return keep
