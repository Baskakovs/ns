"""The STOXX 600's tone counts, keyed by RIC: ``date, ric, pos, neu, neg``.

    from novetly_score.panel import counts

    frame = counts("2025-01", "2025-06")

Everything here is convenience over one call in the data layer::

    data(Index.STOXX600.frame).counts(start_date, end_date, key="ric")

What this module adds is the two things that call cannot know: which universe (the STOXX 600,
which is a YUKKA call and a token) and what to call the columns. The counting, the span and the
PIT RIC keying are all the layer's, so there is one definition of each — see
:meth:`novetly_score.data.Universe.counts` and :func:`novetly_score.data.as_of`.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from yukka.data import Index

from novetly_score.data import data

#: Columns of the panel, in order.
COLUMNS = ("date", "ric", "pos", "neu", "neg")


def counts(
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    *,
    universe: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Positive, neutral and negative article counts per RIC and day.

    A count counts **articles**, not snippets: an article calling one company positive three
    times is positive once, and an article whose snippets carry no sentiment has no tone and is
    in none of the three columns. The day is the UTC day of publication. Those rules belong to
    :class:`novetly_score.data.Counts`, which does the counting.

    Args:
        start_date: Earliest month or day: ``"2025-01"``, ``"2025-01-15"`` or a
            :class:`datetime.date`. ``None`` for the store's earliest, which is 2024-07.
        end_date: Latest, included. ``None`` for the store's latest, as ``None`` for
            *start_date* is its earliest, so either end left open runs to the edge of the store.
        universe: The frame to read and to take RICs from. Defaults to the STOXX 600, which is a
            network call to YUKKA and needs a token.

    Returns:
        ``date``, ``ric``, ``pos``, ``neu``, ``neg``, sorted by date then RIC. A company the store
        has no snippets for, or the universe has no RIC for, is absent rather than present and
        zero — no articles and no coverage are not the same thing.

    Raises:
        NoSnippetsError: If the span selects no file for any company in the universe.
    """
    frame = Index.STOXX600.frame if universe is None else universe
    panel = data(frame).counts(start_date, end_date, key="ric")
    return panel.select(date="day", ric="ric", pos="positive", neu="neutral", neg="negative").sort("date", "ric")
