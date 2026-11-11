"""The STOXX 600's tone counts, and the ratio they reduce to.

    from novetly_score.panel import counts, sentiment

    frame = counts("2025-01", "2025-06")
    score = sentiment(frame)

:func:`counts` is convenience over one call in the data layer::

    data(Index.STOXX600.frame).counts(start_date, end_date, key="ric")

and the only thing it adds is the universe, which that call cannot know: the STOXX 600 is a YUKKA
call and a token. The counting, the span and the PIT RIC keying are all the layer's, so there is
one definition of each — see :meth:`novetly_score.data.Universe.counts` and
:func:`novetly_score.data.as_of`.

**One naming convention, and it is the store's.** A panel is keyed ``(ric, day)`` and its counts
are ``positive``, ``neutral``, ``negative`` — the names the aggregation over the parquet produces.
Nothing here renames them: a 472k-row panel is not worth a relabelling pass, and a second set of
names for the same numbers is worth even less. Where a feed disagrees it is corrected at the
source rather than here — the price snapshot calls the day ``date``, and
:func:`novetly_score.data.prices.keyed` renames it to ``day`` before returning, so both panels
reach a caller keyed alike.

:func:`sentiment` is the one expression this module owns rather than delegates. It is kept out of
the data layer on purpose: the layer counts, and a count is a fact about the store, while the
ratio is the experiment's choice of how to read those counts.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from yukka.data import Index

from novetly_score.data import data

#: Columns of the counts panel, in order — the data layer's own, unrenamed.
COLUMNS = ("ric", "day", "positive", "neutral", "negative", "uncharacterised")

#: Columns of the sentiment frame, in order.
SENTIMENT_COLUMNS = ("ric", "day", "sentiment")


def counts(
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    *,
    universe: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Positive, neutral and negative article counts per RIC and day.

    A count counts **articles**, not snippets: an article calling one company positive three
    times is positive once, and an article whose snippets carry no sentiment has no tone and is
    counted as ``uncharacterised`` rather than folded into ``neutral``. The day is the UTC day of
    publication. Those rules belong to :class:`novetly_score.data.Counts`, which does the
    counting.

    Args:
        start_date: Earliest month or day: ``"2025-01"``, ``"2025-01-15"`` or a
            :class:`datetime.date`. ``None`` for the store's earliest, which is 2024-07.
        end_date: Latest, included. ``None`` for the store's latest, as ``None`` for
            *start_date* is its earliest, so either end left open runs to the edge of the store.
        universe: The frame to read and to take RICs from. Defaults to the STOXX 600, which is a
            network call to YUKKA and needs a token.

    Returns:
        :data:`COLUMNS`, exactly as the data layer returns them. A company the store has no
        snippets for, or the universe has no RIC for, is absent rather than present and zero —
        no articles and no coverage are not the same thing.

    Raises:
        NoSnippetsError: If the span selects no file for any company in the universe.
    """
    frame = Index.STOXX600.frame if universe is None else universe
    return data(frame).counts(start_date, end_date, key="ric")


def sentiment(frame: pl.DataFrame) -> pl.DataFrame:
    """The Laplace-smoothed tone ratio `S = (P + 1) / (P + N + 2)` per RIC and day.

        >>> score = sentiment(counts("2025-01", "2025-06"))     # doctest: +SKIP

    `P` and `N` are the ``positive`` and ``negative`` **article** counts. The ``+1`` and ``+2``
    are a Beta(1, 1) prior: a day with one positive article and nothing else scores 2/3 rather
    than 1, so a company with thin coverage is pulled towards 0.5 instead of to an extreme. The
    denominator is therefore never zero and the result is always in (0, 1).

    **``neutral`` does not appear, and that is the definition, not an omission.** The ratio is
    over the two signed tones only, so a day of ten neutral articles and a day of one neutral
    article both score 0.5 — the same as a day with one positive and one negative. Neutral volume
    is carried by :func:`counts`, where a caller who wants to weight by it can read it.

    Args:
        frame: A counts frame, as :func:`counts` returns. Needs ``ric``, ``day``, ``positive``
            and ``negative``; any other column, ``neutral`` included, is ignored.

    Returns:
        :data:`SENTIMENT_COLUMNS`, one row per row of *frame* and in its order, and so carrying
        its coverage exactly: a company and day absent from the counts is absent here rather than
        scored 0.5. A row whose counts are null scores null.

    Raises:
        ValueError: If *frame* is missing any of ``ric``, ``day``, ``positive`` or ``negative``.
    """
    required = ("ric", "day", "positive", "negative")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        msg = f"counts frame is missing {', '.join(missing)}. Needs: {', '.join(required)}"
        raise ValueError(msg)
    ratio = (pl.col("positive") + 1) / (pl.col("positive") + pl.col("negative") + 2)
    return frame.select("ric", "day", sentiment=ratio.cast(pl.Float64))
