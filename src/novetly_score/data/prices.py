"""EUR price and total-return levels for a universe, from yl-quant's LSEG snapshot.

Reached through :meth:`novetly_score.data.Universe.prices`, so a panel is asked for the same way
the counts are::

    data(Index.STOXX600.frame).prices("2025-01", "2025-06", key="ric")

**Three adjustments, two endpoints.** ``net`` and ``gross`` are
``/company_data/constituent_total_return``: DSS adjusted closes with regular cash dividends
reinvested at the ex-date close, net of withholding tax (the STOXX Net Return convention, what a
fund such as a UCITS receives) or before it. ``raw`` is ``/company_data/constituent_prices``, the
converted close itself. A total-return *level* is anchored so a name's last level is its last
close, so only ratios of it mean anything — never mix a ``net`` level with a ``raw`` one.

**The feed is keyed by current RIC; this panel is not.** A company that changed line carries one
RIC in the snapshot for its whole history, so the rows come back under the line it trades on
today. They are relabelled to the ``yukka_id`` the store files snippets under, and, if a *key* is
asked for, to the identifier the company held **on each day** — the same
:func:`~novetly_score.data.as_of` the counts use. A price panel and a counts panel therefore key
alike and join without a bridge per caller.

That relabelling has one trap, and :func:`keyed` is where it is handled. Three of the STOXX 600's
companies have **two live lines** in the snapshot — Industrivärden's A and C, Lindt's registered
and participation shares, Schibsted's A and B — so keying both to one company would put two
different prices on one ``(company, day)``. Only the company's current line is kept, that being
how the feed is keyed. The consequence worth knowing: before a line change the snapshot restates
history under the *current* line, so a day earlier than the change carries the old line's name
over the new line's restated value. The snippet history begins in 2024-07 and no company with
snippets changes line inside it, so nothing in this experiment's window is affected.

Nothing is cached. The whole universe over a decade is one minute of requests; a caller who needs
it twice should keep the frame.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import polars as pl

#: What a caller can ask for, and which column of which feed answers it.
ADJUSTMENTS: Final = ("net", "gross", "raw")

#: Earliest date the snapshot covers, per the endpoint's own documentation. The floor for an
#: open-ended span, so ``prices()`` with no start is the feed's whole history rather than an error.
HISTORY_START: Final = date(2015, 10, 13)

#: Columns of the panel, in order. ``date`` is a trading date, where the counts' ``day`` is a
#: publication day — join them on ``(entity, day) == (entity, date)``.
COLUMNS: Final = ("entity", "date", "close_price")


def window(start_date: str | date | None, end_date: str | date | None) -> tuple[date, date]:
    """Resolve a span to the inclusive first day and the **exclusive** last the endpoint wants.

    Takes the three shapes :meth:`Universe.counts` takes, so one span argument works for both
    panels: a month, a day, or a :class:`datetime.date`. A month means all of it — as a start its
    first day, as an end its last.

    Args:
        start_date: Earliest month or day, or ``None`` for :data:`HISTORY_START`.
        end_date: Latest month or day, **included**, or ``None`` for today. The endpoint's
            ``date_to`` is exclusive, so the returned value is one day past it.

    Returns:
        ``(date_from, date_to)`` — inclusive and exclusive respectively.

    Raises:
        ValueError: If a span end is neither ``YYYY-MM`` nor ``YYYY-MM-DD``, or it runs backwards.
    """
    from novetly_score.data.snippets import _month_end, _span_end

    first_month, first_day = _span_end(start_date, "start_date")
    last_month, last_day = _span_end(end_date, "end_date")
    first = first_day or (date.fromisoformat(f"{first_month}-01") if first_month else HISTORY_START)
    last = last_day or (_month_end(last_month) if last_month else date.today())
    if last < first:
        msg = f"prices({start_date!r}, {end_date!r}) runs backwards"
        raise ValueError(msg)
    return first, last + timedelta(days=1)


def fetch(rics: list[str], first: date, last: date, adjustment: str) -> pl.DataFrame:
    """Read levels off the snapshot for *rics*, over ``[first, last)``.

    The request is YUKKA's, not ours: it batches the RICs, carries the token, pins the pull if
    one was set, and drops the provider's placeholder rows — a withheld name returns no row
    rather than a stand-in. ``YUKKA_PRICES_PULL_DATE`` and
    :func:`yukka.data.set_total_return_source` are what fix the snapshot, which an experiment
    comparing two arms should do so a re-pull cannot move the baseline under it.

    Args:
        rics: RICs to ask for. Unknown ones simply return no rows.
        first: Inclusive start.
        last: Exclusive end.
        adjustment: One of :data:`ADJUSTMENTS`.

    Returns:
        ``date``, ``ric``, ``close_price``, not forward-filled.

    Raises:
        ValueError: If *adjustment* is not one of :data:`ADJUSTMENTS`.
    """
    if adjustment not in ADJUSTMENTS:
        msg = f"no such adjustment: {adjustment!r}. Known: {', '.join(ADJUSTMENTS)}"
        raise ValueError(msg)
    from yukka.data import prices as yukka_prices
    from yukka.data import total_return as yukka_total_return

    if adjustment == "raw":
        levels = yukka_prices.fetch_quant_api_prices(yukka_prices.price_source(), rics, first, last)
    else:
        levels = yukka_total_return.fetch_quant_api_total_return(
            yukka_total_return.total_return_source(), rics, first, last, kind=adjustment
        )
    return levels.rename({"entity": "ric"})


def keyed(levels: pl.DataFrame, universe: pl.DataFrame, *, key: str | None = None) -> pl.DataFrame:
    """Relabel feed rows from the current RIC to the key the panel is addressed by.

    Args:
        levels: ``date``, ``ric``, ``close_price``, as :func:`fetch` returns.
        universe: The universe frame, which is what knows a RIC's company.
        key: An identifier column of *universe* to key by, point in time, or ``None`` for the
            ``yukka_id`` the store files snippets under.

    Returns:
        ``entity`` (or *key*), ``date``, ``close_price``, sorted by both keys, one row per key and
        day. A RIC the universe does not name is absent, as is a line the company no longer
        trades on: it belongs to no company this panel is about, or to a company already answered
        for by its current line.

    Raises:
        ValueError: If a RIC in *universe* names two companies, which would double its rows.
    """
    import polars as pl

    from novetly_score.data.snippets import as_of, identifier_history

    history = identifier_history(universe, "ric")
    shared = history.select("yukka_id", "ric").unique().group_by("ric").len().filter(pl.col("len") > 1)
    if shared.height:
        named = ", ".join(shared.get_column("ric").to_list()[:3])
        msg = f"{shared.height} RICs name two companies each ({named}), so their levels cannot be keyed"
        raise ValueError(msg)
    # the snapshot is keyed by the line a company trades on today, so that is the one line of a
    # two-line company whose levels belong to it; history is sorted by spell, so last() is current
    current = history.group_by("yukka_id", maintain_order=True).agg(pl.col("ric").last())
    panel = (
        levels.join(current, on="ric", how="inner")
        .select("date", "close_price", entity="yukka_id")
        .select("entity", "date", "close_price")
    )
    if key is None:
        return panel.sort("entity", "date")
    return as_of(panel, universe, key, on="date")
