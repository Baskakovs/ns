"""Turn the count panels into the 149 wide signal panels the IC sweep ranks.

The statistic, and the one variant per event:

    baseline[c,t]  = (pos - neg) / tot                                   tot = pos + neu + neg
    variant_a[c,t] = ((pos - pos_a) - (neg - neg_a)) / (tot - tot_a)

Three properties this module is responsible for:

* **A company untouched by event *a* keeps its baseline value.** The variant is a *left* join onto
  the baseline with the event counts filled to zero, not an inner join — an inner join would
  silently shrink the cross-section to only the companies that had that event that day, which for
  a rare event means a handful of names and an IC computed on noise.
* **A fully-excluded cell is null, not infinite.** ``tot == tot_a`` happens whenever every one of a
  company's matches that day came from tagged articles. Dividing anyway would yield ``inf``, which
  ``Signal.ic`` would rank as the largest value in the cross-section.
* **No smoothing.** This is an end-of-day snapshot, per the experiment's specification.

One further wrinkle the specification does not settle: 8% of sentiment matches are published on
non-trading days (weekends and market holidays). Aligned literally, those are simply discarded —
Saturday's news never reaches any trading day's snapshot. :func:`to_trading_day` offers the other
reading, rolling each non-trading day's *counts* forward into the next trading day, so a Monday
snapshot is "everything published since Friday's close". The roll has to happen on counts, before
the ratio: averaging two days' ratios is not the ratio of their pooled counts. It also preserves
both the ``tot == pos + neu + neg`` identity and the ``pos_a <= pos`` subset property, since it only
coarsens the grouping. The headline sweep uses the literal reading; the roll is a reported variant.

Wide panels are keyed by RIC to match the price panel, using the ``yukka_id -> ric`` bridge that
``constituent_prices`` hands back for free (see :func:`honey.event_exclusion_exp.prices.id_bridge`).
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

#: Column name the wide panels carry the statistic under before pivoting.
SIGNAL = "signal"


#: The cross-sectional statistics this experiment can rank events under, as functions of the three
#: signed count expressions.
#:
#: ``net`` is the experiment's original ``(pos - neg) / tot``. ``laplace`` is what the production
#: book actually trades — :func:`honey.data.counts._laplace_positive_ratio`, which ignores neutral
#: articles entirely and is bounded in (0, 1). The two disagree materially about recent years, so an
#: event ranking is only meaningful next to the statistic it was measured on.
RATIOS: dict[str, Callable[[pl.Expr, pl.Expr, pl.Expr], pl.Expr]] = {
    "net": lambda pos, neg, tot: pl.when(tot > 0).then((pos - neg) / tot).otherwise(None),
    "laplace": lambda pos, neg, _tot: (pos + 1) / (pos + neg + 2),
}

#: Held in a dict rather than a bare module global so `set_ratio` needs no `global` statement.
_ACTIVE = {"name": "net"}


def set_ratio(name: str) -> None:
    """Choose the statistic every later `baseline_statistic`/`variant_statistic` call will build.

    Process-wide and deliberately so: baseline and variants must never be built under different
    definitions, since the whole test is their paired difference.

    Args:
        name: A key of :data:`RATIOS`.

    Raises:
        KeyError: If `name` is not a known statistic.
    """
    if name not in RATIOS:
        msg = f"unknown ratio {name!r}; expected one of {sorted(RATIOS)}"
        raise KeyError(msg)
    _ACTIVE["name"] = name


def active_ratio() -> str:
    """The statistic currently selected, for labelling output."""
    return _ACTIVE["name"]


def _ratio(pos: pl.Expr, neg: pl.Expr, tot: pl.Expr) -> pl.Expr:
    """The active statistic, null where it is undefined rather than infinite.

    Counts are cast to ``Int64`` before subtracting: on unsigned counts ``pos - neg`` underflows to
    a huge positive number for net-negative company-days. ``panel._count_aggs`` already emits
    signed columns, but a panel built elsewhere must not be able to reintroduce the bug here.
    """
    signed_pos, signed_neg = pos.cast(pl.Int64), neg.cast(pl.Int64)
    signed_tot = tot.cast(pl.Int64)
    return RATIOS[_ACTIVE["name"]](signed_pos, signed_neg, signed_tot).alias(SIGNAL)


def baseline_statistic(baseline: pl.DataFrame) -> pl.DataFrame:
    """The unmodified statistic: ``(date, yukka_id, signal)``."""
    return baseline.select(
        "date",
        "yukka_id",
        _ratio(pl.col("pos"), pl.col("neg"), pl.col("tot")),
    )


#: Columns a subtraction frame carries into :func:`variant_statistic`.
_SUBTRACT_COLS = ("date", "yukka_id", "pos_a", "neg_a", "tot_a")


def split_by_event(events: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Partition the event panel into one subtraction frame per event name.

    Done once, because the sweep asks for ~148 variants: filtering the whole panel per variant
    means ~148 full scans of a multi-million-row frame for no reason.

    Returns:
        ``{event_name: (date, yukka_id, pos_a, neg_a, tot_a)}``.
    """
    return {
        name: part.select(_SUBTRACT_COLS) for (name,), part in events.partition_by("event_name", as_dict=True).items()
    }


def variant_statistic(
    baseline: pl.DataFrame,
    events: pl.DataFrame | dict[str, pl.DataFrame],
    event_name: str,
) -> pl.DataFrame:
    """The statistic with ``event_name``'s articles subtracted: ``(date, yukka_id, signal)``.

    Args:
        baseline: ``(date, yukka_id, pos, neu, neg, tot)``.
        events: Either the full event panel ``(date, yukka_id, event_name, pos_a, neu_a, neg_a,
            tot_a)``, or the output of :func:`split_by_event`. Pass the split form when sweeping
            many events, so the panel is scanned once instead of once per event.
        event_name: The event to subtract. An event absent from ``events`` reproduces the baseline.

    Returns:
        One row per baseline company-day — the cross-section is never reduced.
    """
    if isinstance(events, dict):
        subtract = events.get(event_name)
        if subtract is None:
            subtract = pl.DataFrame(schema={c: baseline.schema.get(c, pl.Int64) for c in _SUBTRACT_COLS})
    else:
        subtract = events.filter(pl.col("event_name") == event_name).select(_SUBTRACT_COLS)
    joined = baseline.join(subtract, on=["date", "yukka_id"], how="left").with_columns(
        pl.col("pos_a").fill_null(0),
        pl.col("neg_a").fill_null(0),
        pl.col("tot_a").fill_null(0),
    )
    return joined.select(
        "date",
        "yukka_id",
        _ratio(
            pl.col("pos") - pl.col("pos_a"),
            pl.col("neg") - pl.col("neg_a"),
            pl.col("tot") - pl.col("tot_a"),
        ),
    )


def to_trading_day(counts: pl.DataFrame, spine: pl.DataFrame, group_extra: tuple[str, ...] = ()) -> pl.DataFrame:
    """Roll non-trading-day counts forward into the next trading day and re-aggregate.

    Args:
        counts: A count panel keyed by ``date`` and ``yukka_id`` — either the baseline or the event
            panel. Every column other than the keys and ``group_extra`` is summed.
        spine: Frame with a ``date`` column giving the trading days.
        group_extra: Extra grouping columns, e.g. ``("event_name",)`` for the event panel.

    Returns:
        The same schema, with every ``date`` now on the spine. Counts published after the last
        trading day in ``spine`` have no day to roll into and are dropped.
    """
    trading = spine.select(pl.col("date").alias("trading_date")).unique().sort("trading_date")
    keys = ["date", "yukka_id", *group_extra]
    value_cols = [c for c in counts.columns if c not in keys]
    rolled = (
        counts.sort("date")
        .join_asof(trading, left_on="date", right_on="trading_date", strategy="forward")
        .drop_nulls("trading_date")
        .drop("date")
        .rename({"trading_date": "date"})
    )
    return rolled.group_by(keys).agg(pl.col(c).sum() for c in value_cols).select(*keys, *value_cols).sort(keys)


def to_wide(long: pl.DataFrame, bridge: pl.DataFrame, key: str = "ric") -> pl.DataFrame:
    """Pivot ``(date, yukka_id, signal)`` to the wide ``date x ric`` panel ``Signal`` expects.

    Entities absent from ``bridge`` are dropped rather than carried under their ``yukka_id``: a
    column the price panel has no counterpart for would be silently ignored by ``Signal.ic``
    anyway, and keeping it here would misreport the cross-section width.

    ``key="yukka_id"`` keys the panel by entity id and needs no bridge at all — that is the shape of
    the local-currency robustness panel, whose columns are ``yukka_id`` because those prices come
    from ``price_data_itemized`` rather than ``constituent_prices``. Getting this wrong is quiet
    rather than loud: ``Signal.ic`` intersects column names and would simply find no overlap,
    reporting an IC of NaN rather than raising.

    Args:
        long: ``(date, yukka_id, signal)``.
        bridge: ``(yukka_id, ric)``, one row per entity. Ignored when ``key`` is ``"yukka_id"``.
        key: Bridge column to key the wide panel by.

    Returns:
        Wide frame with ``date`` first, then one column per mapped entity, sorted by date.
    """
    mapped = long if key == "yukka_id" else long.join(bridge.select("yukka_id", key), on="yukka_id", how="inner")
    if mapped.is_empty():
        return long.select("date").unique().sort("date")
    return mapped.pivot(on=key, index="date", values=SIGNAL, aggregate_function="first").sort("date")


def align(wide: pl.DataFrame, spine: pl.DataFrame) -> pl.DataFrame:
    """Reindex a wide signal panel onto a price panel's trading-day spine.

    A left join on the spine keeps every trading day — including those on which a name had no news,
    which become null and are skipped by ``Signal.ic`` rather than dropping the whole date.
    """
    cols = [c for c in wide.columns if c != "date"]
    return spine.select("date").sort("date").join(wide, on="date", how="left").select("date", *cols)


def event_universe(events: pl.DataFrame, min_company_days: int = 0) -> list[str]:
    """The event names to sweep, most frequent first.

    Args:
        events: The event count panel.
        min_company_days: Drop events observed on fewer company-days than this. Left at 0 for the
            headline sweep so the ranking covers every event the data contains; useful when the
            tail of ultra-rare events dominates the extremes purely through estimation noise.

    Returns:
        Event names sorted by descending company-day frequency.
    """
    counts = events.group_by("event_name").len().filter(pl.col("len") >= min_company_days)
    return counts.sort("len", descending=True)["event_name"].to_list()
