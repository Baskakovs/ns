"""§3's three routes at scale — the polars form of :mod:`honey.novelty_score_exp.core.aggregation`.

`core/aggregation.py` is the definition, article by article; this is what runs over a company's
whole history at once. They must agree, and ``tests/test_novelty_panels_honour_core.py`` is what
licenses them to coexist.

**This is where §3 is computed, not in SQL.** The query layer only projects parquet columns and
joins on keys; it holds no rule. Everything that exists only on the way to a panel is here, in
polars, which keeps §3 to two implementations with a test between them rather than three with
none.

## Storage keeps everything; comparison uses what is complete

`mentions_scored` carries every mention the feed holds, with a null calibrated novelty wherever
no score exists — before 2024-07 that is all of them, since novelty reaches only as far as `es_metadata`.
:func:`scoreable` is the one place those rows are dropped, and **every route calls it**,
production included. Two things follow:

- **The arms are comparable.** A difference between them can only come from the map
  `A_{c,t} -> S_{c,t}`, not from one arm trading a corpus the others cannot see. §10.5.
- **The universe is fixed by the data rather than by a date filter.** Novelty is null outside
  the scored span, so filtering it collapses all three routes onto identical `(date, entity)`
  keys with nothing hardcoded that could drift.

:func:`scoring_coverage` reports what that costs, so the gap is a measurable backlog item rather
than an invisible one.

## Three things inherited from the view this replaces

**Do not count feed repeats.** The feed returns 1.087 rows per `(article, company)` pair, almost
all byte-identical re-ingests. `mentions` already aggregates to one row per pair, so counting
rows here is counting articles — which is what §3's `P = |{d in A_{c,t} : tone(d) = +1}|` means,
the cardinality of a set of articles.

**Null sentiment is not neutral.** It is 8.6% of the feed overall and 54% for the busiest names.
Bucketing on sign drops it from every bucket rather than counting it as 0.

**An empty bucket must sum to 0.0, not null.** A null negative would null out `(P+1)/(P+N+2)`
for the whole `(date, entity)` downstream. This is why the weighted route fills nulls after
summing.

A note that survives the move: since `nu_tilde <= 1`, the weighted panel's counts are below
production's, so the Laplace prior carries more relative weight and the panel is shrunk toward
0.5 — more so for low-flow names. That is a change in cross-sectional dispersion, not a
rescaling, and it is a property of the ratio rather than of this module.
"""

from __future__ import annotations

from typing import Final

import polars as pl

#: The column every route reads. §3.1 and §3.2 both specify calibrated `nu_tilde`, never the
#: raw score — and the name is what makes that checkable, since the two share a dtype and range.
CALIBRATED: Final = "calibrated_novelty"

#: What a panel emits, in the order production's `build_portfolio` expects.
PANEL_COLUMNS: Final = ("date", "entity", "positive", "neutral", "negative")

#: The routes of §3.1-3.3, and §3's unweighted baseline.
ROUTES: Final = ("production", "filtered", "weighted")


def scoreable(frame: pl.DataFrame) -> pl.DataFrame:
    """The rows that carry a calibrated novelty — the corpus every route shares.

    The vectorised form of :func:`honey.novelty_score_exp.core.aggregation.scoreable`, and the
    only place rows are dropped. Applied by every route including production, so the arms are
    compared on identical samples.

    Args:
        frame: `mentions_scored` rows — at least ``publish_date``, ``company``, ``sentiment``
            and :data:`CALIBRATED`.

    Returns:
        The rows with a non-null calibrated novelty, in the order given.
    """
    return frame.filter(pl.col(CALIBRATED).is_not_null())


def _buckets(weight: pl.Expr) -> list[pl.Expr]:
    """The three sentiment buckets, each summing *weight* over the rows that fall in it.

    One helper rather than three near-identical blocks: §3.1-3.3 differ only in the summand, so
    the bucketing is written once and the routes supply the weight. ``fill_null(0.0)`` is
    load-bearing — summing an empty bucket yields null, and a null negative nulls out the whole
    `(date, entity)` through the Laplace ratio downstream.

    Args:
        weight: What one row contributes.

    Returns:
        The positive, neutral and negative aggregations.
    """
    return [
        weight.filter(pl.col("sentiment") > 0).sum().fill_null(0.0).alias("positive"),
        weight.filter(pl.col("sentiment") == 0).sum().fill_null(0.0).alias("neutral"),
        weight.filter(pl.col("sentiment") < 0).sum().fill_null(0.0).alias("negative"),
    ]


def panel(frame: pl.DataFrame, route: str = "production", theta: float | None = None) -> pl.DataFrame:
    """One route's `(date, entity, positive, neutral, negative)` panel.

    `build_portfolio` consumes this shape unchanged, which is what keeps `S`, the robust signal,
    the threshold book, the per-name cap and the vol target as production's rather than
    reimplemented here. §10.5's requirement is to hold everything but the map from `A_{c,t}` to
    `S_{c,t}` fixed, and this function is that map.

    Args:
        frame: `mentions_scored` rows.
        route: ``production`` (every row counts 1), ``filtered`` (§3.1, rows at or above
            *theta* count 1), or ``weighted`` (§3.2, each row counts its calibrated novelty).
        theta: The cutoff, required by ``filtered`` and meaningless to the others.

    Returns:
        One row per `(date, entity)`, sorted, with float counts — the weight route sums
        calibrated novelty rather than articles.

    Raises:
        ValueError: If *route* is unknown, or if ``filtered`` is asked for without a *theta*.
    """
    if route not in ROUTES:
        msg = f"unknown route {route!r}; §3 gives {', '.join(ROUTES)}"
        raise ValueError(msg)
    if route == "filtered" and theta is None:
        msg = "the filter route (§3.1) needs a theta; it is the whole of what distinguishes it"
        raise ValueError(msg)

    # built per route rather than as a dict of all three: the filtered expression compares
    # against theta, and constructing it for a route that has none compares against null
    if route == "production":
        weight = pl.lit(1.0)
    elif route == "filtered":
        weight = pl.when(pl.col(CALIBRATED) >= theta).then(1.0).otherwise(0.0)  # §3.1's >=
    else:
        weight = pl.col(CALIBRATED)

    return (
        scoreable(frame)
        .group_by(pl.col("publish_date").alias("date"), pl.col("company").alias("entity"))
        .agg(_buckets(weight))
        .select(PANEL_COLUMNS)
        .sort("date", "entity")
    )


def scoring_coverage(frame: pl.DataFrame) -> pl.DataFrame:
    """What :func:`scoreable` costs, by year and company.

    The gap between what the feed holds and what can be scored is a **data** gap, not a design
    choice: novelty needs a headline, and only 8.1% of the feed's articles have one. Dropping
    those rows is right, and leaving the size of the drop unmeasured is not — a panel that
    covers a fraction of its corpus looks exactly like one that covers all of it.

    This is the target list for a backfill: the company-years with the most to gain.

    Args:
        frame: `mentions_scored` rows.

    Returns:
        `(year, company, mentions, scoreable, share)`, sorted by the mentions lost, descending.
    """
    return (
        frame.group_by(pl.col("publish_date").dt.year().alias("year"), pl.col("company"))
        .agg(
            pl.len().alias("mentions"),
            pl.col(CALIBRATED).is_not_null().sum().alias("scoreable"),
        )
        .with_columns((pl.col("scoreable") / pl.col("mentions")).alias("share"))
        .sort(pl.col("mentions") - pl.col("scoreable"), descending=True)
    )
