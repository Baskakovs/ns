"""Collapse the two article-level dumps into the daily count panels the experiment ranks on.

Two frames come out of here:

``baseline_counts``  ``(date, yukka_id, pos, neu, neg, tot)``
``event_counts``     ``(date, yukka_id, event_name, pos_a, neu_a, neg_a, tot_a)``

The join is the whole experiment, so its grain matters more than anything else in this module.

* The events dump is **multi-row per article-entity** — one row per
  ``(news, entity, event, role, factuality, temporality)``. Left as-is it would fan the join out
  and a single article could contribute its sentiment several times over, making ``pos_a > pos``.
  :func:`event_tags` therefore reduces it to **distinct** ``(news_id, yukka_id, event_name)``
  first. That is what "article-level tag" means: an article-entity either is or is not about event
  *a*, however many event mentions of *a* it contains.
* The sentiment dump is also multi-row per article-entity (~9% of pairs repeat — several
  sentiment-bearing sentences in one article). Those rows are **kept**, so ``pos``/``neg``/``tot``
  stay in the same *match* units as production ``counts()``. Each of an article's matches inherits
  that article's event tags.

Together those two choices make ``pos_a <= pos``, ``neg_a <= neg``, ``tot_a <= tot`` hold cell by
cell, so ``(pos - pos_a)`` can never go negative. `tests/test_event_exclusion.py` asserts it.

Work is done **per entity group**, not per shard index. The events dump is grouped by its own batch
partition, and the sentiment shards actually containing that group's entities are looked up from the
two dumps' ``_plan.json`` files. That keeps 48 independent joins of ~1.4M rows each instead of one
join over 71M — memory bounded by the largest group, not the dataset — while staying correct when
the two dumps were pulled at different batch sizes, which they are: the events dump is month-sharded
(its endpoint rejects wider requests) and the sentiment dump is year-sharded.

Rows with a null ``sentiment`` are dropped, so that ``tot == pos + neu + neg`` is an identity rather
than an approximation — the denominator of the statistic depends on it. These are **not rare**:
8.6% of the dump overall, almost entirely ``non_professional`` sources, and 54% for the busiest
names like ``company:ubs``. A null-sentiment article-entity never also appears with a real
sentiment (measured: 0.0%), so it is genuinely "mentioned, no sentiment assigned" rather than a
duplicate. Such an article carries no sentiment to remove, so it cannot affect any variant, and both
the baseline and the event counts drop it — which is what keeps ``pos_a <= pos`` true.

:func:`build` reports, per entity group, the share of event article-entities absent from the
sentiment dump **entirely** (see :func:`sentiment_pairs` — checked before the null filter, or the
9% with null-only sentiment would be reported as missing data forever). A genuinely missing
sentiment shard is the dangerous failure: the events are still read, find nothing to join to, and
vanish, leaving a panel quietly short of one group's entities with nothing in the output to say so.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import polars as pl

from honey.event_exclusion_exp._io import batches, load_universe

#: Sentiment values as they appear in the dump.
POSITIVE, NEUTRAL, NEGATIVE = 1, 0, -1

#: Shard filename pattern, shared by both dumps: ``dump-b<batch>-<period>.parquet``.
_SHARD_RE = re.compile(r"dump-b(\d{3})-(.+)\.parquet$")

#: Warn when more than this share of a group's event articles are absent from the sentiment dump
#: altogether. The measured rate on complete data is 0.0%, so a percent means a shard is missing.
UNMATCHED_WARN = 0.01


def batch_indices(directory: Path) -> list[int]:
    """Sorted distinct batch indices present in a shard directory."""
    found = {int(m.group(1)) for p in directory.glob("dump-b*.parquet") if (m := _SHARD_RE.search(p.name))}
    return sorted(found)


def batch_shards(directory: Path, batch_ix: int) -> list[Path]:
    """Every shard belonging to one entity batch, in period order."""
    return sorted(directory.glob(f"dump-b{batch_ix:03d}-*.parquet"))


def read_plan(directory: Path) -> dict[str, object]:
    """The sweep's ``_plan.json``, which records the batch size the shards were written at."""
    path = directory / "_plan.json"
    if not path.exists():
        msg = f"{path} is missing — cannot tell which entities each shard covers"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())


def entity_partition(directory: Path, universe_path: str | None = None) -> list[list[str]]:
    """The entity batches a shard directory was written with, reconstructed from its plan.

    The partition is a sorted, deduplicated universe cut into fixed-size batches, so it is fully
    determined by the universe plus ``batch_size``. Recomputing it here is what lets two dumps
    pulled at different batch sizes still be joined by entity.

    The universe is taken from the shard directory's **own plan** when recorded there. Falling back
    to a universe *file* is a hazard worth naming: if that file changed between the two pulls, the
    reconstruction silently assigns entities to the wrong batches and the join mismatches with
    nothing to show for it. Sweeps written by :mod:`honey.event_exclusion_exp._sweep` record their
    universe, so the fallback only applies to older directories.

    Args:
        directory: Shard directory containing ``_plan.json``.
        universe_path: Universe parquet, used only when the plan does not record its universe.

    Returns:
        One list of ``yukka_id``s per batch index.

    Raises:
        ValueError: If the plan records no universe and no ``universe_path`` was given.
    """
    plan = read_plan(directory)
    recorded = plan.get("universe")
    if isinstance(recorded, list) and recorded:
        universe = [str(x) for x in recorded]
    elif universe_path is not None:
        universe = load_universe(universe_path)
    else:
        msg = f"{directory / '_plan.json'} records no universe and no universe_path was given"
        raise ValueError(msg)
    return batches(universe, int(plan["batch_size"]))


def shards_for_entities(directory: Path, partition: list[list[str]], wanted: set[str]) -> list[Path]:
    """Every shard in ``directory`` whose batch contains at least one of ``wanted``."""
    out: list[Path] = []
    for bi, ents in enumerate(partition):
        if wanted.intersection(ents):
            out.extend(batch_shards(directory, bi))
    return sorted(out)


def _scan(paths: list[Path]) -> pl.LazyFrame:
    """Lazily scan a list of shards, or an empty frame when the list is empty."""
    if not paths:
        return pl.LazyFrame()
    return pl.scan_parquet(paths)


def sentiment_matches(
    paths: list[Path], date_from: str, date_to: str, entities: set[str] | None = None
) -> pl.LazyFrame:
    """Sentiment matches for one entity group, dated and windowed, null sentiments dropped.

    Args:
        paths: Sentiment shards covering the group.
        date_from: Inclusive ISO start date.
        date_to: Exclusive ISO end date.
        entities: Restrict to these ``yukka_id``s. Needed when a shard's batch is wider than the
            group being processed, or the same entity would be aggregated twice.

    Returns:
        Lazy frame ``(news_id, yukka_id, date, sentiment)``.
    """
    lf = _scan(paths).select("news_id", "yukka_id", "publish_time", "sentiment")
    if entities is not None:
        lf = lf.filter(pl.col("yukka_id").is_in(list(entities)))
    return (
        lf.drop_nulls("sentiment")
        .with_columns(pl.col("publish_time").dt.date().alias("date"))
        .filter(pl.col("date").is_between(pl.lit(date_from).str.to_date(), pl.lit(date_to).str.to_date(), "left"))
        .drop("publish_time")
    )


def sentiment_pairs(paths: list[Path], date_from: str, date_to: str, entities: set[str] | None = None) -> pl.LazyFrame:
    """Distinct ``(news_id, yukka_id)`` present in the sentiment dump, *including* null sentiments.

    The integrity question "did this article reach the sentiment dump at all" is different from
    "does it carry a usable sentiment". Roughly 9% of article-entities have only a null sentiment —
    concentrated in ``non_professional`` sources, and reaching 54% for the busiest names — so a
    check against the null-filtered matches would report those as missing data forever.
    """
    lf = _scan(paths).select("news_id", "yukka_id", "publish_time")
    if entities is not None:
        lf = lf.filter(pl.col("yukka_id").is_in(list(entities)))
    return (
        lf.with_columns(pl.col("publish_time").dt.date().alias("date"))
        .filter(pl.col("date").is_between(pl.lit(date_from).str.to_date(), pl.lit(date_to).str.to_date(), "left"))
        .select("news_id", "yukka_id")
        .unique()
    )


def event_tags(
    paths: list[Path],
    date_from: str,
    date_to: str,
    *,
    facts_only: bool = False,
    entities: set[str] | None = None,
) -> pl.LazyFrame:
    """Distinct article-level event tags for one entity group: ``(news_id, yukka_id, event_name)``.

    The ``unique`` is load-bearing — see the module docstring. Dropping it turns the join into a
    fan-out and breaks the subset property the statistic relies on.

    Args:
        paths: Event shards covering the group.
        date_from: Inclusive ISO start date.
        date_to: Exclusive ISO end date.
        facts_only: Keep only non-negated events asserted as fact. The robustness variant; the
            headline sweep counts an event however hedged it was.
        entities: Restrict to these ``yukka_id``s.

    Returns:
        Lazy frame ``(news_id, yukka_id, event_name)``, one row per distinct triple.
    """
    lf = _scan(paths).select("news_id", "yukka_id", "publish_time", "event_name", "factuality", "event_is_negated")
    if entities is not None:
        lf = lf.filter(pl.col("yukka_id").is_in(list(entities)))
    lf = (
        lf.drop_nulls("event_name")
        .with_columns(pl.col("publish_time").dt.date().alias("date"))
        .filter(pl.col("date").is_between(pl.lit(date_from).str.to_date(), pl.lit(date_to).str.to_date(), "left"))
    )
    if facts_only:
        lf = lf.filter((pl.col("factuality") == "fact") & (pl.col("event_is_negated") != 1))
    return lf.select("news_id", "yukka_id", "event_name").unique()


def _count_aggs(suffix: str = "") -> list[pl.Expr]:
    """The four count aggregations, optionally suffixed (``""`` for baseline, ``"_a"`` for a variant).

    Cast to ``Int64`` deliberately. A boolean ``.sum()`` gives ``UInt32``, on which the statistic's
    ``pos - neg`` *underflows* for any company-day with more negative than positive news — 0 - 1
    becomes 4294967295, which then ranks as the most positive name in the cross-section. Signed
    counts here mean every downstream consumer of the panel is safe by construction.
    """
    return [
        (pl.col("sentiment") == POSITIVE).sum().cast(pl.Int64).alias(f"pos{suffix}"),
        (pl.col("sentiment") == NEUTRAL).sum().cast(pl.Int64).alias(f"neu{suffix}"),
        (pl.col("sentiment") == NEGATIVE).sum().cast(pl.Int64).alias(f"neg{suffix}"),
        pl.len().cast(pl.Int64).alias(f"tot{suffix}"),
    ]


def baseline_counts(matches: pl.LazyFrame) -> pl.LazyFrame:
    """Daily per-company sentiment counts: ``(date, yukka_id, pos, neu, neg, tot)``.

    ``tot`` is the row count, which equals ``pos + neu + neg`` because null sentiments were already
    dropped upstream.
    """
    return matches.group_by("date", "yukka_id").agg(_count_aggs())


def event_counts(matches: pl.LazyFrame, tags: pl.LazyFrame) -> pl.LazyFrame:
    """Daily per-company-per-event counts: ``(date, yukka_id, event_name, pos_a, neu_a, neg_a, tot_a)``.

    An inner join on ``(news_id, yukka_id)`` attaches each article-entity's distinct event tags to
    every one of that pair's sentiment matches, then groups. A pair with no events drops out, which
    is correct: it contributes to no variant's subtraction.
    """
    return (
        matches.join(tags, on=["news_id", "yukka_id"], how="inner")
        .group_by("date", "yukka_id", "event_name")
        .agg(_count_aggs("_a"))
    )


def build(
    sentiment_dir: Path,
    events_dir: Path,
    date_from: str,
    date_to: str,
    *,
    facts_only: bool = False,
    universe_path: str = "data/stoxx600_ever_members.parquet",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build both count panels, one entity group at a time.

    Groups come from the *events* dump's partition; the sentiment shards covering each group are
    looked up from the sentiment dump's own partition, so the two may have been pulled at different
    batch sizes.

    Args:
        sentiment_dir: Directory of sentiment dump shards.
        events_dir: Directory of events dump shards.
        date_from: Inclusive ISO start date.
        date_to: Exclusive ISO end date.
        facts_only: Restrict event tags to non-negated facts.
        universe_path: Universe parquet both sweeps partitioned, used to reconstruct the partitions.

    Returns:
        ``(baseline, events)`` as eager frames, sorted.
    """
    ev_partition = entity_partition(events_dir, universe_path)
    sent_partition = entity_partition(sentiment_dir, universe_path)
    present = set(batch_indices(events_dir))

    base_parts: list[pl.DataFrame] = []
    event_parts: list[pl.DataFrame] = []
    suspect: list[str] = []
    for bi, group in enumerate(ev_partition):
        if bi not in present:
            continue
        wanted = set(group)
        sent_paths = shards_for_entities(sentiment_dir, sent_partition, wanted)
        if not sent_paths:
            suspect.append(f"b{bi:03d} (no sentiment shards)")
            print(f"  b{bi:03d}  SKIPPED — no sentiment shards cover these entities", flush=True)
            continue

        matches = sentiment_matches(sent_paths, date_from, date_to, entities=wanted)
        tags = event_tags(batch_shards(events_dir, bi), date_from, date_to, facts_only=facts_only, entities=wanted)
        base_parts.append(baseline_counts(matches).collect())
        event_parts.append(event_counts(matches, tags).collect())

        # Integrity is "is this article in the sentiment dump at all", so it is checked against the
        # raw pairs, *before* null sentiments are dropped. Checking against `matches` instead would
        # flag the ~9% of article-entities whose only sentiment row is null — which is a normal
        # outcome, not a missing shard, and would bury a real gap under constant false alarms.
        raw_pairs = sentiment_pairs(sent_paths, date_from, date_to, entities=wanted)
        pairs = tags.select("news_id", "yukka_id").unique()
        n_pairs = pairs.select(pl.len()).collect().item()
        n_matched = pairs.join(raw_pairs, on=["news_id", "yukka_id"], how="semi").select(pl.len()).collect().item()
        unmatched = (n_pairs - n_matched) / n_pairs if n_pairs else 0.0
        flag = ""
        if unmatched > UNMATCHED_WARN:
            flag = f"  <-- {unmatched:.1%} of event articles absent from the sentiment dump"
            suspect.append(f"b{bi:03d} ({unmatched:.1%})")
        print(
            f"  b{bi:03d}  {base_parts[-1].height:>7,} company-days  "
            f"{event_parts[-1].height:>8,} company-day-events{flag}",
            flush=True,
        )

    if suspect:
        print(
            f"\nWARNING: {len(suspect)} entity group(s) where over {UNMATCHED_WARN:.0%} of event "
            f"articles are absent from the sentiment dump: {', '.join(suspect)}.\n"
            "         Those events are silently missing from the panel. The usual cause is a missing\n"
            "         sentiment shard for those entities — check the dump covers the whole window.",
            flush=True,
        )
    if not base_parts:
        msg = "no entity group produced any counts — check the shard directories and date window"
        raise RuntimeError(msg)

    baseline = pl.concat(base_parts).sort("date", "yukka_id")
    events = pl.concat(event_parts).sort("date", "yukka_id", "event_name")
    return baseline, events


def main() -> None:
    """CLI: build both panels and write them as parquet."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--sentiment",
        default="data/sentiment_dump",
        help="sentiment shard directory",
    )
    ap.add_argument("--events", default="data/events_dump", help="events shard directory")
    ap.add_argument("--out", default="data/panels", help="output directory")
    ap.add_argument("--from", dest="lo", default="2024-01-01", help="inclusive start date")
    ap.add_argument("--to", dest="hi", default="2026-09-01", help="exclusive end date")
    ap.add_argument("--facts-only", action="store_true", help="restrict event tags to non-negated facts")
    ap.add_argument("--universe", default="data/stoxx600_ever_members.parquet", help="universe both sweeps used")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "_facts" if a.facts_only else ""
    baseline, events = build(
        Path(a.sentiment), Path(a.events), a.lo, a.hi, facts_only=a.facts_only, universe_path=a.universe
    )

    baseline.write_parquet(out / f"baseline_counts{tag}.parquet", compression="zstd")
    events.write_parquet(out / f"event_counts{tag}.parquet", compression="zstd")
    print(
        f"\nbaseline: {baseline.height:,} company-days, {baseline['yukka_id'].n_unique()} entities, "
        f"{baseline['date'].min()} .. {baseline['date'].max()}\n"
        f"events:   {events.height:,} company-day-events, {events['event_name'].n_unique()} distinct events",
        flush=True,
    )


if __name__ == "__main__":
    main()
