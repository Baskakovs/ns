"""The feed-derived tables — `mentions`, `articles`, `sources` — built once, as stamped parquet.

These used to be SQL views, which put rules in the query layer that nothing pinned to `core/`:
the sign-of-sum tone over a pair's feed rows, the per-article dedup, the headline join. They
are built here instead, in polars, so the query layer only projects columns, and so each rule
has one implementation that is tested against `core/`.

It is also much faster to use. The views re-aggregated all 217M feed rows on every query —
21.5s for `count(*)` on `mentions`, and every join paid it again. Built once, the aggregation
costs 14s for `mentions` and ~80s for `articles`, and each query after that reads finished rows.

## How the builds stay small

**`mentions` is built per feed bucket.** The dump is sharded ``dump-b{NNN}-{year}.parquet`` and
a company never spans two buckets (measured: 945 companies, none split), so every
`(article, company)` pair is inside one bucket and each bucket aggregates on its own — ~4.5M
rows at a time instead of one 200M-group hash table. :func:`build_mentions` checks that
precondition before relying on it, since a pair split across buckets would silently become two
rows.

**`articles` is built per hash partition of `article_id`.** An article *does* span buckets (it
is repeated once per company it names), so the dedup is global — and done globally in polars it
ran past ten minutes. Partitioning on a hash of the key makes each partition's dedup
independent: 16 passes at ~5s each.

## Staleness

Every file is stamped with this module's fingerprint
(:func:`honey.novelty_score_exp.twins.stamp`) and :func:`check` refuses a file written by other
code. The fingerprint covers the code, not the input: **rebuild after the dump changes**.

    uv run python -m honey.novelty_score_exp.engine.derive
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import polars as pl

from honey.novelty_score_exp.core.representation import Mention
from honey.novelty_score_exp.engine.params import _DATA, DERIVED
from honey.novelty_score_exp.twins import check_stamp, stamp

#: The module every derived parquet is stamped with.
PRODUCER: Final = "engine.derive"

#: The raw feed's shards, and how a shard names its bucket.
FEED_GLOB: Final = "sentiment_dump/dump-b*.parquet"
_BUCKET: Final = re.compile(r"dump-(b\d+)-")

#: Hash partitions for the `articles` dedup. 16 keeps each partition near 8M articles.
ARTICLE_PARTITIONS: Final = 16

#: The three tables, as directories (or a file) under :data:`DERIVED`.
TABLES: Final = ("mentions", "articles", "sources")


def feed_mentions(sentiments: Sequence[int | None]) -> list[Mention]:
    """The bridge from a pair's feed rows to `core`'s representation of it.

    A feed row's sentiment is Yukka's verdict for the whole pair, and NULL means it found **no
    polarity for the pair at all** — which is `core`'s "named but not characterised", an empty
    mention list. So a NULL row becomes no :class:`Mention` rather than ``Mention(None)``: the
    latter is a span carrying no polarity, which `core` counts as 0 and so as *neutral* — the
    NULL-is-not-neutral distortion, 8.6% of the feed.

    Args:
        sentiments: The pair's feed rows' sentiments, repeats included.

    Returns:
        One mention per row that carries a polarity.
    """
    return [Mention(s) for s in sentiments if s is not None]


def feed_tone() -> pl.Expr:
    """`core.counts.tone` over a pair's feed rows, through :func:`feed_mentions`.

    The sign of the sum of the non-null rows, or null where every row is null. Pinned to
    ``core.tone(feed_mentions(rows))`` by ``tests/test_novelty_derive.py``.

    Returns:
        An aggregation yielding -1, 0, +1 or null, aliased ``sentiment``.
    """
    s = pl.col("sentiment")
    return pl.when(s.is_not_null().any()).then(s.sum().sign()).cast(pl.Int8).alias("sentiment")


def _feed(data: Path) -> pl.LazyFrame:
    """The raw feed, every shard."""
    return pl.scan_parquet(str(data / FEED_GLOB))


def _buckets(data: Path) -> list[str]:
    """The feed's bucket names, sorted."""
    return sorted({m.group(1) for p in data.glob(FEED_GLOB) if (m := _BUCKET.search(p.name))})


def _replace(staging: Path, target: Path) -> None:
    """Swap a finished build into place, so a failed build never leaves a half-written table."""
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    staging.rename(target)


def build_mentions(data: str | Path = _DATA, out: str | Path = DERIVED) -> Path:
    """`mentions`: one row per `(article, company)`, the grain the signal is built on.

    Aggregated rather than deduplicated, deliberately. The feed returns 1.10 rows per pair
    (10.2% of pairs repeat), almost all byte-identical re-ingests; left in, they would count an
    article more than once and put the panels 1.10x above §3's `P`, which is a count of
    articles. The tone is :func:`feed_tone` — `core`'s tone, so the 0.05% of repeats that
    disagree are resolved by the definition rather than by whichever row came first.

    ``publish_date`` is carried here because every query downstream aggregates by date.

    Args:
        data: Repo ``data/``.
        out: Where derived tables live.

    Returns:
        The ``mentions/`` directory.

    Raises:
        ValueError: If a company spans two feed buckets, which would split its pairs.
    """
    data, out = Path(data), Path(out)
    spans = (
        pl.scan_parquet(str(data / FEED_GLOB), include_file_paths="file")
        .select("yukka_id", pl.col("file").str.extract(_BUCKET.pattern).alias("bucket"))
        .group_by("yukka_id")
        .agg(pl.col("bucket").n_unique().alias("buckets"))
        .filter(pl.col("buckets") > 1)
        .collect(engine="streaming")
    )
    if spans.height:
        msg = f"{spans.height} companies span several feed buckets, e.g. {spans['yukka_id'][0]}; pairs would split"
        raise ValueError(msg)

    staging = out / ".building-mentions"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    meta = stamp(PRODUCER)
    for bucket in _buckets(data):
        (
            pl.scan_parquet(str(data / f"sentiment_dump/dump-{bucket}-*.parquet"))
            .group_by(pl.col("news_id").alias("article_id"), pl.col("yukka_id").alias("company"))
            .agg(
                pl.col("publish_time").min().dt.date().alias("publish_date"),
                feed_tone(),
                pl.col("implied").any().alias("implied"),
                pl.len().cast(pl.UInt32).alias("feed_rows"),  # >1 means the feed repeated this pair
            )
            .sink_parquet(staging / f"{bucket}.parquet", metadata=meta)
        )
    _replace(staging, out / "mentions")
    return out / "mentions"


def build_articles(data: str | Path = _DATA, out: str | Path = DERIVED) -> Path:
    """`articles`: one row per article, with its headline where a dump has one.

    The feed returns one row per `(article, company)`, so an article repeats once per company
    it names (fan-out 1.579). Collapsing that recovers the article grain; it is not a dedup of
    stories and drops no coverage. Where the feed contradicts itself about an article — two
    publish times, two sources — the row kept is arbitrary, and
    :func:`honey.novelty_score_exp.engine.checks.article_consistency` is what says whether that
    ever happens.

    The headline is left-joined, so ``title`` is null for the 91.9% of articles no dump covers:
    the article exists and its headline is unknown, and ``title IS NULL`` is the coverage gap.

    Args:
        data: Repo ``data/``.
        out: Where derived tables live.

    Returns:
        The ``articles/`` directory.
    """
    data, out = Path(data), Path(out)
    staging = out / ".building-articles"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    meta = stamp(PRODUCER)
    part = pl.col("article_id").hash(seed=0) % ARTICLE_PARTITIONS
    titles = pl.scan_parquet(str(data / "article_titles.parquet")).select(
        "article_id", "title", pl.col("dump").alias("title_source")
    )
    for i in range(ARTICLE_PARTITIONS):
        feed = (
            _feed(data)
            .select(
                pl.col("news_id").alias("article_id"),
                "publish_time",
                pl.col("source_uri").alias("source_id"),
                pl.col("news_lang").alias("lang"),
                "news_type",
            )
            .filter(part == i)
            .unique(subset="article_id", keep="any")
        )
        feed.join(titles.filter(part == i), on="article_id", how="left").sink_parquet(
            staging / f"p{i:02d}.parquet", metadata=meta
        )
    _replace(staging, out / "articles")
    return out / "articles"


def build_sources(data: str | Path = _DATA, out: str | Path = DERIVED) -> Path:
    """`sources`: one row per outlet, the distinct ``source_uri`` of the feed.

    Args:
        data: Repo ``data/``.
        out: Where derived tables live.

    Returns:
        The ``sources.parquet`` path.
    """
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    staging = out / ".building-sources.parquet"
    (
        _feed(data)
        .select(pl.col("source_uri").alias("source_id"))
        .drop_nulls()
        .unique()
        .sort("source_id")
        .sink_parquet(staging, metadata=stamp(PRODUCER))
    )
    _replace(staging, out / "sources.parquet")
    return out / "sources.parquet"


def files(out: str | Path = DERIVED) -> dict[str, list[Path]]:
    """The parquet files of each derived table that exists.

    Args:
        out: Where derived tables live.

    Returns:
        ``{table: [files]}``, omitting tables not yet built.
    """
    out = Path(out)
    found = {
        "mentions": sorted((out / "mentions").glob("*.parquet")),
        "articles": sorted((out / "articles").glob("*.parquet")),
        "sources": [out / "sources.parquet"] if (out / "sources.parquet").exists() else [],
    }
    return {table: paths for table, paths in found.items() if paths}


def check(out: str | Path = DERIVED) -> None:
    """Raise unless every derived file present was written by this module as it stands.

    Args:
        out: Where derived tables live.

    Raises:
        StaleArtifactError: Naming the first stale file.
    """
    for paths in files(out).values():
        for path in paths:
            check_stamp(path, PRODUCER)


def build(data: str | Path = _DATA, out: str | Path = DERIVED) -> None:
    """Build all three tables.

    Args:
        data: Repo ``data/``.
        out: Where derived tables live.
    """
    build_mentions(data, out)
    build_articles(data, out)
    build_sources(data, out)


if __name__ == "__main__":
    import time

    t = time.perf_counter()
    for name, fn in (("mentions", build_mentions), ("articles", build_articles), ("sources", build_sources)):
        fn()
        print(f"{name}: {time.perf_counter() - t:.1f}s")
