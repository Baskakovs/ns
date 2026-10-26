"""Validation queries over the derived tables and the scores — what ``sql/99_checks.sql`` used to be.

Views carry no PK, FK or CHECK, so the invariants a table load would have enforced live here as
functions, each returning **how many rows violate it**. Zero is clean. They are rules about the
data, so they sit with the code that builds it rather than in the query layer.

Two are informational rather than errors — :func:`feed_repeats` and :func:`company_coverage` —
and say so; the rest should always be zero.

Several scan the 217M-row feed, so they are run on demand (:func:`run`), not on connect.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import polars as pl

from honey.novelty_score_exp.engine import derive
from honey.novelty_score_exp.engine.io import CALIBRATED, RAW
from honey.novelty_score_exp.engine.params import _DATA, DERIVED


def _count(frame: pl.LazyFrame) -> int:
    """Rows in a lazy frame, streamed."""
    return int(frame.select(pl.len()).collect(engine="streaming").item())


def _mentions(out: Path) -> pl.LazyFrame:
    """The derived `mentions` table."""
    return pl.scan_parquet(str(out / "mentions" / "*.parquet"))


def sentiment_domain(out: str | Path = DERIVED) -> int:
    """Mentions whose sentiment is outside {-1, 0, +1}. NULL is legitimate and not counted.

    Args:
        out: Where derived tables live.

    Returns:
        Violating rows.
    """
    return _count(
        _mentions(Path(out)).filter(pl.col("sentiment").is_not_null() & ~pl.col("sentiment").is_in([-1, 0, 1]))
    )


def mention_grain(out: str | Path = DERIVED) -> int:
    """`(article, company)` pairs appearing more than once — `mentions` must be one row per pair.

    Checked per bucket file, which suffices because :func:`derive.build_mentions` refuses to
    build when a company spans buckets.

    Args:
        out: Where derived tables live.

    Returns:
        Duplicated pairs.
    """
    total = 0
    for path in derive.files(out).get("mentions", []):
        frame = pl.scan_parquet(path)
        total += _count(frame) - _count(frame.select("article_id", "company").unique())
    return total


def feed_repeats(data: str | Path = _DATA) -> int:
    """Informational: feed pairs repeated with *disagreeing* sentiment.

    Measured over 20 shards, 10.2% of pairs repeat and only 536 of 1,167,510 disagree.
    `mentions` resolves them by tone, so a rising count is a data-quality signal.

    Args:
        data: Repo ``data/``.

    Returns:
        Disagreeing pairs.
    """
    data = Path(data)
    total = 0
    for bucket in derive._buckets(data):
        total += _count(
            pl.scan_parquet(str(data / f"sentiment_dump/dump-{bucket}-*.parquet"))
            .group_by("news_id", "yukka_id")
            .agg(pl.len().alias("rows"), pl.col("sentiment").n_unique().alias("distinct"))
            .filter((pl.col("rows") > 1) & (pl.col("distinct") > 1))
        )
    return total


def article_consistency(data: str | Path = _DATA) -> int:
    """Articles the feed contradicts itself about — two publish times, sources or languages.

    `articles` keeps an arbitrary row per article; this says whether there was a choice to make.

    Args:
        data: Repo ``data/``.

    Returns:
        Contradicted articles.
    """
    part = pl.col("news_id").hash(seed=0) % derive.ARTICLE_PARTITIONS
    total = 0
    for i in range(derive.ARTICLE_PARTITIONS):
        total += _count(
            derive._feed(Path(data))
            .filter(part == i)
            .group_by("news_id")
            .agg(pl.col("publish_time", "source_uri", "news_lang").n_unique())
            .filter((pl.col("publish_time") > 1) | (pl.col("source_uri") > 1) | (pl.col("news_lang") > 1))
        )
    return total


def article_titles_grain(data: str | Path = _DATA) -> int:
    """Article ids with more than one headline; the dump dedup would have picked arbitrarily.

    Args:
        data: Repo ``data/``.

    Returns:
        Duplicated ids.
    """
    frame = pl.scan_parquet(str(Path(data) / "article_titles.parquet"))
    return _count(frame) - _count(frame.select("article_id").unique())


def titles_grain(data: str | Path = _DATA) -> int:
    """Headlines embedded more than once; a duplicate would fan out the join to `article_titles`.

    Args:
        data: Repo ``data/``.

    Returns:
        Duplicated headlines.
    """
    frame = pl.scan_parquet(str(Path(data) / "es_title_embeddings" / "titles_emb_*.parquet")).select("title")
    return _count(frame) - _count(frame.unique())


def company_coverage(data: str | Path = _DATA, out: str | Path = DERIVED) -> int:
    """Informational: companies with mentions but no universe row. A large count means a stale universe.

    Args:
        data: Repo ``data/``.
        out: Where derived tables live.

    Returns:
        Companies outside the universe.
    """
    universe = pl.scan_parquet(str(Path(data) / "stoxx600_ever_members.parquet")).select(
        pl.col("yukka_id").alias("company")
    )
    return _count(_mentions(Path(out)).select("company").unique().join(universe, on="company", how="anti"))


def novelty_range(path: str | Path) -> int:
    """Scores outside [0, 1] on either scale — the scorer clips, so a hit means something wrote past it.

    Args:
        path: A scores parquet.

    Returns:
        Violating rows.
    """
    frame = pl.scan_parquet(str(path))
    cols = [c for c in frame.collect_schema().names() if c.startswith((RAW + "_k", CALIBRATED + "_k"))]
    outside = pl.any_horizontal([(pl.col(c) < 0.0) | (pl.col(c) > 1.0) for c in cols])
    return _count(frame.filter(outside))


#: Every check that needs only the data root, by name, as :func:`run` reports them.
CHECKS: Final[dict[str, Callable[[], int]]] = {
    "sentiment_domain": sentiment_domain,
    "mention_grain": mention_grain,
    "feed_repeats": feed_repeats,
    "article_consistency": article_consistency,
    "article_titles_grain": article_titles_grain,
    "titles_grain": titles_grain,
    "company_coverage": company_coverage,
}


def run(scores: str | Path | None = None) -> dict[str, int]:
    """Every check over the default store, plus :func:`novelty_range` when given a scores file.

    Args:
        scores: A scores parquet to range-check.

    Returns:
        ``{check: violations}``.
    """
    out = {name: fn() for name, fn in CHECKS.items()}
    if scores is not None:
        out["novelty_range"] = novelty_range(scores)
    return out
