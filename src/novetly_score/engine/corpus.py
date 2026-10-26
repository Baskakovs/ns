"""Parsing `es_metadata` into the `(article, company)` rows everything downstream is built on.

`es_metadata` is the only asset that joins to itself: one record carries the article id, the
headline, the company, the sentiment spans, the publish time, the language and Yukka's own
reprint label. Nothing here needs a second source.

It is also 38 GB of NDJSON across 234 per-company files, so the schema is declared rather than
inferred and is kept as narrow as the work allows — see :data:`DUMP_SCHEMA`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import polars as pl

from honey.novelty_score_exp.core.representation import Article, Mention

#: Only the fields the pipeline needs. Declaring the schema makes the reader a fixed cost and
#: stops a file that gains a field from silently changing what downstream sees.
#:
#: Narrow on purpose: ``matches_spans`` is reduced to ``sentiment`` alone, which keeps the other
#: five span fields — most of the dump's bytes — from ever being materialised. Widening this is
#: the single easiest way to make a 38 GB parse much slower, so add a field only when something
#: uses it.
#:
#: ``id`` is the addition that makes a score joinable. The previous implementation omitted it to
#: keep the parse narrow and, as a result, could not attach a novelty score to the mention that
#: produced it. It costs one string column.
DUMP_SCHEMA: Final[dict[str, pl.DataType]] = {
    "id": pl.Utf8,  # article id, 40-hex; the feed's news_id
    "entity": pl.Utf8,  # company, e.g. 'company:allianz'
    "publish_time": pl.Utf8,
    "title": pl.Utf8,
    "lang": pl.Utf8,  # for per-language rank calibration (§10.4)
    "duplicate_group_id": pl.Utf8,  # Yukka's own reprint label — evaluation only, never a filter
    "matches_spans": pl.List(pl.List(pl.Struct({"sentiment": pl.Int64}))),
}

#: Filenames the per-company dump uses.
COMPANY_GLOB: Final = "company_*.jsonl"


def tone_expr() -> pl.Expr:
    """The vectorised form of :func:`honey.novelty_score_exp.core.counts.tone`.

    Aggregates within a group, so it belongs in an ``agg`` over exploded spans.

    **Null where there are no spans**, matching the scalar form. That case is the whole reason
    this is a ``when/then`` rather than a bare ``sign()``: summing nothing yields 0, and 0 means
    "mentions that cancel", not "the company was named and nothing was said about it". Returning
    0 there is the NULL-is-not-neutral distortion — 8.6% of the feed overall and 54% for the
    busiest names.

    Owning that rule here rather than leaving it to a caller's filter is what lets this be used
    on its own, and what lets the equivalence test compare like with like.

    Returns:
        Expression yielding -1, 0, +1 or null, aliased ``sentiment``.
    """
    spans = pl.col("matches_spans")
    return (
        pl.when(spans.is_not_null().sum() == 0)
        .then(pl.lit(None))
        .otherwise(spans.struct.field("sentiment").fill_null(0).sum().sign())
        .cast(pl.Int64)
        .alias("sentiment")
    )


def discover_dump(dump_dir: str | Path) -> dict[str, str]:
    """Map company -> file path by reading each file's first record.

    The filename is a slug of the company name and is not the id, so the mapping is read from
    the data rather than derived from the name. Empty files are skipped.

    Args:
        dump_dir: Directory holding the per-company ``.jsonl`` files.

    Returns:
        ``{entity: path}``.
    """
    out: dict[str, str] = {}
    for path in sorted(Path(dump_dir).glob(COMPANY_GLOB)):
        if path.stat().st_size == 0:
            continue
        with path.open() as handle:
            first = handle.readline()
        if not first.strip():
            continue
        entity = json.loads(first).get("entity")
        if entity:
            out[entity] = str(path)
    return out


def load_company(path: str | Path) -> pl.DataFrame:
    """One company's file as time-sorted on-topic mentions.

    Two rules, both from the previous implementation and both load-bearing:

    **On-topic means at least one span.** An article that mentions the company without a
    sentiment-bearing span is not a mention of it and is dropped.

    **Timestamps come back tz-aware, in UTC.** Polars' ``.dt.epoch()`` reads a naive datetime
    as UTC and would be correct either way, but Python's ``datetime.timestamp()`` reads one as
    *local* time: measured on this dump, a naive parse is 7,200 s off in CEST, and the error
    changes across DST. Attaching the zone at the parse closes that for every later consumer
    rather than relying on each of them to use the right accessor.

    Args:
        path: One ``company_*.jsonl`` file.

    Returns:
        ``(article_id, entity, ts, title, lang, duplicate_group_id, sentiment)``, sorted by ts.
    """
    return (
        pl.scan_ndjson(str(path), schema=DUMP_SCHEMA)
        .with_row_index("row")
        .explode("matches_spans")  # list[list[struct]] -> list[struct]
        .explode("matches_spans")  # -> struct, null for an article with no spans
        .group_by("row")
        .agg(
            pl.first("id"),
            pl.first("entity"),
            pl.first("publish_time"),
            pl.first("title"),
            pl.first("lang"),
            pl.first("duplicate_group_id"),
            tone_expr(),
        )
        # on-topic only. tone_expr is null exactly when the company was named with no
        # sentiment-bearing span, so this is the same rule rather than a second one.
        .filter(pl.col("sentiment").is_not_null())
        .select(
            pl.col("id").alias("article_id"),
            "entity",
            pl.col("publish_time").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC").alias("ts"),
            "title",
            "lang",
            "duplicate_group_id",
            "sentiment",
        )
        .sort("ts")
        .collect()
    )


def articles(frame: pl.DataFrame) -> list[Article]:
    """One :class:`Article` per row of a :func:`load_company` frame.

    **The bridge between the dump and `core/`.** Nothing else in the package constructs an
    `Article`, so without this `core/` is a specification with no path from real data and every
    test against it runs on synthetic corpora — which checks that the rules are self-consistent
    and nothing about whether the two layers see the same rows.

    The frame is one company's mentions, so each row becomes an article mentioning exactly that
    one company. An article naming three companies appears once per company, in three different
    frames, with a different tone in each — which is §2.1's point that novelty is a property of
    the pair. Reassembling them into one `Article` with three entries would be wrong here: the
    reference set is per company, so nothing downstream would read the other two.

    **The tone is already reduced.** ``sentiment`` on the frame is the sign of the span sum,
    computed by :func:`tone_expr`, so a single :class:`Mention` carrying it is the faithful
    record at this grain — not a loss. The individual span tonalities are not in
    :data:`DUMP_SCHEMA` and never reach here.

    Args:
        frame: A :func:`load_company` result — ``(article_id, entity, ts, title, lang,
            duplicate_group_id, sentiment)``.

    Returns:
        The articles, in the frame's order, which :func:`load_company` sorts by ``ts``.
    """
    return [
        Article(
            article_id=row["article_id"],
            published=row["ts"],
            title=row["title"],
            mentions={row["entity"]: [Mention(row["sentiment"])]},
        )
        for row in frame.iter_rows(named=True)
    ]
