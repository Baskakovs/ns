"""Sweep Yukka's `events_dump` into resumable per-(batch, month) parquet shards.

    uv run src/honey/event_exclusion_exp/events_sweep.py --help

The article-level events feed, and the pull this experiment exists to join against the sentiment
dump on ``(news_id, yukka_id)``. The sweep machinery — month grid, resume, atomic writes, manifest,
401 abort, date-bisection on gateway failures — lives in :mod:`honey.event_exclusion_exp._sweep`;
this module is only the endpoint's shape.

Two things specific to this dump:

* **The complexity limit.** ``events_dump`` rejects any request where ``entities x days > 1000``
  with a 400 and an explicit message, which is why the grid is monthly and
  ``_sweep.check_complexity`` asserts ``--batch`` before a single request goes out.
* **The grain is multi-row per article.** One row per
  ``(news, entity, event, role, factuality, temporality)``, and ``deduplicate=false`` keeps it that
  way. Collapsing to one article-level tag per event is :mod:`panel`'s job, and doing it there
  rather than here is what keeps ``pos_a <= pos`` true.

Shards must use the **same universe and batch size** as whatever sentiment dump they will be joined
to, or entities land in different groups on the two sides. ``_plan.json`` records the universe
verbatim so :func:`panel.entity_partition` can reconstruct the partition rather than trust a file
that may have changed since.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from honey.event_exclusion_exp import _sweep
from honey.event_exclusion_exp._io import TS_FORMAT

#: Columns kept per shard, in order. `role`, `factuality`, `temporality` and `event_is_negated` are
#: unused by the headline sweep but are what the robustness variants filter on, and they come free.
SHARD_SCHEMA: dict[str, Any] = {
    "news_id": pl.String,
    "yukka_id": pl.String,
    "ISIN": pl.String,
    "publish_time": pl.Datetime("us"),
    "event_name": pl.String,
    "role": pl.String,
    "factuality": pl.String,
    "temporality": pl.String,
    "event_is_negated": pl.Int8,
    "news_lang": pl.String,
    "news_type": pl.String,
    "source_uri": pl.String,
}


def normalise(entries: list[dict[str, Any]]) -> pl.DataFrame:
    """Cast raw API entries to `SHARD_SCHEMA`, including the empty case.

    An empty shard must still carry the full schema, otherwise a later `scan_parquet` over the
    directory fails on a schema mismatch rather than reading zero rows.
    """
    if not entries:
        return pl.DataFrame(schema=SHARD_SCHEMA)
    df = pl.DataFrame(entries, infer_schema_length=None)
    have = set(df.columns)

    def col(name: str, dtype: pl.DataType) -> pl.Expr:
        """The named column cast to `dtype`, or an all-null column when the API omitted it."""
        return pl.col(name).cast(dtype) if name in have else pl.lit(None, dtype).alias(name)

    return df.select(
        pl.col("news_id").cast(pl.String),
        pl.col("yukka_id").cast(pl.String),
        col("ISIN", pl.String),
        pl.col("news_publish_time").cast(pl.String).str.to_datetime(TS_FORMAT).alias("publish_time"),
        col("event_name", pl.String),
        col("role", pl.String),
        col("factuality", pl.String),
        col("temporality", pl.String),
        col("event_is_negated", pl.Int8),
        col("news_lang", pl.String),
        col("news_type", pl.String),
        col("source_uri", pl.String),
    )


SPEC = _sweep.DumpSpec(
    name="events_dump",
    url="https://quant.api.yukkalab.com/company_data/events_dump",
    schema=SHARD_SCHEMA,
    normalise=normalise,
)


def main() -> None:
    """CLI entry point."""
    _sweep.main(SPEC, "data/events_dump", __doc__)


if __name__ == "__main__":
    main()
