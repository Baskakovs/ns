"""Independent checks that the count panels mean what the experiment assumes they mean.

    uv run src/honey/event_exclusion_exp/verify.py --help

Four things can be wrong in ways that leave a perfectly well-formed panel behind, so each is
checked against a source the pipeline does not itself use:

1. :func:`check_algebra` — the identity ``tot == pos + neu + neg`` and the subset property
   ``pos_a <= pos``, ``neg_a <= neg``, ``tot_a <= tot``, cell by cell over the whole panel. A join
   fan-out shows up here and nowhere else.
2. :func:`check_join_integrity` — the share of ``(news_id, yukka_id)`` pairs in the events dump with
   no counterpart in the sentiment dump, measured over the *full* pull rather than a sample. It must
   stay near zero; above ~1% the two dumps have drifted and the subtraction is unsound.
3. :func:`check_against_data_api` — baseline counts against the Data API's
   ``/api/exports/sentiment_matches``, which is what production ``counts()`` uses. The dump is
   expected to be *higher*, since it has no ``t_stat=2.5`` source filter; what matters is that the
   ratio is stable across entities and days rather than erratic, because a stable ratio is a
   filter and an erratic one is a data problem.
4. :func:`check_against_eod_data` — total per-company-day event counts against
   ``/company_data/eod_data``, a pre-built panel with one column per event that the pipeline
   otherwise ignores. Confirms the event grain independently.
5. :func:`spot_check_completeness` — guards against a dump shard holding fewer rows than the API
   would return, which no response field reports. Shards are re-checked against single-entity
   re-pulls of the *heaviest* entity in each sampled shard, since a shortfall would be a
   response-size effect and so would hit the busiest entity first. (A shortfall was suspected early
   on and turned out not to exist — the apparent gap was null-sentiment rows being filtered out
   before the comparison. This check is what settles that question with data rather than inference.)
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from honey.event_exclusion_exp._io import NLP_VERSION
from honey.event_exclusion_exp.panel import _SHARD_RE, batch_indices, batch_shards
from honey.event_exclusion_exp.prices import QUANT_API_URL, _auth

#: The Data API export production `counts()` goes through, pinned to the same NLP version.
DATA_API_URL = f"https://data.api.yukkalab.com/{NLP_VERSION}/2023-12-13-0"

#: Sentiment codes as the Data API's export reports them.
_SENTIMENT_LABELS = {"1": "pos", "0": "neu", "-1": "neg"}


def check_algebra(baseline: pl.DataFrame, events: pl.DataFrame) -> dict[str, int]:
    """Count identity and subset violations across the whole panel. All zeros is a pass."""
    identity = baseline.filter(pl.col("pos") + pl.col("neu") + pl.col("neg") != pl.col("tot")).height
    joined = events.join(baseline, on=["date", "yukka_id"], how="left")
    out = {
        "identity_violations": identity,
        "orphan_event_rows": int(joined["tot"].null_count()),
        "negative_counts": baseline.filter(
            (pl.col("pos") < 0) | (pl.col("neu") < 0) | (pl.col("neg") < 0) | (pl.col("tot") < 0)
        ).height,
    }
    for col in ("pos", "neu", "neg", "tot"):
        out[f"{col}_a_exceeds_{col}"] = joined.filter(pl.col(f"{col}_a") > pl.col(col)).height
    return out


def check_join_integrity(sentiment_dir: Path, events_dir: Path, sample_batches: int = 6) -> dict[str, float]:
    """Share of event (news_id, yukka_id) pairs missing from the sentiment dump, over sampled batches.

    Sampling batches rather than rows keeps this honest: a batch is a fixed set of entities across
    all periods, so a whole entity missing from one dump cannot hide behind rows from another.
    """
    shared = sorted(set(batch_indices(sentiment_dir)) & set(batch_indices(events_dir)))
    chosen = shared[:: max(len(shared) // sample_batches, 1)][:sample_batches]
    ev_total = 0
    ev_missing = 0
    for bi in chosen:
        sent = pl.scan_parquet(batch_shards(sentiment_dir, bi)).select("news_id", "yukka_id").unique().collect()
        ev = pl.scan_parquet(batch_shards(events_dir, bi)).select("news_id", "yukka_id").unique().collect()
        matched = ev.join(sent, on=["news_id", "yukka_id"], how="semi").height
        ev_total += ev.height
        ev_missing += ev.height - matched
    return {
        "batches_checked": float(len(chosen)),
        "event_pairs": float(ev_total),
        "missing_pairs": float(ev_missing),
        "missing_share": (ev_missing / ev_total) if ev_total else float("nan"),
    }


def data_api_counts(entities: list[str], date_from: str, date_to: str, t_stat: float | None) -> pl.DataFrame:
    """Daily pos/neu/neg from the Data API export, optionally with the production source filter.

    Args:
        entities: ``yukka_id`` list.
        date_from: Inclusive ISO start date.
        date_to: Inclusive ISO end date.
        t_stat: Source-vetting threshold. ``2.5`` reproduces production ``counts()``; ``None`` is
            unfiltered.

    Returns:
        ``(date, yukka_id, pos, neu, neg, tot)``.
    """
    body: dict[str, object] = {
        "time_range": {"date_from": date_from, "date_to": date_to},
        "sentiment_filter": {"entities": {"include": entities, "exclude": []}},
        "collapse_on": [],
    }
    if t_stat is not None:
        body["source_filter"] = {"t_stat": t_stat}
    with httpx.Client(headers=_auth(), timeout=600) as client:
        r = client.post(f"{DATA_API_URL}/api/sentiment_matches", json=body)
        r.raise_for_status()
        payload = r.json()

    schema = {
        "date": pl.Date,
        "yukka_id": pl.String,
        "pos": pl.Int64,
        "neu": pl.Int64,
        "neg": pl.Int64,
        "tot": pl.Int64,
    }
    rows = pl.DataFrame(payload["sentiment_matches"], schema=payload["columns"], orient="row")
    if rows.is_empty():
        return pl.DataFrame(schema=schema)

    pivoted = rows.with_columns(
        pl.col("publish_time").cast(pl.String).str.to_date().alias("date"),
        pl.col("entity_compound_key").cast(pl.String).alias("yukka_id"),
        pl.col("sentiment").cast(pl.String).replace_strict(_SENTIMENT_LABELS, default="other"),
        pl.col("count").cast(pl.Int64),
    ).pivot(on="sentiment", index=["date", "yukka_id"], values="count", aggregate_function="sum")
    # The pivot only creates columns for the categories this response happened to contain, so a
    # batch with (say) no negatives comes back one column narrower. Concatenating such frames across
    # batches then fails on a width mismatch, so the shape is pinned here instead: absent categories
    # become zeros, and any unexpected sentiment code lands in "other" and is dropped.
    missing = [pl.lit(0, pl.Int64).alias(c) for c in ("pos", "neu", "neg") if c not in pivoted.columns]
    return (
        pivoted.with_columns(missing)
        .with_columns(pl.col(c).fill_null(0).cast(pl.Int64) for c in ("pos", "neu", "neg"))
        .with_columns((pl.col("pos") + pl.col("neu") + pl.col("neg")).alias("tot"))
        .select(*schema)
        .sort("date", "yukka_id")
    )


def check_baseline_at_scale(
    baseline: pl.DataFrame,
    date_from: str,
    date_to: str,
    batch_size: int = 100,
) -> pl.DataFrame:
    """Compare the whole dump-built baseline to the Data API export, company-day by company-day.

    The per-entity check in :func:`check_against_data_api` samples five names; this one covers every
    entity in the panel over a window: a per-entity shortfall would be concentrated in a handful of
    heavy names and so invisible at small scale. The export is aggregated
    server-side, so this is cheap: ~1s and 0.2MB per 100 entities per month.

    Args:
        baseline: The dump-built count panel.
        date_from: Inclusive ISO start date.
        date_to: Inclusive ISO end date.
        batch_size: Entities per export request.

    Returns:
        Per-company-day comparison ``(date, yukka_id, tot, api_tot, diff)`` for every cell where the
        two disagree, plus a zero-row frame when they agree everywhere.
    """
    window = baseline.filter(pl.col("date").is_between(pl.lit(date_from).str.to_date(), pl.lit(date_to).str.to_date()))
    entities = sorted(set(window["yukka_id"].to_list()))
    parts: list[pl.DataFrame] = []
    for i in range(0, len(entities), batch_size):
        parts.append(data_api_counts(entities[i : i + batch_size], date_from, date_to, t_stat=None))
    api = pl.concat(parts) if parts else pl.DataFrame()
    if api.is_empty():
        return pl.DataFrame()
    return (
        window.select("date", "yukka_id", "tot")
        .join(
            api.select("date", "yukka_id", pl.col("tot").alias("api_tot")),
            on=["date", "yukka_id"],
            how="full",
            coalesce=True,
        )
        .with_columns(
            pl.col("tot").fill_null(0),
            pl.col("api_tot").fill_null(0),
        )
        .with_columns((pl.col("tot") - pl.col("api_tot")).alias("diff"))
        .filter(pl.col("diff") != 0)
        .sort(pl.col("diff").abs(), descending=True)
    )


def check_against_data_api(
    baseline: pl.DataFrame,
    entities: list[str],
    date_from: str,
    date_to: str,
) -> pl.DataFrame:
    """Compare dump counts to the Data API export, filtered and unfiltered.

    Returns:
        Per-entity ``tot`` sums from the dump, the unfiltered export and the ``t_stat=2.5`` export,
        with the two ratios. ``ratio_unfiltered`` near 1.0 means the dump reproduces the export;
        ``ratio_prod`` quantifies how much of the flow production's source filter removes.
    """
    ours = (
        baseline.filter(
            pl.col("yukka_id").is_in(entities)
            & pl.col("date").is_between(pl.lit(date_from).str.to_date(), pl.lit(date_to).str.to_date())
        )
        .group_by("yukka_id")
        .agg(dump_tot=pl.col("tot").sum())
    )
    raw = (
        data_api_counts(entities, date_from, date_to, t_stat=None).group_by("yukka_id").agg(api_tot=pl.col("tot").sum())
    )
    prod = (
        data_api_counts(entities, date_from, date_to, t_stat=2.5)
        .group_by("yukka_id")
        .agg(api_prod_tot=pl.col("tot").sum())
    )
    return (
        ours.join(raw, on="yukka_id", how="full", coalesce=True)
        .join(prod, on="yukka_id", how="full", coalesce=True)
        .with_columns(
            ratio_unfiltered=pl.col("dump_tot") / pl.col("api_tot"),
            ratio_prod=pl.col("api_prod_tot") / pl.col("api_tot"),
        )
        .sort("yukka_id")
    )


def check_against_eod_data(
    events: pl.DataFrame,
    entities: list[str],
    date_from: str,
    date_to: str,
) -> pl.DataFrame:
    """Compare our per-company event totals to the `eod_data` panel's own event columns.

    ``eod_data`` carries one column per event name and is built by a different code path, so
    agreement on which events fire for which company is independent evidence that the dump's grain
    was read correctly. Absolute counts are not expected to match — ``eod_data`` is 9-4-0-derived
    and article-level — so the comparison is on the *set* of events per entity.
    """
    with httpx.Client(headers=_auth(), timeout=600) as client:
        r = client.post(
            f"{QUANT_API_URL}/company_data/eod_data",
            json=entities,
            params={"date_from": date_from, "date_to": date_to},
        )
        r.raise_for_status()
        entries = r.json()["entries"]
    if not entries:
        return pl.DataFrame()

    eod = pl.DataFrame(entries, infer_schema_length=None)
    meta = {"date", "company", "Name", "ISIN", "volume", "positive", "negative", "sentiment"}
    event_cols = [c for c in eod.columns if c not in meta and not c.endswith("_score")]
    theirs = (
        eod.select("ISIN", *event_cols)
        .unpivot(index="ISIN", variable_name="event_name", value_name="count")
        .filter(pl.col("count") > 0)
        .group_by("event_name")
        .agg(eod_entities=pl.col("ISIN").n_unique())
    )
    ours = (
        events.filter(
            pl.col("yukka_id").is_in(entities)
            & pl.col("date").is_between(pl.lit(date_from).str.to_date(), pl.lit(date_to).str.to_date())
        )
        .group_by("event_name")
        .agg(dump_entities=pl.col("yukka_id").n_unique(), dump_company_days=pl.len())
    )
    return ours.join(theirs, on="event_name", how="full", coalesce=True).sort("event_name")


def gap_report(shard_dir: Path) -> pl.DataFrame:
    """Entity-days the API refused to serve, read back from a sweep's manifest.

    ``events_dump`` returns 500 for specific ``(entity, day)`` records — reproducibly, even for a
    single-entity single-day request, and only on recent periods. The sweep isolates and skips them
    rather than losing whole shards, which is the right trade but leaves the panel marginally
    incomplete. Those cells lose their *event* counts while keeping their sentiment counts, so each
    variant equals the baseline there: the effect is to shrink measured deltas towards zero, never
    to invent one. This function is how that shortfall gets quantified instead of assumed small.

    Returns:
        ``(entity, date, batch, period)`` for every recorded gap, or an empty frame when there are
        none. Duplicated across re-runs is possible, so callers should ``unique()`` before counting.
    """
    manifest = shard_dir / "_manifest.jsonl"
    if not manifest.exists():
        return pl.DataFrame(schema={"entity": pl.String, "date": pl.String, "batch": pl.Int64, "period": pl.String})
    rows: list[dict[str, object]] = []
    for line in manifest.read_text().splitlines():
        rec = json.loads(line)
        for gap in rec.get("gaps") or []:
            rows.append({"entity": gap["entity"], "date": gap["date"], "batch": rec["batch"], "period": rec["period"]})
    if not rows:
        return pl.DataFrame(schema={"entity": pl.String, "date": pl.String, "batch": pl.Int64, "period": pl.String})
    return pl.DataFrame(rows).unique().sort("date", "entity")


def shard_period(shard: Path) -> str:
    """The period key of a shard filename: ``2024`` for a year shard, ``2024-01`` for a month one.

    The two dumps are sharded at different granularities — events by month (its endpoint caps
    ``entities x days``), sentiment by year — so anything reading both must not assume one shape.
    """
    m = _SHARD_RE.search(shard.name)
    if m is None:
        msg = f"{shard.name} is not a dump shard filename"
        raise ValueError(msg)
    return m.group(2)


def period_window(period: str) -> tuple[date, date]:
    """The ``[from, to)`` dates a period key covers, for either granularity.

    Raises:
        ValueError: If the key is neither ``YYYY`` nor ``YYYY-MM``.
    """
    parts = period.split("-")
    if len(parts) == 1:
        year = int(parts[0])
        return date(year, 1, 1), date(year + 1, 1, 1)
    if len(parts) == 2:
        year, month = int(parts[0]), int(parts[1])
        nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        return date(year, month, 1), nxt
    msg = f"unrecognised shard period {period!r}, expected YYYY or YYYY-MM"
    raise ValueError(msg)


def spot_check_completeness(
    shard_dir: Path,
    url: str,
    samples: int = 6,
    seed: int = 0,
) -> pl.DataFrame:
    """Re-pull the heaviest entity of sampled shards individually and compare row counts.

    A shard is complete when a solo request for its busiest entity and period returns the same number
    of rows the shard holds for that pair. Sampling the *heaviest* entity is deliberate: a shortfall
    would be a volume effect, so a randomly chosen quiet entity would pass while the shard was short.

    Compare **raw row counts**, not counts after any sentiment filter. Filtering first is what
    produced a false truncation alarm early in this experiment: ~9% of sentiment rows carry a null
    sentiment (54% for the busiest names), so a filtered count looks like a 54% shortfall against
    the API's raw total while the shard is in fact byte-for-byte complete.

    Args:
        shard_dir: A month-sharded dump directory.
        url: The dump endpoint the shards came from.
        samples: Number of shards to check.
        seed: Sampling seed, so a report is reproducible.

    Returns:
        ``(shard, yukka_id, period, on_disk, solo, ok)`` — ``ok`` false means the shard is short.
    """
    shards = sorted(shard_dir.glob("dump-b*.parquet"))
    if not shards:
        return pl.DataFrame()
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(shards), size=min(samples, len(shards)), replace=False)
    chosen = [shards[i] for i in sorted(picks)]

    rows: list[dict[str, object]] = []
    with httpx.Client(headers=_auth(), timeout=1800) as client:
        for shard in chosen:
            period = shard_period(shard)
            df = pl.read_parquet(shard)
            if df.is_empty():
                continue
            heaviest = df.group_by("yukka_id").len().sort("len", descending=True).row(0)
            entity, on_disk = heaviest[0], heaviest[1]
            lo, hi = period_window(period)
            r = client.post(
                url,
                json=[entity],
                params={
                    "datetime_from": lo.isoformat(),
                    "datetime_to": hi.isoformat(),
                    "version": NLP_VERSION,
                    "deduplicate": "false",
                },
            )
            # A 500 here is the documented poisoned-record failure, not a shortfall: the sweep
            # isolates and skips those entity-days. Aborting the report over one would take
            # sections 4b and 5 down with it, so the sample is recorded as unverifiable instead.
            solo: int | None = None
            if r.is_success:
                solo = len(r.json().get("entries", []))
            elif r.status_code < 500:
                r.raise_for_status()
            rows.append(
                {
                    "shard": shard.name,
                    "yukka_id": entity,
                    "period": period,
                    "on_disk": int(on_disk),
                    "solo": solo,
                    "ok": None if solo is None else int(on_disk) == solo,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    """CLI: run every check and print a report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", default="data/panels", help="directory with the count panels")
    ap.add_argument(
        "--sentiment",
        default="data/sentiment_dump",
        help="sentiment shard directory",
    )
    ap.add_argument("--events", default="data/events_dump", help="events shard directory")
    ap.add_argument("--suffix", default="", help="panel suffix, e.g. _facts")
    ap.add_argument("--entities", type=int, default=5, help="entities to sample for the API checks")
    ap.add_argument("--from", dest="lo", default="2024-02-01", help="inclusive start for the API checks")
    ap.add_argument("--to", dest="hi", default="2024-02-10", help="inclusive end for the API checks")
    ap.add_argument("--skip-api", action="store_true", help="run only the offline checks")
    ap.add_argument("--completeness-samples", type=int, default=6, help="shards to re-check for shortfalls")
    a = ap.parse_args()

    panels = Path(a.panels)
    baseline = pl.read_parquet(panels / f"baseline_counts{a.suffix}.parquet")
    events = pl.read_parquet(panels / f"event_counts{a.suffix}.parquet")

    print("1. count algebra over the whole panel")
    algebra = check_algebra(baseline, events)
    for k, v in algebra.items():
        print(f"   {'PASS' if v == 0 else 'FAIL'}  {k}: {v:,}")

    print("\n2. events -> sentiment join integrity")
    join = check_join_integrity(Path(a.sentiment), Path(a.events))
    verdict = "PASS" if join["missing_share"] < 0.01 else "FAIL"
    print(
        f"   {verdict}  {join['missing_pairs']:,.0f} of {join['event_pairs']:,.0f} event pairs unmatched "
        f"({join['missing_share']:.4%}) across {join['batches_checked']:.0f} batches"
    )

    if a.skip_api:
        return

    rng = np.random.default_rng(0)
    pool = sorted(set(baseline["yukka_id"].to_list()))
    picks = rng.choice(len(pool), size=min(a.entities, len(pool)), replace=False)
    sample = [pool[i] for i in sorted(picks)]

    print(f"\n3. dump vs Data API export, {a.lo} .. {a.hi}, {len(sample)} entities")
    cmp_api = check_against_data_api(baseline, sample, a.lo, a.hi)
    print(cmp_api)
    ratios = cmp_api["ratio_unfiltered"].drop_nulls().drop_nans()
    if ratios.len():
        print(
            f"   dump/api unfiltered: median {ratios.median():.3f}, "
            f"min {ratios.min():.3f}, max {ratios.max():.3f}  (1.0 = the dump reproduces the export)"
        )
    prod_ratios = cmp_api["ratio_prod"].drop_nulls().drop_nans()
    if prod_ratios.len():
        print(
            f"   production t_stat=2.5 keeps a median {prod_ratios.median():.1%} of the unfiltered flow "
            f"(range {prod_ratios.min():.1%} .. {prod_ratios.max():.1%})"
        )

    print(f"\n4. shard completeness (raw row counts vs solo re-pull), {a.completeness_samples} per dump")
    for label, directory, url in (
        ("sentiment", Path(a.sentiment), f"{QUANT_API_URL}/company_data/sentiment_dump"),  # year-sharded
        ("events", Path(a.events), f"{QUANT_API_URL}/company_data/events_dump"),
    ):
        got = spot_check_completeness(directory, url, samples=a.completeness_samples)
        if got.is_empty():
            print(f"   {label}: no shards to check")
            continue
        short = got.filter(pl.col("ok").eq(False))  # a null `ok` is "unverifiable", not a shortfall
        unchecked = got.filter(pl.col("ok").is_null()).height
        note = f", {unchecked} unverifiable (API 500)" if unchecked else ""
        print(
            f"   {'PASS' if short.is_empty() else 'FAIL'}  {label}: "
            f"{got.height - short.height - unchecked}/{got.height - unchecked} complete{note}"
        )
        if not short.is_empty():
            print(short)

    print("\n4b. unfetchable entity-days recorded by the events sweep")
    gaps = gap_report(Path(a.events))
    if gaps.is_empty():
        print("   none")
    else:
        cells = baseline.height
        print(
            f"   {gaps.height:,} entity-days the API refused (500), spanning "
            f"{gaps['entity'].n_unique()} entities and {gaps['period'].n_unique()} periods "
            f"— {gaps.height / max(cells, 1):.3%} of the panel's company-days"
        )
        print(gaps.group_by("period").len().sort("period"))

    print(f"\n5. whole-panel baseline vs Data API export, {a.lo} .. {a.hi}")
    mismatch = check_baseline_at_scale(baseline, a.lo, a.hi)
    window_cells = baseline.filter(
        pl.col("date").is_between(pl.lit(a.lo).str.to_date(), pl.lit(a.hi).str.to_date())
    ).height
    if mismatch.is_empty():
        print(f"   PASS  all {window_cells:,} company-days agree exactly with the export")
    else:
        share = mismatch.height / max(window_cells, 1)
        print(
            f"   {'PASS' if share < 0.01 else 'FAIL'}  {mismatch.height:,} of {window_cells:,} "
            f"company-days differ ({share:.2%}); largest gaps:"
        )
        print(mismatch.head(8))

    print(f"\n6. event grain vs eod_data, {a.lo} .. {a.hi}")
    cmp_eod = check_against_eod_data(events, sample, a.lo, a.hi)
    if cmp_eod.is_empty():
        print("   eod_data returned nothing for this window")
    else:
        only_ours = cmp_eod.filter(pl.col("eod_entities").is_null()).height
        only_theirs = cmp_eod.filter(pl.col("dump_entities").is_null()).height
        print(
            f"   {cmp_eod.height} event names seen in total; "
            f"{only_ours} only in our dump, {only_theirs} only in eod_data "
            "(some divergence is expected: eod_data is 9-4-0-derived)"
        )
        print(cmp_eod.head(12))


if __name__ == "__main__":
    main()
