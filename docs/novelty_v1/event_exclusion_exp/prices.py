"""Price panels for the event-exclusion experiment: the EUR headline and the wide local-currency check.

Two feeds, because neither one alone gives both breadth and a common currency:

* :func:`fetch_eur_constituent_prices` — ``GET /company_data/constituent_prices``. ~315 names,
  primary listings, normalised to EUR, point-in-time. This is the headline panel and the feed
  production uses. ``close_price`` is EUR on all but a handful of rows per name (~23 of 738 dates,
  mostly market holidays such as 2023-12-25) where the conversion did not apply and the per-row
  ``currency`` field correctly reports the local currency instead. Those rows are nulled rather
  than converted — they are non-trading days, so nothing is lost. Verified against the committed
  production panel ``src/honey/data/cache/prices/top200/2026-08-13.parquet``: 218k overlapping
  cells, median relative difference 0.0, p99 2e-16.
* :func:`fetch_local_panel` — ``POST /trading/portfolio/price_data_itemized``. 946 of the 951
  ever-members return data, including names delisted mid-sample, so it is free of the headline
  panel's survivorship gap — but each name comes back on one arbitrary listing in local currency,
  and 45 of them are US ADRs. Those are dropped; the rest stay in local currency because implied FX
  from the two feeds disagrees by ~3.8% across names (different listings and price scales), so a
  derived rate would be worse than none. Used only to check the *ranking* of events is stable.

Returns come from :func:`honey.data.preamble.to_returns` — simple, not log, which is what
``Signal.ic`` requires since it sums forward rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from honey.data.utils import clean_prices
from honey.event_exclusion_exp._io import auth_headers

#: Prices live on the Quant API, not the Data API the sentiment/events dumps use.
QUANT_API_URL = "https://quant.api.yukkalab.com"

#: Exchange names that are ADR/secondary US listings rather than the European primary line.
ADR_EXCHANGES = frozenset({"US", "USA Stocks", "NYSE", "NASDAQ"})

#: Entities per `price_data_itemized` request. 200 returns in ~3s; the endpoint has no documented cap.
LOCAL_BATCH = 200


def _auth() -> dict[str, str]:
    """Bearer header, via the package's env-then-``.env`` token lookup."""
    return auth_headers()


def fetch_eur_constituent_prices(date_from: str, date_to: str) -> pl.DataFrame:
    """Long EUR price frame from ``constituent_prices``: (date, ric, yukka_id, close_price).

    Rows whose ``currency`` is not EUR carry an unconverted local price and would otherwise sit in
    the panel looking perfectly well-formed, so ``close_price`` is nulled on those rows. They are a
    fraction of a percent and land on market holidays.

    Args:
        date_from: Inclusive ISO start date.
        date_to: Exclusive ISO end date, per the endpoint's own convention.

    Returns:
        Long frame ``(date, ric, yukka_id, close_price, currency, original_currency)`` with
        ``date`` as ``pl.Date`` and ``close_price`` null wherever the row was not EUR.

    Raises:
        RuntimeError: If no row came back EUR at all, which would mean the feed changed.
    """
    with httpx.Client(headers=_auth(), timeout=900) as client:
        r = client.get(
            f"{QUANT_API_URL}/company_data/constituent_prices",
            params={"date_from": date_from, "date_to": date_to},
        )
        r.raise_for_status()
        entries: list[dict[str, Any]] = r.json()["entries"]

    df = pl.DataFrame(entries, infer_schema_length=None)
    if "EUR" not in set(df["currency"].unique().to_list()):
        msg = (
            f"constituent_prices returned no EUR rows (currencies={sorted(set(df['currency'].unique()))}); "
            "the feed's EUR normalisation appears to have changed"
        )
        raise RuntimeError(msg)
    return df.select(
        pl.col("date").cast(pl.Date),
        pl.col("ric").cast(pl.String),
        pl.col("yukka_id").cast(pl.String),
        pl.when(pl.col("currency") == "EUR")
        .then(pl.col("close_price").cast(pl.Float64))
        .otherwise(None)
        .alias("close_price"),
        pl.col("currency").cast(pl.String),
        pl.col("original_currency").cast(pl.String),
    ).sort("date", "ric")


def mask_membership(wide: pl.DataFrame, rank_range: tuple[int, int] | None = None) -> pl.DataFrame:
    """Null out prices for dates on which a name was not a point-in-time index member.

    Without this the cross-section silently includes companies before they joined the index and
    after they left — 5.0% of priced cells on the 2024-2026 EUR panel, taking the median
    cross-section from 314 names to 298. Over a two-year window that is a modest distortion; over
    the ten-year history it is the difference between ranking the index and ranking whatever names
    happen to have a price today.

    Membership comes from the SDK's ``Index.STOXX600.membership_intervals`` — a wide date x RIC panel
    of daily market-cap ranks where a null rank means "not a member". It is the same source
    ``YukkaRepository.prices(mask=True)`` uses, so this panel and production agree on the universe.

    Args:
        wide: Wide ``date x ric`` price panel.
        rank_range: Optional inclusive ``(lo, hi)`` market-cap rank band, e.g. ``(1, 200)`` for the
            production top-200 universe. ``None`` keeps every member.

    Returns:
        The same panel with non-member cells nulled. Columns absent from the membership panel are
        left untouched and reported, rather than silently dropped or silently kept.
    """
    from yukka.data import Index

    ranks = Index.STOXX600.membership_intervals
    cols = [c for c in wide.columns if c != "date"]
    known = [c for c in cols if c in ranks.columns]
    unknown = [c for c in cols if c not in ranks.columns]
    if unknown:
        print(
            f"mask_membership: {len(unknown)} column(s) absent from the membership panel, left "
            f"unmasked: {', '.join(unknown[:6])}{' ...' if len(unknown) > 6 else ''}",
            flush=True,
        )
    if not known:
        return wide

    member = ranks.select(
        "date",
        *[
            (
                pl.col(c).is_not_null()
                if rank_range is None
                else (pl.col(c).is_not_null() & pl.col(c).is_between(*rank_range))
            ).alias(f"_m_{c}")
            for c in known
        ],
    )
    joined = wide.join(member, on="date", how="left")
    return joined.select(
        "date",
        *[
            (
                pl.when(pl.col(f"_m_{c}").fill_null(value=False)).then(pl.col(c)).otherwise(None).alias(c)
                if c in known
                else pl.col(c)
            )
            for c in cols
        ],
    ).sort("date")


def to_wide(long: pl.DataFrame, on: str = "ric", values: str = "close_price") -> pl.DataFrame:
    """Pivot a long price frame to the wide ``date x entity`` panel the Signal API expects.

    Sentinel values are stripped via :func:`honey.data.utils.clean_prices` — the endpoint documents
    that provider sentinels (e.g. ``-9000000``) are passed through raw, and one such value would
    otherwise become a catastrophic return.
    """
    wide = long.drop_nulls("date").pivot(on=on, index="date", values=values, aggregate_function="last").sort("date")
    return clean_prices(wide)


def id_bridge(long: pl.DataFrame) -> pl.DataFrame:
    """The ``yukka_id -> ric`` map, taken from the price feed itself.

    ``constituent_prices`` returns both identifiers on every row, so the bridge needs no separate
    lookup against ``YukkaRepository.assets``. Deduped on ``yukka_id``: a dual-listed name resolves
    to one RIC, chosen by sort order so the assignment is deterministic across runs.
    """
    return (
        long.select("yukka_id", "ric")
        .drop_nulls()
        .unique()
        .sort("yukka_id", "ric")
        .unique(subset="yukka_id", keep="first", maintain_order=True)
    )


def fetch_local_panel(entities: list[str], date_from: str, date_to: str) -> pl.DataFrame:
    """Long local-currency price frame for the wide robustness universe, ADR listings dropped.

    Args:
        entities: ``yukka_id`` list (e.g. ``["company:bmw", ...]``).
        date_from: Inclusive ISO start date.
        date_to: Exclusive ISO end date, per the endpoint's own convention.

    Returns:
        Long frame ``(date, yukka_id, close_price, currency, exchange_name, ticker)``.
    """
    rows: list[dict[str, Any]] = []
    with httpx.Client(headers=_auth(), timeout=900) as client:
        for i in range(0, len(entities), LOCAL_BATCH):
            r = client.post(
                f"{QUANT_API_URL}/trading/portfolio/price_data_itemized",
                json=entities[i : i + LOCAL_BATCH],
                params={"date_from": date_from, "date_to": date_to},
            )
            r.raise_for_status()
            for item in r.json()["price_data_itemized"]:
                if item.get("exchange_name") in ADR_EXCHANGES:
                    continue
                yid = item["entity_info"]["compound_key"]
                rows.extend(
                    {
                        "date": v["date"],
                        "yukka_id": yid,
                        "close_price": v["value"],
                        "currency": item.get("currency"),
                        "exchange_name": item.get("exchange_name"),
                        "ticker": item.get("ticker"),
                    }
                    for v in (item.get("values") or [])
                )
    if not rows:
        msg = "price_data_itemized returned no usable listings"
        raise RuntimeError(msg)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .select(
            pl.col("date").cast(pl.Date),
            pl.col("yukka_id").cast(pl.String),
            pl.col("close_price").cast(pl.Float64),
            pl.col("currency").cast(pl.String),
            pl.col("exchange_name").cast(pl.String),
            pl.col("ticker").cast(pl.String),
        )
        .sort("date", "yukka_id")
    )


def main() -> None:
    """CLI: pull both panels and write them as parquet."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/prices", help="output directory")
    ap.add_argument("--from", dest="lo", default="2023-11-01", help="inclusive start (early for return warmup)")
    ap.add_argument("--to", dest="hi", default="2026-09-01", help="inclusive end")
    ap.add_argument("--universe", default="data/stoxx600_ever_members.parquet", help="parquet with a yukka_id column")
    ap.add_argument("--skip-local", action="store_true", help="only pull the EUR headline panel")
    ap.add_argument(
        "--no-mask",
        action="store_true",
        help="skip point-in-time index-membership masking (keeps names outside their membership window)",
    )
    ap.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="restrict to market-cap rank <= this, e.g. 200 for the production universe",
    )
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    band = None if a.max_rank is None else (1, a.max_rank)
    eur = fetch_eur_constituent_prices(a.lo, a.hi)
    eur.write_parquet(out / "eur_long.parquet", compression="zstd")
    wide = to_wide(eur)
    if not a.no_mask:
        wide = mask_membership(wide, band)
    wide.write_parquet(out / "eur_wide.parquet", compression="zstd")
    bridge_frame = id_bridge(eur)
    bridge_frame.write_parquet(out / "id_bridge.parquet", compression="zstd")
    print(
        f"EUR panel: {wide.height} dates x {wide.width - 1} RICs, "
        f"{eur['date'].min()} .. {eur['date'].max()} -> {out / 'eur_wide.parquet'}",
        flush=True,
    )

    if a.skip_local:
        return
    ids = pl.read_parquet(a.universe)["yukka_id"].drop_nulls().unique().sort().to_list()
    loc = fetch_local_panel([str(i) for i in ids], a.lo, a.hi)
    loc.write_parquet(out / "local_long.parquet", compression="zstd")
    loc_wide = to_wide(loc, on="yukka_id")
    # The membership panel is RIC-keyed, so the local panel is masked via the bridge.
    if not a.no_mask:
        ric_of = dict(zip(bridge_frame["yukka_id"], bridge_frame["ric"], strict=True))
        renamed = loc_wide.rename({y: ric_of[y] for y in loc_wide.columns if y in ric_of})
        masked = mask_membership(renamed, band)
        back = {v: k for k, v in ric_of.items()}
        loc_wide = masked.rename({r: back[r] for r in masked.columns if r in back})
    loc_wide.write_parquet(out / "local_wide.parquet", compression="zstd")
    print(
        f"local panel: {loc_wide.height} dates x {loc_wide.width - 1} entities "
        f"({loc['currency'].n_unique()} currencies) -> {out / 'local_wide.parquet'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
