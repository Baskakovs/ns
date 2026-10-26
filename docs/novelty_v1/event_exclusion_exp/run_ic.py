"""Run the 1 + N arm IC sweep and rank events by their marginal contribution to signal quality.

Steps 3 and 4 of the experiment. For every arm — the unmodified statistic plus one variant per
event — compute the per-date cross-sectional Spearman IC against forward returns at each horizon,
then test whether removing that event *changed* the IC.

**The comparison is paired, and that is the point.** Market-wide IC swings are an order of
magnitude larger than any single event's contribution, so comparing ``mean_ic(variant)`` to
``mean_ic(baseline)`` as two independent samples throws away almost all the power. Instead the
per-date difference

    d_t = ic_variant[t] - ic_baseline[t]

is formed first — the common date effect cancels exactly — and ``mean(d_t)`` is tested against
zero. Standard errors are Newey-West with lag ``h``, because at horizon ``h`` consecutive dates
share ``h - 1`` days of forward return and the ``d_t`` are autocorrelated by construction.

**Multiplicity is corrected.** With ~148 events tested at once, ~7 will clear ``p < 0.05`` by
chance, so uncorrected p-values would manufacture findings. Benjamini-Hochberg q-values are
reported alongside and are what should be read.

IC itself comes from :class:`honey.signals.Signal` — ``Signal.ic`` already computes exactly the
per-date rank correlation against the sum of the next *h* return rows, so nothing is reimplemented
here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from honey.data.preamble import to_returns
from honey.event_exclusion_exp import statistic
from honey.signals import Signal
from honey.strategy.sentiment import _robust_signal as robust_signal

#: Forward horizons in trading days.
HORIZONS = (1, 5, 21)

#: Label for the unmodified arm.
BASELINE = "__baseline__"

#: Production's robust-signal parameters, so `--smooth` reproduces the traded object rather than an
#: approximation of it. Sourced from `honey.strategy.sentiment` and `novelty_knn_model.PROD_PARAMS`.
SPAN = 19
CLIP_C = 4.2

#: Evaluation-date schemes. ``daily`` scores every trading day; ``month-end`` keeps only the last
#: trading day of each month, which is the rebalance cadence but leaves ~32 observations per arm —
#: far too few to rank ~148 events, so it is reported alongside the daily grid rather than instead.
EVAL_SCHEMES = ("daily", "month-end")


def month_end_dates(dates: pl.Series) -> pl.Series:
    """The last trading day present in each calendar month of ``dates``."""
    frame = pl.DataFrame({"date": dates}).drop_nulls("date")
    return (
        frame.group_by(pl.col("date").dt.truncate("1mo").alias("month")).agg(pl.col("date").max()).sort("month")["date"]
    )


def restrict(ic: pl.DataFrame, scheme: str) -> pl.DataFrame:
    """Keep only the evaluation dates ``scheme`` calls for.

    Applied to the per-date IC series *after* it is computed, so the signal and forward returns are
    identical between schemes and only the dates scored differ. That keeps the two comparable.
    """
    if scheme == "daily":
        return ic
    keep = month_end_dates(ic["date"])
    return ic.filter(pl.col("date").is_in(keep))


def newey_west_se(x: np.ndarray, lag: int) -> float:
    """Newey-West standard error of the mean of ``x``, Bartlett-weighted to ``lag``.

    At horizon ``h`` the per-date IC differences overlap by ``h - 1`` days, so the naive
    ``std / sqrt(T)`` understates the standard error and inflates every t-statistic.

    Args:
        x: The series whose mean is being tested. Must be finite.
        lag: Truncation lag. ``0`` reduces to the plain standard error of the mean.

    Returns:
        The standard error, or ``nan`` for fewer than two observations.
    """
    n = len(x)
    if n < 2:
        return float("nan")
    centred = x - x.mean()
    gamma0 = float(centred @ centred) / n
    total = gamma0
    for lead in range(1, min(lag, n - 1) + 1):
        cov = float(centred[lead:] @ centred[:-lead]) / n
        total += 2.0 * (1.0 - lead / (lag + 1.0)) * cov
    if total <= 0:
        return float("nan")
    return float(np.sqrt(total / n))


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values for a family of p-values, preserving input order.

    NaN p-values (an arm with too few dates to test) pass through as NaN and are excluded from the
    family size, so they neither receive nor dilute a correction.
    """
    q = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return q
    order = finite[np.argsort(p[finite])]
    n = order.size
    ranks = np.arange(1, n + 1)
    raw = p[order] * n / ranks
    q[order] = np.minimum.accumulate(raw[::-1])[::-1].clip(max=1.0)
    return q


def arm_ic(
    long: pl.DataFrame,
    bridge: pl.DataFrame,
    prices: pl.DataFrame,
    returns: pl.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    key: str = "ric",
    smooth: bool = False,
) -> dict[int, pl.DataFrame]:
    """Per-date IC of one arm at each horizon, keyed by horizon.

    Args:
        long: ``(date, yukka_id, signal)`` for this arm.
        bridge: ``(yukka_id, ric)``.
        prices: Wide price panel, used only for its trading-day spine.
        returns: Wide simple-return panel.
        horizons: Forward horizons in trading days.
        key: Column the price panel is keyed by — ``"ric"`` for the EUR panel, ``"yukka_id"`` for
            the local-currency one.
        smooth: Apply production's robust transform (EW median over :data:`SPAN` days, lagged,
            clipped, cross-sectionally z-scored) before measuring IC. Off by default because the
            experiment's specification is an end-of-day snapshot; on, this is the object the live
            book actually trades.

    Returns:
        ``{h: DataFrame(date, ic)}``.
    """
    wide = statistic.align(statistic.to_wide(long, bridge, key), prices)
    if smooth:
        wide = robust_signal(wide, [c for c in wide.columns if c != "date"], span=SPAN, clip_c=CLIP_C)
    signal = Signal(wide)
    return {h: signal.ic(returns, h=h, method="spearman") for h in horizons}


def sweep(
    baseline: pl.DataFrame,
    events: pl.DataFrame,
    bridge: pl.DataFrame,
    prices: pl.DataFrame,
    event_names: list[str],
    horizons: tuple[int, ...] = HORIZONS,
    key: str = "ric",
    scheme: str = "daily",
    smooth: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run every arm and return the per-date IC panel and the ranked summary.

    Returns:
        ``(ic_long, summary)``. ``ic_long`` is ``(event, h, date, ic)`` with the baseline stored
        under :data:`BASELINE`; ``summary`` is one row per (event, horizon) with the paired test.
    """
    returns = to_returns(prices)

    by_event = statistic.split_by_event(events)
    base_ic = {
        h: restrict(df, scheme)
        for h, df in arm_ic(
            statistic.baseline_statistic(baseline), bridge, prices, returns, horizons, key, smooth
        ).items()
    }
    ic_parts: list[pl.DataFrame] = [
        df.select(pl.lit(BASELINE).alias("event"), pl.lit(h, pl.Int32).alias("h"), "date", "ic")
        for h, df in base_ic.items()
    ]
    rows: list[dict[str, object]] = []

    # Event exposure, for reading the ranking: a huge delta on an event seen twice is noise.
    exposure = events.group_by("event_name").agg(
        company_days=pl.len(),
        share=(pl.col("tot_a").sum() / baseline["tot"].sum()),
    )
    exposure_map = {r["event_name"]: r for r in exposure.to_dicts()}

    for i, name in enumerate(event_names, start=1):
        variant = statistic.variant_statistic(baseline, by_event, name)
        var_ic = {
            h: restrict(df, scheme) for h, df in arm_ic(variant, bridge, prices, returns, horizons, key, smooth).items()
        }
        for h, df in var_ic.items():
            ic_parts.append(df.select(pl.lit(name).alias("event"), pl.lit(h, pl.Int32).alias("h"), "date", "ic"))
            paired = (
                base_ic[h]
                .rename({"ic": "ic_base"})
                .join(df.rename({"ic": "ic_var"}), on="date", how="inner")
                .drop_nulls(["ic_base", "ic_var"])
            )
            diff = (paired["ic_var"] - paired["ic_base"]).to_numpy()
            diff = diff[np.isfinite(diff)]
            se = newey_west_se(diff, lag=h) if diff.size else float("nan")
            mean_d = float(diff.mean()) if diff.size else float("nan")
            t = mean_d / se if se and np.isfinite(se) and se > 0 else float("nan")
            dof = max(diff.size - 1, 1)
            p = float(2.0 * stats.t.sf(abs(t), dof)) if np.isfinite(t) else float("nan")
            meta = exposure_map.get(name, {})
            rows.append(
                {
                    "event": name,
                    "h": h,
                    "n_dates": int(diff.size),
                    "mean_ic_base": float(np.nanmean(paired["ic_base"].to_numpy())) if paired.height else float("nan"),
                    "mean_ic_var": float(np.nanmean(paired["ic_var"].to_numpy())) if paired.height else float("nan"),
                    "delta_mean_ic": mean_d,
                    "nw_se": se,
                    "t_stat": t,
                    "p": p,
                    "company_days": int(meta.get("company_days", 0)),
                    "match_share": float(meta.get("share", 0.0) or 0.0),
                }
            )
        if i % 20 == 0 or i == len(event_names):
            print(f"  {i}/{len(event_names)} events", flush=True)

    summary = pl.DataFrame(rows)
    # BH within each horizon: the 148 tests at h=1 are one family, h=5 another.
    summary = pl.concat(
        [
            grp.with_columns(pl.Series("q", benjamini_hochberg(grp["p"].to_numpy())))
            for (_, grp) in summary.group_by("h", maintain_order=True)
        ]
    ).sort("h", "delta_mean_ic")
    return pl.concat(ic_parts), summary


def main() -> None:
    """CLI: run the sweep and write the IC panel and summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", default="data/panels", help="directory with the count panels")
    ap.add_argument("--prices", default="data/prices/eur_wide.parquet", help="wide price panel")
    ap.add_argument("--bridge", default="data/prices/id_bridge.parquet", help="yukka_id -> ric bridge")
    ap.add_argument("--out", default="data/results", help="output directory")
    ap.add_argument("--tag", default="eur", help="label for the output files, e.g. eur or local")
    ap.add_argument(
        "--key",
        default="ric",
        choices=("ric", "yukka_id"),
        help="column the price panel is keyed by: ric for the EUR panel, yukka_id for the local one",
    )
    ap.add_argument("--suffix", default="", help="panel suffix, e.g. _facts")
    ap.add_argument("--min-company-days", type=int, default=0, help="drop events rarer than this")
    ap.add_argument(
        "--eval",
        dest="scheme",
        default="daily",
        choices=EVAL_SCHEMES,
        help="evaluation dates: every trading day, or only each month's last trading day",
    )
    ap.add_argument(
        "--accumulate",
        action="store_true",
        help="roll non-trading-day counts into the next trading day instead of discarding them (~8%% of matches)",
    )
    ap.add_argument(
        "--ratio",
        default="net",
        choices=sorted(statistic.RATIOS),
        help="cross-sectional statistic: net is (pos-neg)/tot, laplace is production's (pos+1)/(pos+neg+2)",
    )
    ap.add_argument(
        "--smooth",
        action="store_true",
        help="apply production's robust transform (EW median span 19, lagged, clipped, z-scored) before IC",
    )
    a = ap.parse_args()
    statistic.set_ratio(a.ratio)

    panels = Path(a.panels)
    baseline = pl.read_parquet(panels / f"baseline_counts{a.suffix}.parquet")
    events = pl.read_parquet(panels / f"event_counts{a.suffix}.parquet")
    prices = pl.read_parquet(a.prices).drop_nulls("date").sort("date")
    bridge = pl.read_parquet(a.bridge)

    if a.accumulate:
        before = baseline["tot"].sum()
        baseline = statistic.to_trading_day(baseline, prices)
        events = statistic.to_trading_day(events, prices, group_extra=("event_name",))
        print(f"accumulated onto trading days: {before:,} -> {baseline['tot'].sum():,} matches retained", flush=True)

    names = statistic.event_universe(events, a.min_company_days)
    print(
        f"{len(names)} events x {len(HORIZONS)} horizons | "
        f"{baseline.height:,} company-days | {prices.width - 1} price columns | "
        f"statistic {statistic.active_ratio()}{' + production smoothing' if a.smooth else ''}",
        flush=True,
    )

    ic_long, summary = sweep(baseline, events, bridge, prices, names, key=a.key, scheme=a.scheme, smooth=a.smooth)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ic_long.write_parquet(out / f"event_ic_long_{a.tag}{a.suffix}.parquet", compression="zstd")
    summary.write_parquet(out / f"event_ic_summary_{a.tag}{a.suffix}.parquet", compression="zstd")

    print(
        "\nbaseline mean IC: "
        + ", ".join(f"h={h}: {summary.filter(pl.col('h') == h)['mean_ic_base'][0]:+.4f}" for h in HORIZONS)
    )
    worst = summary.filter(pl.col("h") == 1).head(8)
    print("\nmost IC-improving to remove (h=1, most negative delta means removal HURT):")
    for r in summary.filter(pl.col("h") == 1).sort("delta_mean_ic", descending=True).head(8).to_dicts():
        print(
            f"  {r['event'][:42]:42s} d={r['delta_mean_ic']:+.5f} t={r['t_stat']:+.2f} "
            f"q={r['q']:.3f} share={r['match_share']:.4f}"
        )
    print("\nmost IC-damaging to remove (h=1):")
    for r in worst.to_dicts():
        print(
            f"  {r['event'][:42]:42s} d={r['delta_mean_ic']:+.5f} t={r['t_stat']:+.2f} "
            f"q={r['q']:.3f} share={r['match_share']:.4f}"
        )


if __name__ == "__main__":
    main()
