import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from basanos.math import BasanosConfig, BasanosEngine
    from yukka.data import Index

    from novetly_score.data import data
    from novetly_score.panel import counts, sentiment

    return (
        BasanosConfig,
        BasanosEngine,
        Index,
        counts,
        data,
        mo,
        np,
        pl,
        sentiment,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # IC of the tone counts and the sentiment ratio

        Cross-sectional information coefficient of four signals — `pos`, `neu`, `neg` and
        `sentiment` — against 1- to 10-day forward total returns, over the STOXX 600 as it was
        actually composed. The IC itself is `basanos`: `BasanosEngine.ic` and `.rank_ic`.

        Three choices this rests on:

        - **Point in time throughout.** Every symbol is the one the company held on the day in
          question — `counts(key="ric")` and `prices(key="ric")` key by `as_of`, and the
          membership mask uses `membership_history.ric`, not `ric_current`. Each day's
          cross-section is the 600 names the index held that day.
        - **No look-ahead.** A day's articles are complete only at the end of that UTC day, which
          is after the European close, so news is re-dated to the *next* trading day. Friday,
          Saturday and Sunday news all price into Monday.
        - **Coverage is not zero.** A company-day with no articles carries no signal and is
          dropped from that day's cross-section rather than scored 0.5.

        The two store reads take about four minutes.
        """
    )
    return


@app.cell
def _():
    signals = ("pos", "neu", "neg", "sentiment")
    horizons = tuple(range(1, 11))
    return horizons, signals


@app.cell
def _(pl):
    def trading_days(prices, first, last):
        """The dates the price feed quotes in ``[first, last]``, which become the panel's rows.

        basanos reads a horizon as a row offset, so the row grid has to be trading days for
        ``h`` to mean trading days.
        """
        return prices.filter(pl.col("date").is_between(first, last)).get_column("date").unique().sort()

    def entry_day(counts_frame, days):
        """Re-date each company-day of news to the first trading day strictly after it.

        Returns the counts summed over the calendar days that feed each trading day, so a
        Monday row holds Friday's, Saturday's and Sunday's articles. Every calendar day is used
        exactly once, and none is used before it was published.
        """
        calendar = pl.DataFrame({"news": pl.date_range(days.min(), days.max(), eager=True)})
        mapping = (
            calendar.with_columns(after=pl.col("news").dt.offset_by("1d"))
            .sort("after")
            .join_asof(
                pl.DataFrame({"date": days}).sort("date"),
                left_on="after",
                right_on="date",
                strategy="forward",
            )
            .select("news", "date")
            .drop_nulls("date")
        )
        return (
            counts_frame.join(mapping, left_on="date", right_on="news", suffix="_entry")
            .group_by("date_entry", "ric")
            .agg(pl.col("pos", "neu", "neg").sum())
            .rename({"date_entry": "date"})
        )

    def pit_mask(membership, days):
        """The ``(date, ric)`` pairs the index actually held, in each day's own symbols.

        Membership changes at quarterly reviews, so a day takes the composition of the latest
        review on or before it.
        """
        reviews = membership.filter(pl.col("is_member")).select("effective_date", "ric").unique()
        return (
            pl.DataFrame({"date": days})
            .sort("date")
            .join_asof(
                reviews.select("effective_date").unique().sort("effective_date"),
                left_on="date",
                right_on="effective_date",
                strategy="backward",
            )
            .drop_nulls("effective_date")
            .join(reviews, on="effective_date")
            .select("date", "ric")
        )

    def wide(frame, value, days, rics):
        """``date`` plus one float column per RIC, on exactly *days* and *rics*.

        basanos wants prices and signals as matched wide frames; a RIC with nothing on a day is
        null, which is what excludes it from that day's cross-section.
        """
        pivot = frame.select("date", "ric", value).pivot(on="ric", index="date", values=value)
        absent = [ric for ric in rics if ric not in pivot.columns]
        return (
            pl.DataFrame({"date": days})
            .join(pivot, on="date", how="left")
            .with_columns([pl.lit(None, dtype=pl.Float64).alias(ric) for ric in absent])
            .select(["date", *rics])
            .with_columns(pl.col(rics).cast(pl.Float64))
        )

    def eligible(prices_wide, rics):
        """The RICs whose price column basanos will accept: positive, varying, mostly present.

        Its validators reject a non-positive price, a series that only ever moves one way, and a
        column more than ``max_nan_fraction`` null. Half a window of quotes is the bar here.
        """
        keep = []
        for ric in rics:
            column = prices_wide.get_column(ric)
            quoted = column.drop_nulls()
            if quoted.len() < 3 or quoted.min() <= 0 or column.null_count() / prices_wide.height > 0.5:
                continue
            steps = quoted.diff().drop_nulls()
            if (steps >= 0).all() or (steps <= 0).all():
                continue
            keep.append(ric)
        return keep

    return eligible, entry_day, pit_mask, trading_days, wide


@app.cell
def _(Index, counts, data):
    universe = Index.STOXX600.frame
    membership = Index.STOXX600.membership_history
    tone = counts(universe=universe)
    levels = data(universe).prices(key="ric")
    return levels, membership, tone


@app.cell
def _(entry_day, levels, membership, pit_mask, sentiment, tone, trading_days):
    days = trading_days(levels, tone["date"].min(), tone["date"].max())
    news = entry_day(tone, days)
    scored = news.join(sentiment(news), on=["date", "ric"])
    panel = scored.join(pit_mask(membership, days), on=["date", "ric"], how="inner")
    return days, news, panel, scored


@app.cell
def _(days, eligible, levels, panel, pl, signals, wide):
    candidates = sorted(set(panel.get_column("ric")) & set(levels.get_column("ric")))
    prices_all = wide(levels, "close_price", days, candidates)
    assets = eligible(prices_all, candidates)
    prices_wide = prices_all.select(["date", *assets])
    signals_wide = {signal: wide(panel, signal, days, assets) for signal in signals}

    coverage = pl.DataFrame(
        {
            "trading days": [len(days)],
            "candidate RICs": [len(candidates)],
            "RICs with usable prices": [len(assets)],
            "company-days of news": [panel.height],
        }
    )
    coverage
    return assets, candidates, prices_wide, signals_wide


@app.cell
def _(BasanosConfig, BasanosEngine, horizons, np, pl, prices_wide, signals_wide):
    # The config is required to build an engine but no part of it touches the IC path, which
    # reads only prices, mu and the asset list.
    cfg = BasanosConfig(vola=20, corr=60, clip=3.0, shrink=0.5, aum=1_000_000)

    def ic_table(prices, signals, horizons):
        """Mean IC per signal and horizon, with the daily series' mean/std ratio beside it."""
        rows = []
        for signal, mu in signals.items():
            engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            for h in horizons:
                rank = engine.rank_ic(h=h).get_column("rank_ic").to_numpy()
                pearson = engine.ic(h=h).get_column("ic").to_numpy()
                rank, pearson = rank[np.isfinite(rank)], pearson[np.isfinite(pearson)]
                rows.append(
                    {
                        "signal": signal,
                        "h": h,
                        "rank_ic": float(rank.mean()),
                        "ic": float(pearson.mean()),
                        "icir": float(rank.mean() / rank.std(ddof=1)),
                        "days": len(rank),
                    }
                )
        return pl.DataFrame(rows)

    table = ic_table(prices_wide, signals_wide, horizons)
    return cfg, ic_table, table


@app.cell
def _(mo):
    mo.md(
        """
        ## Mean IC over the window

        `rank_ic` is Spearman and `ic` is Pearson, each averaged over every day in the window.
        `icir` is the mean of the daily rank IC over its standard deviation — basanos's own
        summary, and **not** a t-statistic: at `h > 1` the daily ICs overlap by `h - 1` days, so
        the spread they are divided by understates the real sampling error.
        """
    )
    return


@app.cell
def _(horizons, pl, table):
    table.pivot(on="h", index="signal", values="rank_ic").select("signal", *[pl.col(str(h)).round(5) for h in horizons])
    return


@app.cell
def _(table):
    table
    return


if __name__ == "__main__":
    app.run()
