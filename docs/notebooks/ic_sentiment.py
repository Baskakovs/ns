# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "basanos>=0.7.3",
#     "marimo>=0.19",
#     "novetly-score",
#     "numpy>=2",
#     "plotly>=6",
#     "polars==2.1.0",
#     "python-dotenv==1.2.4",
#     "yukka==3.0.0",
# ]
#
# # The BOOK job exports these with `marimo export html --sandbox`, which builds an isolated
# # environment from this header alone -- a notebook-group dependency is invisible to it. The
# # project is a path dependency so polars, duckdb, boto3, yukka and python-dotenv come with it.
# [tool.uv.sources]
# novetly-score = { path = "../..", editable = false }
#
# # basanos 0.7.3 imports cvx.linalg.ewm_cov, dropped in cvx-linalg 0.9, but pins only >=0.6.1.
# [tool.uv]
# constraint-dependencies = ["cvx-linalg<0.9"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import numpy as np
    import polars as pl
    from basanos.math import BasanosConfig, BasanosEngine
    from dotenv import load_dotenv
    from yukka.data import Index

    from novetly_score.align import entry_day, pit_members, tradable, trading_days, wide
    from novetly_score.data import data
    from novetly_score.data.s3 import ACCESS_KEY_VAR, ENDPOINT_VAR, SECRET_VAR
    from novetly_score.panel import sentiment

    return (
        ACCESS_KEY_VAR,
        BasanosConfig,
        BasanosEngine,
        ENDPOINT_VAR,
        Index,
        SECRET_VAR,
        data,
        entry_day,
        load_dotenv,
        mo,
        np,
        os,
        pit_members,
        pl,
        sentiment,
        tradable,
        trading_days,
        wide,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # IC of the tone counts and the sentiment ratio

        Cross-sectional information coefficient of four signals — `positive`, `neutral`,
        `negative` and `sentiment` — against 1- to 10-day forward total returns, over the STOXX 600 as it was
        actually composed. The IC itself is `basanos`: `BasanosEngine.ic` and `.rank_ic`.

        Three choices this rests on:

        - **Point in time throughout.** Every symbol is the one the company held on the day in
          question — `store.counts(key="ric")` and `store.prices(key="ric")` key by `as_of`, and the
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
    signals = ("positive", "neutral", "negative", "sentiment")
    horizons = tuple(range(1, 11))
    return horizons, signals


@app.cell
def _(ACCESS_KEY_VAR, ENDPOINT_VAR, Index, SECRET_VAR, data, load_dotenv, mo, os):
    # The MARIMO and BOOK jobs run this notebook with no credentials, so stop here with a note
    # rather than fail: every cell below needs the store and a YUKKA token. .env is loaded the
    # way the data layer loads it, so a local run with a .env is unaffected.
    load_dotenv()
    missing = [name for name in (ENDPOINT_VAR, ACCESS_KEY_VAR, SECRET_VAR, "YUKKA_TOKEN") if not os.environ.get(name)]
    mo.stop(
        bool(missing),
        mo.md(f"**Not run.** Set `{'`, `'.join(missing)}` to read the store and the universe."),
    )
    # counts and prices are now the same call: span and key are the arguments. The counts panel
    # is read from the store rather than aggregated per caller -- see novetly_score.data.panels.
    store = data(Index.STOXX600.frame)
    membership = Index.STOXX600.membership_history
    tone = store.counts(key="ric")
    levels = store.prices("2024-07", key="ric")
    return levels, membership, store, tone


@app.cell
def _(entry_day, levels, membership, pit_members, sentiment, tone, trading_days):
    days = trading_days(levels, tone["day"].min(), tone["day"].max())
    news = entry_day(tone, days)
    scored = news.join(sentiment(news), on=["ric", "day"])
    panel = scored.join(pit_members(membership, days), on=["ric", "day"], how="inner")
    return days, news, panel, scored


@app.cell
def _(days, levels, panel, pl, signals, tradable, wide):
    candidates = sorted(set(panel.get_column("ric")) & set(levels.get_column("ric")))
    prices_all = wide(levels, "close_price", days, candidates)
    assets = tradable(prices_all, candidates)
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
