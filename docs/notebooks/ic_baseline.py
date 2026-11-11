# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "basanos>=0.7.3",
#     "marimo>=0.19",
#     "novetly-score",
#     "numpy>=2",
#     "plotly>=6",
#     "polars>=2.1",
#     "python-dotenv>=1.0",
#     "yukka>=3.0.0",
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
    import plotly.graph_objects as go
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
        go,
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
    mo.md("""
    # IC baseline: tone counts and the sentiment ratio, 1–22 days

    Cross-sectional rank IC of `positive`, `neutral`, `negative` and `sentiment` against forward
    returns over the **PIT top 200** of the STOXX 600 — each day's cross-section is the largest
    200 names the index held that day, by that review's own ranking — from `basanos`. Signals
    are raw, with no z-scoring. Only cells whose value is `>= 0` enter a cross-section, so a
    company-day the store has no article for is absent rather than counted as a zero.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data definition and import
    """)
    return


@app.cell
def _():
    signals = ("positive", "neutral", "negative", "sentiment")
    horizons = tuple(range(1, 23))
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
    # One view of the store, both panels off it: same interface, same keys, same column names.
    store = data(Index.STOXX600.frame)
    membership = Index.STOXX600.membership_history
    return membership, store


@app.cell
def _(store):
    # Read from the stored panel, not aggregated: one 1.3 MB object rather than 181M snippet
    # rows. Rebuild with write_counts() when the store gains a month; a panel the store has
    # outgrown is refused rather than served short.
    tone = store.counts(key="ric")
    return (tone,)


@app.cell
def _(store):
    levels = store.prices("2024-07", key="ric")
    return (levels,)


@app.cell
def _(
    entry_day,
    levels,
    membership,
    pit_members,
    sentiment,
    tone,
    trading_days,
):
    days = trading_days(levels, tone["day"].min(), tone["day"].max())
    news = entry_day(tone, days)
    panel = news.join(sentiment(news), on=["ric", "day"]).join(
        pit_members(membership, days, top=200), on=["ric", "day"], how="inner"
    )
    return days, panel


@app.cell
def _(days, levels, panel, pl, tradable, wide):
    candidates = sorted(set(panel.get_column("ric")) & set(levels.get_column("ric")))
    assets = tradable(wide(levels, "close_price", days, candidates), candidates)
    prices = wide(levels, "close_price", days, assets)
    coverage = pl.DataFrame(
        {
            "trading days": [len(days)],
            "candidate RICs": [len(candidates)],
            "assets kept": [len(assets)],
            "company-days": [panel.height],
        }
    )
    coverage
    return assets, prices


@app.cell
def _(
    BasanosConfig,
    BasanosEngine,
    assets,
    days,
    horizons,
    np,
    panel,
    pl,
    prices,
    signals,
    wide,
):
    # The config is required to construct an engine; no part of it touches the IC path.
    cfg = BasanosConfig(vola=20, corr=60, clip=3.0, shrink=0.5, aum=1_000_000)

    # One lagged price frame per horizon, built once and shared by every signal: row t holds the
    # close h-1 days on, so rank_ic(h=1) over it reads the single day t+h-1 -> t+h.
    shifted = {
        h: (prices if h == 1 else prices.with_columns([pl.col(a).shift(-(h - 1)) for a in assets])) for h in horizons
    }

    def curve(signal, *, nonzero=False):
        """Mean rank IC of one signal against the single day *h* ahead, at every horizon.

        Every point uses a one-day return, so the noise is the same across the axis and the
        slope is readable. Pairing the signal with the whole ``h``-day return instead would
        make any persistent signal look stronger at longer horizons whether or not it carries
        information, because drift accumulates like ``h`` while noise grows like ``sqrt(h)``.

        Args:
            signal: The column of *panel* to rank companies by.
            nonzero: Tighten the cross-section. For a count, rank only the company-days with
                at least one article of that sign. For the ratio, rank only the company-days
                carrying **both** polarities, rather than the merely-signed ones.

        The ratio is never ranked on a company-day with no signed article at all. Such a day
        scores exactly 0.5 by the prior, which is not neutral but the 14th percentile of the
        cross-section -- news skews positive, so 0.5 sits near the bottom -- and ranking it
        would read an absence of coverage as bad news.
        """
        if signal == "sentiment":
            signed = (pl.col("positive") + pl.col("negative")) > 0
            both = (pl.col("positive") > 0) & (pl.col("negative") > 0)
            scope = both if nonzero else signed
        else:
            scope = pl.col(signal) > 0 if nonzero else pl.col(signal) >= 0
        mu = wide(panel.filter(scope), signal, days, assets)
        breadth = float(np.isfinite(mu.drop("date").to_numpy()).sum(axis=1).mean())
        rows = []
        for h in horizons:
            series = BasanosEngine(prices=shifted[h], mu=mu, cfg=cfg).rank_ic(h=1)
            values = series.get_column("rank_ic").to_numpy()
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "signal": signal,
                    "h": h,
                    "ic": float(values.mean()),
                    "t": float(values.mean() / values.std(ddof=1) * np.sqrt(len(values))),
                    "names": breadth,
                }
            )
        return rows

    table = pl.DataFrame([row for signal in signals for row in curve(signal)])
    nonzero = pl.DataFrame([row for signal in signals for row in curve(signal, nonzero=True)])
    return nonzero, table


@app.cell
def _(go, pl, signals):
    def plot(table, title):
        """One line per signal across the horizons, with a zero reference."""
        figure = go.Figure()
        for signal in signals:
            part = table.filter(pl.col("signal") == signal).sort("h")
            figure.add_scatter(
                x=part.get_column("h").to_list(),
                y=part.get_column("ic").to_list(),
                name=signal,
                mode="lines+markers",
                customdata=part.select("t", "names").rows(),
                hovertemplate=(
                    "%{fullData.name}<br>h=%{x}<br>IC %{y:.4f}"
                    "<br>t %{customdata[0]:+.1f}<br>names/day %{customdata[1]:.0f}<extra></extra>"
                ),
            )
        figure.add_hline(y=0, line_dash="dot", line_color="grey")
        figure.update_layout(
            title=title,
            xaxis_title="horizon h (trading days)",
            yaxis_title="mean rank IC",
            template="plotly_white",
            height=420,
            hovermode="x unified",
        )
        return figure

    return (plot,)


@app.cell
def _(mo):
    mo.md("""
    ## Rank IC against the single day *t+h-1 → t+h*

    Each point pairs the signal at *t* with a one-day return, so the noise is identical across
    the axis and the slope of the curve is readable. The curves are flat: the signals predict
    day 22 about as well as day 1, which is the mark of a standing cross-sectional tilt rather
    than news being priced in. One standard error is roughly 0.005, so none of the wiggles are
    interpretable.
    """)
    return


@app.cell
def _(plot, table):
    plot(table, "Marginal (one-day) rank IC, PIT top 200")
    return


@app.cell
def _(mo):
    mo.md("""
    ## The same, ranking only company-days whose score is **> 0**

    Above, a company enters the cross-section on any day the store has an article for it, so a
    name whose articles carried no tone of that sign is ranked at zero — tied with every other
    such name. Here those names are dropped instead, which sharpens the ranking at the cost of
    breadth; hover for the names/day each curve is built on. For `sentiment` the tightening is
    `positive > 0 and negative > 0` — both polarities present, rather than merely one signed
    article — since a one-sided ratio is pinned near an extreme by a single article.
    """)
    return


@app.cell
def _(nonzero, plot):
    plot(nonzero, "Marginal rank IC, score > 0 only, PIT top 200")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Mean IC and its t-statistic

    `t` is the mean of the daily IC series over its standard error, `mean / (sd / sqrt(days))`.
    These marginal ICs use one-day returns that do not overlap between consecutive days, so the
    t-statistic is not inflated by overlap the way a cumulative one would be — though the signal
    itself is autocorrelated, so read |t| ≈ 2 as suggestive rather than decisive.
    """)
    return


@app.cell
def _(nonzero, pl, signals, table):
    pl.concat(
        [
            frame.with_columns(scope=pl.lit(label), stat=pl.lit(stat))
            .pivot(on="signal", index=["scope", "stat", "h"], values=stat)
            .with_columns(pl.col(list(signals)).round(5 if stat == "ic" else 2))
            for label, frame in (("all", table), ("tightened", nonzero))
            for stat in ("ic", "t")
        ]
    ).sort("scope", "stat", "h")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
