import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from basanos.math import BasanosConfig, BasanosEngine
    from yukka.data import Index

    from novetly_score.align import entry_day, pit_members, tradable, trading_days, wide
    from novetly_score.data import data, read_counts
    from novetly_score.panel import sentiment

    return (
        BasanosConfig,
        BasanosEngine,
        Index,
        data,
        entry_day,
        go,
        mo,
        np,
        pit_members,
        pl,
        read_counts,
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
    returns over the PIT STOXX 600, from `basanos`. Signals are raw — no z-scoring. Only cells
    whose value is `>= 0` enter a cross-section, so a company-day with no articles is absent
    rather than counted as a zero.
    """)
    return


@app.cell
def _():
    signals = ("positive", "neutral", "negative", "sentiment")
    horizons = tuple(range(1, 23))
    return horizons, signals


@app.cell
def _(Index, data):
    # One view of the store, both panels off it: same interface, same keys, same column names.
    store = data(Index.STOXX600.frame)
    membership = Index.STOXX600.membership_history
    return membership, store


@app.cell
def _(read_counts):
    # Materialised in the store: one 1.3 MB object, not the aggregation over 181M snippet rows
    # that produced it. Rebuild with write_counts() when the store gains a month; read_counts()
    # refuses a panel the store has outgrown.
    tone = read_counts()
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
        pit_members(membership, days), on=["ric", "day"], how="inner"
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

    def curve(signal):
        """Mean rank IC of one signal against the single day *h* ahead, at every horizon.

        Every point uses a one-day return, so the noise is the same across the axis and the
        slope is readable. Pairing the signal with the whole ``h``-day return instead would
        make any persistent signal look stronger at longer horizons whether or not it carries
        information, because drift accumulates like ``h`` while noise grows like ``sqrt(h)``.
        """
        mu = wide(panel.filter(pl.col(signal) >= 0), signal, days, assets)
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
                }
            )
        return rows

    table = pl.DataFrame([row for signal in signals for row in curve(signal)])
    return (table,)


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
                customdata=part.get_column("t").to_list(),
                hovertemplate="%{fullData.name}<br>h=%{x}<br>IC %{y:.4f}<br>t %{customdata:+.1f}<extra></extra>",
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
    plot(table, "Marginal (one-day) rank IC, PIT STOXX 600")
    return


@app.cell
def _(pl, signals, table):
    table.pivot(on="signal", index="h", values="ic").sort("h").with_columns(pl.col(list(signals)).round(5))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
