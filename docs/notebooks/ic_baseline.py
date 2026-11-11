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
    from novetly_score.data import data
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
def _(Index, data):
    # One view of the store, both panels off it: same interface, same keys, same column names.
    store = data(Index.STOXX600.frame)
    membership = Index.STOXX600.membership_history
    return membership, store


@app.cell
def _(store):
    # Materialised in the store: one 1.3 MB object, not the aggregation over 181M snippet rows
    # that produced it. Rebuild with write_counts() when the store gains a month; read_counts()
    # refuses a panel the store has outgrown.
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
def _(mo):
    mo.md("""
    ## Is sparsity driving the ranks?

    The ranks below are the ones `spearmanr` actually uses: per day, over the same cross-section
    the IC is computed on, with ties taking their average rank. Each is normalised to [0, 1] so
    days of different width are comparable — a tie-free cross-section would be flat across the
    axis, and every spike is a block of companies sharing one score.
    """)
    return


@app.cell
def _(assets, panel, pl):
    ranked = (
        panel.filter(pl.col("ric").is_in(assets) & ((pl.col("positive") + pl.col("negative")) > 0))
        .with_columns(depth=pl.col("positive") + pl.col("negative"))
        .with_columns(
            rank=pl.col("sentiment").rank(method="average").over("day"),
            width=pl.len().over("day"),
        )
        .with_columns(pct=(pl.col("rank") - 1) / (pl.col("width") - 1))
    )
    ties = ranked.group_by("day").agg(
        distinct=pl.col("sentiment").n_unique(),
        width=pl.len(),
        biggest_tie=pl.col("sentiment").value_counts().struct.field("count").max(),
    )
    pl.DataFrame(
        {
            "company-days": [ranked.height],
            "names/day (median)": [ties.get_column("width").median()],
            "distinct scores/day": [ties.get_column("distinct").median()],
            "largest tied block/day": [ties.get_column("biggest_tie").median()],
            "depth = 0 (no signed article)": [
                f"{100 * ranked.filter(pl.col('depth') == 0).height / ranked.height:.1f}%"
            ],
        }
    )
    return (ranked,)


@app.cell
def _(go, np, ranked):
    # binned here rather than by plotly, which would otherwise embed all 116k raw values
    _counts, _edges = np.histogram(ranked.get_column("pct").to_numpy(), bins=60, range=(0, 1))
    _figure = go.Figure(go.Bar(x=(_edges[:-1] + _edges[1:]) / 2, y=_counts, width=1 / 60, marker_line_width=0))
    _figure.update_layout(
        title="Assigned rank of sentiment, normalised per day",
        xaxis_title="rank within the day's cross-section (0 = lowest, 1 = highest)",
        yaxis_title="company-days",
        template="plotly_white",
        height=380,
    )
    _figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Signed article count per company-day

    The depth each score is estimated from, and so the source of the spikes above: a ratio built
    from one or two signed articles can only land on a handful of values, and every company
    sharing that value shares its rank.
    """)
    return


@app.cell
def _(go, np, pl, ranked):
    _depth = ranked.filter(pl.col("depth") <= 150).get_column("depth").to_numpy()
    _counts, _edges = np.histogram(_depth, bins=75, range=(0, 150))
    _figure = go.Figure(go.Bar(x=(_edges[:-1] + _edges[1:]) / 2, y=_counts, width=2, marker_line_width=0))
    _figure.update_layout(
        title="positive + negative per company-day (clipped at 150)",
        xaxis_title="signed articles",
        yaxis_title="company-days",
        template="plotly_white",
        height=380,
    )
    _figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Where the thin names land in the ranking

    The question this section is for: if sparsity drove the ranks, shallow company-days would sit
    systematically in one part of the ordering rather than spread across it. Mean normalised rank
    by depth bucket, with the share of all company-days each bucket holds — a flat line near 0.5
    means depth is not deciding rank.
    """)
    return


@app.cell
def _(go, pl, ranked):
    _buckets = (
        ranked.with_columns(
            bucket=pl.when(pl.col("depth") == 0)
            .then(pl.lit("0"))
            .when(pl.col("depth") <= 2)
            .then(pl.lit("1-2"))
            .when(pl.col("depth") <= 5)
            .then(pl.lit("3-5"))
            .when(pl.col("depth") <= 10)
            .then(pl.lit("6-10"))
            .when(pl.col("depth") <= 20)
            .then(pl.lit("11-20"))
            .when(pl.col("depth") <= 50)
            .then(pl.lit("21-50"))
            .when(pl.col("depth") <= 100)
            .then(pl.lit("51-100"))
            .otherwise(pl.lit("100+"))
        )
        .group_by("bucket")
        .agg(mean_rank=pl.col("pct").mean(), share=pl.len() / ranked.height)
    )
    _order = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "100+"]
    _b = _buckets.with_columns(order=pl.col("bucket").replace_strict(_order, range(len(_order)))).sort("order")
    _figure = go.Figure()
    _figure.add_bar(
        x=_b.get_column("bucket").to_list(),
        y=_b.get_column("mean_rank").to_list(),
        customdata=[[v] for v in _b.get_column("share").to_list()],
        hovertemplate="depth %{x}<br>mean rank %{y:.3f}<br>%{customdata[0]:.1%} of company-days<extra></extra>",
        name="mean normalised rank",
    )
    _figure.add_hline(y=0.5, line_dash="dot", line_color="grey")
    _figure.update_layout(
        title="Mean normalised rank of sentiment by signed-article depth",
        xaxis_title="signed articles that day",
        yaxis_title="mean rank (0.5 = middle of the cross-section)",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=380,
    )
    _figure
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
