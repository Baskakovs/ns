# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.19.6",
#     "honey[research]",
# ]
# [tool.uv.sources]
# honey = { path = "../../.." , editable = true }
# ///
"""Which detected events make the sentiment signal worse? — interactive event-exclusion explorer.

The hypothesis: Yukka's sentiment dataset is a superset of the event dataset, and some of the ~147
event types contribute negatively to signal quality. Remove one event's articles from the sentiment
statistic; if the information coefficient goes *up*, that event was noise or contrarian.

    · BASELINE       — (pos - neg) / tot, unmodified. THE BAR.
    · minus <event>  — ((pos - pos_a) - (neg - neg_a)) / (tot - tot_a), one arm per event.

**Every event is included here — nothing is filtered out.** The rarity control highlights and sorts,
it never drops rows, because "which events were excluded from the chart" is exactly the question a
reader of a 147-way ranking should not have to ask.

Read three columns together, or the ranking will mislead you:

    · delta_mean_ic — positive means *removing* the event improved IC
    · q             — Benjamini-Hochberg FDR across all events at that horizon. Read this, not p:
                      147 simultaneous tests put ~7 events under p<0.05 by chance alone
    · match_share   — how much of the sentiment flow the event accounts for. The extremes of the
                      ranking are dominated by rare events whose deltas are estimation noise

Sections: 1 panels & exposure · 2 baseline IC · 3 the full ranking · 4 per-event drill-down
· 5 frequency vs effect · 6 cross-arm agreement.

Thin by design: the per-date IC series are read straight from the pipeline's parquet output rather
than recomputed here, so opening the notebook costs no API calls and no signal rebuild. Everything
experiment-specific lives in `honey.event_exclusion_exp`; see that package's CLAUDE.md for the data
routes and the API defects worked around to get here.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl

    import honey  # noqa: F401  (import side effect: loads YUKKA_TOKEN from repo-root .env)
    from honey.event_exclusion_exp import statistic
    from honey.event_exclusion_exp.run_ic import BASELINE

    #: Where the pipeline wrote its outputs. Override with ``EVENT_EXCLUSION_DATA`` to point at a
    #: variant run — e.g. the month-end/full-history build — without editing the notebook.
    #: Regenerate with, in order: events_sweep.py, prices.py, panel.py, run_ic.py.
    DATA = Path(os.environ.get("EVENT_EXCLUSION_DATA", "data"))
    PANELS, PRICES, RESULTS = DATA / "panels", DATA / "prices", DATA / "results"

    COLORS = {
        "blend": "#11D48E",
        "momentum": "#2E6FB0",
        "sentiment": "#E8843A",
        "benchmark": "#9AAEAE",
        "neg": "#E5484D",
        "zero": "#898781",
    }

    #: Arms the pipeline may have produced, in the order they should be offered. Only those actually
    #: present on disk are shown, so a partial run still opens.
    ARMS = {
        "eur_prodmask": "production-equivalent (laplace + smoothing + PIT universe)",
        "eur_prod": "production statistic + smoothing, wide universe",
        "eur_laplace": "production statistic, no smoothing",
        "eur": "EUR panel (headline, net ratio)",
        "eur_facts": "facts-only events",
        "eur_pit": "EUR panel, point-in-time membership",
        "eur_accum": "weekend news rolled forward",
        "local": "local-currency panel (~640 names/date)",
        "eur_monthend": "month-end evaluation dates",
    }

    def available_arms() -> dict[str, str]:
        """The arms whose summary parquet exists, label-keyed for a dropdown."""
        return {label: tag for tag, label in ARMS.items() if (RESULTS / f"event_ic_summary_{tag}.parquet").exists()}

    def load_summary(tag: str) -> pl.DataFrame:
        """One arm's per-(event, horizon) summary."""
        return pl.read_parquet(RESULTS / f"event_ic_summary_{tag}.parquet")

    def load_ic(tag: str) -> pl.DataFrame:
        """One arm's per-date IC series, all events plus the baseline."""
        return pl.read_parquet(RESULTS / f"event_ic_long_{tag}.parquet")


@app.cell
def _():
    mo.md("""
    # Which events make the sentiment signal worse?

    One arm per detected event, each removing that event's articles from the daily sentiment
    statistic, ranked by the **paired** change in cross-sectional Spearman IC. Every event in
    the data is shown — the controls sort and highlight, they never filter rows away.
    """)
    return


@app.cell
def _():
    _arms = available_arms()
    arm = mo.ui.dropdown(options=_arms, value=next(iter(_arms)), label="arm")
    mo.md(f"**Arms found on disk:** {len(_arms)} — {', '.join(_arms)}")
    return (arm,)


@app.cell
def _():
    mo.md("""
    ## 1 Panels and event exposure
    """)
    return


@app.cell
def _():
    baseline = pl.read_parquet(PANELS / "baseline_counts.parquet")
    events = pl.read_parquet(PANELS / "event_counts.parquet")
    prices = pl.read_parquet(PRICES / "eur_wide.parquet").drop_nulls("date").sort("date")
    bridge = pl.read_parquet(PRICES / "id_bridge.parquet")
    mo.md(
        f"**Baseline:** {baseline.height:,} company-days · {baseline['yukka_id'].n_unique()} entities · "
        f"{baseline['date'].min()} … {baseline['date'].max()} · {baseline['tot'].sum():,} sentiment matches.<br>"
        f"**Events:** {events.height:,} company-day-events over **{events['event_name'].n_unique()} distinct "
        f"events** · {events['tot_a'].sum() / baseline['tot'].sum():.1%} of matches carry a tag "
        f"(events overlap, so this exceeds the per-event shares).<br>"
        f"**Prices:** {prices.height} trading days × {prices.width - 1} RICs · {bridge.height} entities bridged."
    )
    return baseline, bridge, events, prices


@app.cell
def _(baseline, events):
    exposure = (
        events.group_by("event_name")
        .agg(
            company_days=pl.len(),
            matches=pl.col("tot_a").sum(),
            share=pl.col("tot_a").sum() / baseline["tot"].sum(),
            net_tone=(pl.col("pos_a").sum() - pl.col("neg_a").sum()) / pl.col("tot_a").sum(),
        )
        .sort("share", descending=True)
    )
    mo.vstack(
        [
            mo.md(
                "How much of the sentiment flow each event actually accounts for, all "
                f"{exposure.height} of them. `net_tone` is the event's own sentiment tilt — an event "
                "that is overwhelmingly positive or negative moves the statistic further per article."
            ),
            mo.ui.table(exposure, page_size=12),
        ]
    )
    return (exposure,)


@app.cell
def _(exposure):
    _e = exposure.sort("share")
    _fig = go.Figure(
        go.Bar(
            x=_e["share"],
            y=_e["event_name"],
            orientation="h",
            marker_color=[COLORS["sentiment"] if n >= 0 else COLORS["neg"] for n in _e["net_tone"]],
            customdata=np.stack([_e["company_days"], _e["net_tone"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>share %{x:.3%}<br>%{customdata[0]:,} company-days"
            "<br>net tone %{customdata[1]:+.3f}<extra></extra>",
        )
    )
    _fig.update_layout(
        template="plotly_white",
        title=f"Share of all sentiment matches, all {_e.height} events"
        "<br><sub>bar colour = the event's own net tone · scroll: the tail is genuinely this long</sub>",
        xaxis_title="share of sentiment matches (log)",
        xaxis_type="log",
        height=max(500, 15 * _e.height),
        margin={"l": 250, "r": 30, "t": 70, "b": 40},
    )
    _fig
    return


@app.cell
def _():
    mo.md("""
    ## 2 Baseline IC
    """)
    return


@app.cell
def _(arm, baseline, bridge, prices):
    _ic = load_ic(arm.value)
    horizons = sorted(_ic["h"].unique().to_list())
    _rows = [
        {
            "h": h,
            "mean_ic": _ic.filter((pl.col("event") == BASELINE) & (pl.col("h") == h))["ic"].mean(),
            "ic_vol": _ic.filter((pl.col("event") == BASELINE) & (pl.col("h") == h))["ic"].std(),
            "n_dates": _ic.filter((pl.col("event") == BASELINE) & (pl.col("h") == h)).height,
        }
        for h in horizons
    ]
    _wide = statistic.align(statistic.to_wide(statistic.baseline_statistic(baseline), bridge), prices)
    _names = [c for c in _wide.columns if c != "date"]
    _cov = _wide.select(pl.sum_horizontal(pl.col(c).is_not_null().cast(pl.Int32) for c in _names).alias("n"))
    _live = _cov.filter(pl.col("n") > 0)["n"]
    mo.vstack(
        [
            mo.md(
                f"The unmodified statistic on the **{arm.selected_key}**. Median cross-section: "
                f"**{int(_live.median())}** names with a value per date "
                f"(p10 {int(_live.quantile(0.1))}, p90 {int(_live.quantile(0.9))}). "
                "The event deltas in section 3 are differences against this, computed date by date."
            ),
            mo.ui.table(pl.DataFrame(_rows), page_size=6),
        ]
    )
    return (horizons,)


@app.cell
def _(arm, horizons):
    _base = load_ic(arm.value).filter((pl.col("event") == BASELINE) & (pl.col("h") == horizons[0])).sort("date")
    _fig = go.Figure(
        go.Scatter(
            x=_base["date"],
            y=_base["ic"].rolling_mean(window_size=min(63, max(_base.height // 4, 2)), min_samples=2),
            mode="lines",
            line={"color": COLORS["sentiment"], "width": 2},
            name="rolling mean IC",
        )
    )
    _fig.add_hline(y=0, line_color=COLORS["zero"], line_width=1)
    _fig.update_layout(
        template="plotly_white",
        title=f"Baseline IC through time (rolling mean, h={horizons[0]})"
        "<br><sub>day-to-day IC swings by ±0.08; event effects are ~0.001, which is why the test is paired</sub>",
        yaxis_title="IC",
        height=380,
    )
    _fig
    return


@app.cell
def _():
    mo.md("""
    ## 3 The full ranking
    """)
    return


@app.cell
def _(arm, horizons):
    summary = load_summary(arm.value)
    horizon = mo.ui.dropdown(
        options={str(h): h for h in horizons}, value=str(horizons[0]), label="forward horizon (days)"
    )
    rare = mo.ui.slider(
        start=0,
        stop=int(summary["company_days"].max()),
        step=250,
        value=min(5_000, int(summary["company_days"].max())),
        label="highlight threshold: grey events seen on fewer company-days than",
        show_value=True,
    )
    mo.hstack([horizon, rare], justify="start", gap=2)
    return horizon, rare, summary


@app.cell
def _(arm, horizon, rare, summary):
    _s = summary.filter(pl.col("h") == horizon.value).sort("delta_mean_ic", descending=True)
    _n_sig = _s.filter(pl.col("q") < 0.10).height
    mo.vstack(
        [
            mo.md(
                f"All **{_s.height}** events at h={horizon.value} on the **{arm.selected_key}**, "
                f"sortable and searchable. **{_n_sig}** clear q<0.10. "
                f"Positive `delta_mean_ic` means removing the event *improved* IC. Events below "
                f"{rare.value:,} company-days are greyed in the charts as too rare to estimate — but "
                f"they stay in this table, and in every count above."
            ),
            mo.ui.table(
                _s.select(
                    "event",
                    "delta_mean_ic",
                    "t_stat",
                    "q",
                    "p",
                    "mean_ic_base",
                    "mean_ic_var",
                    "company_days",
                    "match_share",
                    "n_dates",
                ),
                page_size=15,
            ),
        ]
    )
    return


@app.cell
def _(arm, horizon, rare, summary):
    _s = summary.filter(pl.col("h") == horizon.value).sort("delta_mean_ic")
    _grey = _s["company_days"] < rare.value
    _fig = go.Figure(
        go.Bar(
            x=_s["delta_mean_ic"],
            y=_s["event"],
            orientation="h",
            error_x={"type": "data", "array": 1.96 * _s["nw_se"], "color": COLORS["zero"], "thickness": 1},
            marker_color=[
                COLORS["zero"] if g else (COLORS["blend"] if d > 0 else COLORS["neg"])
                for g, d in zip(_grey, _s["delta_mean_ic"], strict=True)
            ],
            customdata=np.stack([_s["q"], _s["company_days"], _s["match_share"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>delta IC %{x:+.5f}<br>q %{customdata[0]:.3f}"
            "<br>%{customdata[1]:,} company-days<br>share %{customdata[2]:.3%}<extra></extra>",
        )
    )
    _fig.add_vline(x=0, line_color=COLORS["zero"], line_width=1)
    _fig.update_layout(
        template="plotly_white",
        title=(
            f"Change in mean IC from removing each event — all {_s.height} events "
            f"(h={horizon.value}, {arm.selected_key}, 95% Newey-West CI)"
            "<br><sub>green: removal helped · red: removal hurt · grey: below the rarity threshold</sub>"
        ),
        xaxis_title="delta mean IC",
        height=max(600, 16 * _s.height),
        margin={"l": 260, "r": 30, "t": 80, "b": 40},
    )
    _fig
    return


@app.cell
def _(rare, summary):
    _w = summary.select("event", "h", "delta_mean_ic").pivot(on="h", index="event", values="delta_mean_ic")
    _hc = [c for c in _w.columns if c != "event"]
    _w = _w.sort(_hc[-1], descending=True)
    _meta = summary.group_by("event").agg(pl.col("company_days").first())
    _w = _w.join(_meta, on="event", how="left")
    _lab = [
        f"{e}{'' if cd >= rare.value else '  (rare)'}" for e, cd in zip(_w["event"], _w["company_days"], strict=True)
    ]
    _z = _w.select(_hc).to_numpy()
    _lim = float(np.nanpercentile(np.abs(_z), 98))
    _fig = go.Figure(
        go.Heatmap(
            z=_z,
            x=[f"h={c}" for c in _hc],
            y=_lab,
            colorscale=[[0, COLORS["neg"]], [0.5, "#ffffff"], [1, COLORS["blend"]]],
            zmid=0,
            zmin=-_lim,
            zmax=_lim,
            hovertemplate="<b>%{y}</b> %{x}<br>delta IC %{z:+.5f}<extra></extra>",
        )
    )
    _fig.update_layout(
        template="plotly_white",
        title=f"Delta mean IC by event and horizon — all {_w.height} events"
        "<br><sub>colour clipped at the 98th percentile so the tail does not wash out the middle</sub>",
        height=max(500, 15 * _w.height),
        margin={"l": 270, "r": 30, "t": 70, "b": 40},
    )
    _fig
    return


@app.cell
def _():
    mo.md("""
    ## 4 Per-event drill-down

    A mean delta of 0.001 can come from a steady edge or from two offsetting outliers. This is
    where you tell them apart: pick an event and watch how its advantage over the baseline
    actually accumulates. A real effect drifts in one direction; noise wanders and is decided by
    a handful of dates.
    """)
    return


@app.cell
def _(horizon, summary):
    _order = summary.filter(pl.col("h") == horizon.value).sort(pl.col("delta_mean_ic").abs(), descending=True)
    event = mo.ui.dropdown(
        options=_order["event"].to_list(), value=_order["event"][0], label="event (largest |delta| first)"
    )
    event
    return (event,)


@app.cell
def _(arm, event, horizon, summary):
    _ic = load_ic(arm.value).filter(pl.col("h") == horizon.value)
    _b = _ic.filter(pl.col("event") == BASELINE).select("date", base="ic")
    _v = _ic.filter(pl.col("event") == event.value).select("date", var="ic")
    _j = _b.join(_v, on="date", how="inner").drop_nulls().sort("date")
    _j = _j.with_columns((pl.col("var") - pl.col("base")).alias("d"))
    _j = _j.with_columns(pl.col("d").cum_sum().alias("cum"))
    _r = summary.filter((pl.col("event") == event.value) & (pl.col("h") == horizon.value)).to_dicts()[0]

    _fig = go.Figure(
        go.Scatter(
            x=_j["date"],
            y=_j["cum"],
            mode="lines",
            line={"color": COLORS["blend"] if _r["delta_mean_ic"] > 0 else COLORS["neg"], "width": 2},
        )
    )
    _fig.add_hline(y=0, line_color=COLORS["zero"], line_width=1)
    _fig.update_layout(
        template="plotly_white",
        title=(
            f"{event.value} — cumulative IC advantage over baseline (h={horizon.value})"
            f"<br><sub>mean delta {_r['delta_mean_ic']:+.5f} · t {_r['t_stat']:+.2f} · q {_r['q']:.3f} · "
            f"share {_r['match_share']:.3%} · {_r['company_days']:,} company-days · "
            f"{_r['n_dates']} dates</sub>"
        ),
        yaxis_title="cumulative (IC_variant - IC_baseline)",
        height=420,
    )
    _worst = _j.sort(pl.col("d").abs(), descending=True).head(3)
    mo.vstack(
        [
            _fig,
            mo.md(
                "**Concentration check.** The three dates contributing most to this event's delta: "
                + " · ".join(f"{d} ({x:+.4f})" for d, x in zip(_worst["date"], _worst["d"], strict=True))
                + f". Together they are {_worst['d'].abs().sum() / max(_j['d'].abs().sum(), 1e-12):.1%} of the "
                "total absolute movement — a high share means the mean is a few dates, not an edge."
            ),
        ]
    )
    return


@app.cell
def _():
    mo.md("""
    ## 5 Frequency versus effect

    The check that stops a spurious headline. Sorting 147 events by effect size puts the
    *noisiest* estimates at both ends, and rare events are the noisiest. A real effect sits away
    from zero **and** rests on enough company-days to be estimable, so the cloud should narrow
    towards zero as frequency rises. If the only large deltas are rare events, the honest
    conclusion is that no event measurably moves the signal.
    """)
    return


@app.cell
def _(horizon, rare, summary):
    _s = summary.filter(pl.col("h") == horizon.value)
    _fig = go.Figure(
        go.Scatter(
            x=_s["company_days"],
            y=_s["delta_mean_ic"],
            mode="markers",
            marker={
                "size": 6 + 24 * (_s["match_share"] / max(float(_s["match_share"].max()), 1e-9)) ** 0.5,
                "color": [COLORS["blend"] if q < 0.10 else COLORS["zero"] for q in _s["q"].fill_null(1.0)],
                "opacity": 0.75,
                "line": {"width": 0},
            },
            text=_s["event"],
            customdata=np.stack([_s["q"], _s["match_share"]], axis=-1),
            hovertemplate="<b>%{text}</b><br>%{x:,} company-days<br>delta IC %{y:+.5f}"
            "<br>q %{customdata[0]:.3f}<br>share %{customdata[1]:.3%}<extra></extra>",
        )
    )
    _fig.add_hline(y=0, line_color=COLORS["zero"], line_width=1)
    _fig.add_vline(x=rare.value, line_color=COLORS["zero"], line_width=1, line_dash="dot")
    _fig.update_layout(
        template="plotly_white",
        title=f"Effect size against observation frequency — all {_s.height} events (h={horizon.value})"
        "<br><sub>marker size = share of matches · green: q &lt; 0.10 · dotted: rarity threshold</sub>",
        xaxis_title="company-days the event was observed on (log)",
        xaxis_type="log",
        yaxis_title="delta mean IC",
        height=520,
    )
    _fig
    return


@app.cell
def _():
    mo.md("""
    ## 6 Cross-arm agreement

    An effect that only exists in one specification is a specification artifact. Each panel plots
    the selected arm's deltas against another arm's, over every event they share.

    The local-currency arm deserves a specific caveat: its prices are unconverted, and FX shifts
    each currency bloc's returns against the others every day. Measured, that leaves the *event
    ranking* almost intact at h=1 (rank correlation +0.94) but only +0.77 at h=21 — so a
    monthly-horizon effect visible **only** in that arm is more likely FX than signal.
    """)
    return


@app.cell
def _(arm, horizon, summary):
    _base = summary.filter(pl.col("h") == horizon.value).select("event", "delta_mean_ic", "company_days")
    _panels = []
    for _label, _tag in available_arms().items():
        if _tag == arm.value:
            continue
        _o = load_summary(_tag).filter(pl.col("h") == horizon.value)
        if _o.is_empty():
            continue
        _j = _base.join(_o.select("event", other="delta_mean_ic"), on="event", how="inner")
        if _j.height < 3:
            continue
        _rho = np.corrcoef(
            np.argsort(np.argsort(_j["delta_mean_ic"].to_numpy())),
            np.argsort(np.argsort(_j["other"].to_numpy())),
        )[0, 1]
        _lim = float(np.nanmax(np.abs(np.concatenate([_j["delta_mean_ic"], _j["other"]]))))
        _f = go.Figure(
            go.Scatter(
                x=_j["delta_mean_ic"],
                y=_j["other"],
                mode="markers",
                marker={"size": 7, "color": COLORS["sentiment"], "opacity": 0.7},
                text=_j["event"],
                hovertemplate="<b>%{text}</b><br>%{x:+.5f} vs %{y:+.5f}<extra></extra>",
            )
        )
        _f.add_shape(type="line", x0=-_lim, y0=-_lim, x1=_lim, y1=_lim, line={"color": COLORS["zero"], "width": 1})
        _f.update_layout(
            template="plotly_white",
            title=f"vs {_label}<br><sub>rank corr {_rho:+.2f} over {_j.height} events</sub>",
            xaxis_title=f"{arm.selected_key} delta",
            yaxis_title=f"{_label} delta",
            height=400,
            width=420,
        )
        _panels.append(_f)
    mo.hstack(_panels, wrap=True) if _panels else mo.md("_Only one arm on disk — nothing to compare._")
    return


if __name__ == "__main__":
    app.run()
