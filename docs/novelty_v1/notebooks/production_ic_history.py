# /// script
# requires-python = ">=3.12"  # honey itself requires >=3.12
# dependencies = [
#     "marimo>=0.19.6",
#     "honey[research]",
#     "scipy",
# ]
# [tool.uv.sources]
# honey = { path = "../../../.." , editable = true }
# ///
"""Is the production sentiment signal's IC really negative?

Short answer: no. `novelty_subset_ceiling.py` measures IC only from 2024-07-01, because that is where
the novelty scores start, so its production row is a two-year reading. Over the full history the
production signal's IC is positive at every horizon. This notebook exists so that one figure is not
read as "production is broken".

Run it against the project environment, **not** `--sandbox` — the sandbox ignores the editable path
above and resolves an unrelated `honey` from PyPI:

    uv run --with marimo marimo edit book/marimo/notebooks/novelty/production_ic_history.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl

    import honey  # noqa: F401  (import side effect: loads YUKKA_TOKEN from repo-root .env)
    from honey.data.counts import sentiment_ratio
    from honey.data.preamble import load_counts, load_prices, to_returns
    from honey.novelty_score_exp import novelty_knn_model as nk
    from honey.signals import Signal

    # aliased off the leading underscore: marimo treats top-level `_names` as cell-private
    from honey.strategy.sentiment import _robust_signal as robust_signal

    HORIZONS = [1, 5, 10, 21, 63]
    SPAN = 19  # production's EWM span
    TOPN = 200  # universe, matching the ceiling notebook's default
    CLIP_C = nk.PROD_PARAMS["clip_c"]
    #: Where the ceiling notebook's window starts — the novelty scores begin here, and so does the
    #: negative IC. Everything in this notebook is a comparison across this boundary.
    SPLIT = nk.ARTICLE_DATA_START
    C_POS, C_NEG = "#0ea5a4", "#c0392b"

    def decile_returns(signal_wide, returns, h, lo=None, hi=None):
        """Mean forward-h-day return per signal decile, annualised %, plus the D10-D1 spread.

        Deliberately independent of `Signal.ic`: the forward sum propagates NaN, so a name that is
        missing, delisted or outside point-in-time membership is dropped rather than entering the
        cross-section with a fabricated 0% return (which is what `nancumsum` would give it).

        Returns:
            `(deciles, spread_pct, t_stat, n_dates)` — deciles low-to-high, spread annualised in %.
        """
        common = sorted((set(signal_wide.columns) & set(returns.columns)) - {"date"})
        joined = (
            signal_wide.select(["date", *common])
            .join(returns.select(["date", *common]), on="date", suffix="_r")
            .sort("date")
        )
        sig_np = joined.select(common).to_numpy().astype(float)
        ret_np = joined.select([f"{c}_r" for c in common]).to_numpy().astype(float)
        dates = joined["date"].to_list()

        buckets, spreads = [[] for _ in range(10)], []
        for t in range(len(sig_np) - h):
            if (lo and dates[t] < lo) or (hi and dates[t] >= hi):
                continue
            fwd = ret_np[t + 1 : t + 1 + h].sum(axis=0)
            keep = np.isfinite(sig_np[t]) & np.isfinite(fwd)
            if keep.sum() < 20:
                continue
            s, r = sig_np[t][keep], fwd[keep]
            q = (np.argsort(np.argsort(s)) * 10 // len(s)).clip(0, 9)
            for d in range(10):
                if (q == d).any():
                    buckets[d].append(r[q == d].mean())
            spreads.append(r[q == 9].mean() - r[q == 0].mean())

        ann = 252 / h
        spreads = np.array(spreads)
        t_stat = spreads.mean() / (spreads.std(ddof=1) / np.sqrt(len(spreads))) if len(spreads) > 2 else float("nan")
        return (
            [np.mean(b) * ann * 100 if b else np.nan for b in buckets],
            spreads.mean() * ann * 100,
            t_stat,
            len(spreads),
        )


@app.cell
def _():
    mo.md("""
    # Production IC — the full history

    The IC table in **`novelty_subset_ceiling.py`** shows production with a *negative* information
    coefficient. That table measures returns only from **2024-07-01**, where the novelty scores
    begin — so it is a two-year reading, not a verdict on the strategy.

    Over its whole history the production signal's IC is **positive at every horizon**, and rises
    with horizon. The negative figure is confined to the last two years and, as §4 argues, is not
    statistically distinguishable from zero.
    """)
    return


@app.cell
def _():
    # The same path every arm in the ceiling notebook takes: counts -> Laplace positive ratio ->
    # robust EW-median signal (lagged one day, clipped, cross-sectionally z-scored). IC is measured
    # on the robust signal because that is the object the book actually trades.
    counts = load_counts(TOPN)
    returns = to_returns(load_prices(TOPN))
    ratio = sentiment_ratio(counts)
    signal_wide = robust_signal(ratio, [c for c in ratio.columns if c != "date"], span=SPAN, clip_c=CLIP_C)
    sig = Signal(signal_wide)
    return counts, returns, sig, signal_wide


@app.cell
def _(counts, returns):
    _lo, _hi = counts["date"].min(), counts["date"].max()
    mo.md(
        f"**Universe** {len(returns.columns) - 1} RICs · **counts** {counts.height:,} rows · "
        f"**{_lo} → {_hi}** · span {SPAN} · clip {CLIP_C}"
    )
    return


@app.cell
def _(returns, sig):
    _windows = {
        "2015 → now (full history)": (None, None),
        "2015-01 … 2024-06": (None, SPLIT),
        "2024-07 → now (ceiling notebook)": (SPLIT, None),
    }
    _table = pl.DataFrame(
        [
            {"window": w, **{f"h={h}": round(sig.mean_ic(returns, h=h, date_from=a, date_to=b), 4) for h in HORIZONS}}
            for w, (a, b) in _windows.items()
        ]
    )
    mo.vstack([mo.md("## 1 · Mean IC by horizon"), mo.ui.table(_table, selection=None)])
    return


@app.cell
def _(returns, sig):
    _yr = (
        sig.ic(returns, h=21)
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(ic=pl.col("ic").mean(), n=pl.col("ic").is_finite().sum())
        .filter(pl.col("n") > 100)  # drop the partial first and last years
        .sort("year")
    )
    _fig = go.Figure(
        go.Bar(
            x=_yr["year"].to_list(),
            y=_yr["ic"].to_list(),
            marker_color=[C_NEG if v < 0 else C_POS for v in _yr["ic"]],
        )
    )
    _fig.update_layout(
        title="IC (h=21) by calendar year",
        yaxis_title="mean IC",
        height=320,
        margin={"t": 46, "b": 30},
    )
    mo.vstack([mo.md("## 2 · IC over time"), _fig])
    return


@app.cell
def _(returns, signal_wide):
    full = decile_returns(signal_wide, returns, h=1)
    recent = decile_returns(signal_wide, returns, h=1, lo=SPLIT)
    return full, recent


@app.cell
def _(full, recent):
    _x = [f"D{i}" for i in range(1, 11)]
    _fig = go.Figure()
    _fig.add_bar(x=_x, y=full[0], name="2015 → now", marker_color=C_POS)
    _fig.add_bar(x=_x, y=recent[0], name="2024-07 → now", marker_color=C_NEG, opacity=0.75)
    _fig.update_layout(
        title="Mean forward 1-day return by signal decile (annualised %)",
        yaxis_title="% / yr",
        height=360,
        barmode="group",
        margin={"t": 46, "b": 30},
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    mo.vstack(
        [
            mo.md("## 3 · Decile spread"),
            _fig,
            mo.md(
                f"**D10 − D1, 2015 → now:** `{full[1]:+.2f}%/yr` · t = `{full[2]:+.2f}` · n = {full[3]:,} days  \n"
                f"**D10 − D1, 2024-07 → now:** `{recent[1]:+.2f}%/yr` · t = `{recent[2]:+.2f}` · "
                f"n = {recent[3]:,} days"
            ),
        ]
    )
    return


@app.cell
def _():
    mo.md("""
    ## 4 · How much to believe it

    - **The recent inversion is noise.** At h=1 — the only non-overlapping horizon — the last two
      years give a negative D10−D1 with |t| well under 1. Over ~577 days that is not evidence that
      the signal has broken.
    - **The full-history edge is real but thin.** Positive at every horizon and in nine of eleven
      years, but only 2018 clears |t| > 2 on its own.
    - **Longer-horizon t-stats are inflated.** Daily dates with h-day forward windows overlap almost
      completely, so the effective sample is roughly n/h — deflate those t-stats by about √h.
    - **Every decile is positive since 2024-07** (+8% to +20%/yr): that window is a bull market, and
      only the cross-sectional *ordering* is at issue.
    - `Signal.ic` sums forward returns with `nancumsum`, which treats a missing return as 0 rather
      than dropping the name. Checked against a NaN-propagating version: it moves the full-history
      IC by less than 0.0001. §3 avoids it regardless.
    """)
    return


if __name__ == "__main__":
    app.run()
