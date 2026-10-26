# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.19.6",
#     "honey[research]",
#     "optuna>=3.6",
# ]
# [tool.uv.sources]
# honey = { path = "../../../.." , editable = true }
# ///
"""Does weighting sentiment by article novelty beat the production sentiment strategy? — ceiling test.

A discovery notebook: sweep the embedding-novelty grid broadly and see whether *any* setting has
headroom over the production signal, before spending effort on one candidate.

    · PRODUCTION (baseline)  — Yukka's t-stat >= 2.5 counts (`load_counts`). THE BAR TO BEAT.
    · Novel (kNN, W, n)      — one arm per (lookback window, n) cell of the novelty grid.
    · STOXX 600              — the index, as background context only.

Each kNN arm scores every article for novelty (1 - similarity to the nearest earlier headlines of the
same company) and lets that score enter the daily sentiment count in one of two ways, chosen by a
toggle in section 1:

    · threshold — `nk.threshold_long`: keep articles above theta, count survivors
    · weight    — `nk.weighted_long`:  every article contributes its own novelty, no theta

Every arm, production included, is then built by `nk.build_arm`, which pins honey's sentiment
`build_portfolio` to `nk.PROD_PARAMS` — the production leg parameter for parameter. Only the counts
differ, so a difference in outcome is attributable to the signal rather than to the book.

Sections: 1 data & universe · 2 signals and IC · 3 backtest · 4 optimising theta · 5 what the swept
axis does firm by firm · 6 how many articles reach the count · 7 the whole grid as a surface.

Thin by design: universe / PIT / prices / returns come from `honey.data.preamble`, the sentiment ratio
from `honey.data.counts`, the robust signal transform and backtest from `honey.strategy.sentiment`, and
IC / plots from `honey.signals.Signal`. Only `novelty_knn_model.py` is experiment-specific.

Strategy background and parameter definitions are in `book/marimo/notebooks/NOVELTY.md`.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import optuna
    import plotly.graph_objects as go
    import polars as pl

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep the trial log out of the notebook
    from jquantstats import Portfolio
    from yukka.data import Index

    import honey  # noqa: F401  (import side effect: loads YUKKA_TOKEN from repo-root .env)
    from honey.data import YukkaRepository
    from honey.data.counts import sentiment_ratio
    from honey.data.preamble import load_benchmark, load_counts, load_prices, load_returns, load_sector_mask
    from honey.novelty_score_exp import novelty_knn_model as nk
    from honey.signals import Signal

    # aliased off the leading underscore: marimo treats top-level `_names` as cell-private
    from honey.strategy.sentiment import _robust_signal as robust_signal

    HORIZONS = [1, 5, 10, 15, 21, 42, 63]
    #: Every arm — production and novelty alike — is built by `nk.build_arm`, which is the *production
    #: sentiment leg* parameter for parameter (`nk.PROD_PARAMS`: threshold 1.15 on the robust z, 10%
    #: max weight, 10-day rebalance, 20% vol target). Only the counts feeding it differ, so a Sharpe
    #: difference between arms can only come from the signal, never from the book.
    CLIP_C = nk.PROD_PARAMS["clip_c"]
    #: Overrides layered on `PROD_PARAMS` for *every* arm at once. Empty = exactly production.
    #: `{"rebalance_freq": 1}` reverts to the daily book used before this notebook was rewired: a
    #: multi-day basket makes Sharpe depend on which day of the cycle the panel starts on (production
    #: swung 1.11-1.42 across the five phases of a 5-day schedule), so daily is the reproducible
    #: comparison while 10 is what actually trades. Both are legitimate; they answer different questions.
    BOOK: dict[str, object] = {}
    SECTOR_CAP = None  # production runs uncapped; set a fraction to cap Financials in every arm
    # kNN embedding novelty (novelty_knn_model.py): per-article novelty = 1 - mean cos-sim to the n nearest
    # earlier titles inside a lookback window. The notebook only *reads* the scores.
    #
    # The grid is (window x n): one parquet per window, one `novelty_n{n}` column per n inside it, all
    # with `max_cand=0` so the window — not a candidate cap — is the binding lookback. The full product
    # is far too many curves for one chart, so a run fixes one axis and overlays the other.
    C_KNN = "#0ea5a4"
    C_BENCH = "#1f9d57"
    PROD_LABEL = "Production (BASELINE, t-stat >= 2.5)"
    #: Teal ramp, light -> dark, walked in whichever order the overlaid axis is sorted.
    KNN_RAMP = ["#bfe4e3", "#8ed0cf", "#5cbcba", "#2ba3a1", "#0b7a79", "#08514f"]

    def knn_manifest():
        """Every built scores parquet as `(window_label, window_hours, path, KNNParams)`, shortest window first.

        Discovered from disk and self-describing: each file carries the `KNNParams` it was written with,
        so the window and the available n's come from the data rather than a hardcoded list here.
        """
        out = []
        for path in Path(nk.EMB).glob("novelty_scores_w*h.parquet"):
            p = nk.read_params(path)
            if p is not None:
                out.append((p.window_label, p.window_hours, path, p))
        return sorted(out, key=lambda r: r[1])

    def knn_label(window_label, n):
        """Arm label for one (window, n) cell of the grid."""
        return f"Novel (kNN, {window_label}, n={n})"

    def parse_arm(label):
        """Inverse of `knn_label` — recover `(window_label, n)` from an arm name."""
        inner = label[len("Novel (kNN, ") : -1]
        w, n = inner.split(", n=")
        return w, int(n)

    def knn_ramp(i, total):
        """Colour for the i-th overlaid arm — interpolated along the teal ramp.

        Interpolating rather than indexing keeps the arms visually distinct for any slice size: the
        ramp has six stops, so a nine-value n grid would otherwise reuse three of them.
        """
        if total <= 1:
            return KNN_RAMP[-1]
        x = i / (total - 1) * (len(KNN_RAMP) - 1)
        lo = min(int(x), len(KNN_RAMP) - 2)
        t = x - lo
        a, b = KNN_RAMP[lo].lstrip("#"), KNN_RAMP[lo + 1].lstrip("#")
        rgb = (round(int(a[k : k + 2], 16) * (1 - t) + int(b[k : k + 2], 16) * t) for k in (0, 2, 4))
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    #: Production is the only non-kNN arm; every other arm is one cell of the embedding-novelty grid,
    #: drawn light-to-dark along the teal ramp in whichever axis section 1 is sweeping.
    PROD_STYLE = ("#111111", "dash")

    def knn_scores(path, n, id_map, universe):
        """One (window, n) cell: per-article scores mapped to RIC, universe-filtered, novelty column renamed.

        Only the one `novelty_n{n}` column is read off disk — the files carry one per n — and it comes
        back as plain `novelty`, so everything downstream stays agnostic to which cell of the grid it holds.
        """
        col = nk.novelty_col(n)
        return (
            pl.read_parquet(path, columns=["entity", "date", "sentiment", col])
            .rename({col: "novelty"})
            # yukka_id -> RIC; clip to [0, 1] because float32 cosine on a near-identical reprint lands
            # just above 1.0 and makes novelty negative (28.6% of articles) — see `nk.CLIP_NOVELTY`.
            .with_columns(pl.col("entity").replace(id_map), nk.CLIP_NOVELTY)
            .filter(pl.col("entity").is_in(universe))  # same cross-section as the other arms
        )

    def knn_slice(overlay, fixed, id_map, universe):
        """The selected row/column of the grid as `{arm_label: scores}`.

        `overlay="window"` fixes n=`fixed` and walks every built window; `overlay="n"` fixes the window
        and walks every n in that file. Either way the result is one arm per curve.
        """
        rows = knn_manifest()
        if not rows:
            return {}
        if overlay == "window":
            return {knn_label(w, fixed): knn_scores(path, fixed, id_map, universe) for w, _h, path, _p in rows}
        w, _h, path, p = next((r for r in rows if r[0] == fixed), rows[-1])
        return {knn_label(w, n): knn_scores(path, n, id_map, universe) for n in sorted(p.n_values)}

    def arm_signal(counts_long, span):
        """(ratio_wide, Signal) for one arm: sentiment_ratio -> robust signal (honey), wrapped in Signal."""
        ratio = sentiment_ratio(counts_long)
        cols = [c for c in ratio.columns if c != "date"]
        return ratio, Signal(robust_signal(ratio, cols, span=span, clip_c=CLIP_C))

    def build_arm(prices, ratio, span, mask=None):
        """Backtest one arm through the **production sentiment book** (`nk.build_arm`).

        The only knob a call site gets is `span`, because that is the one parameter this experiment
        sweeps. Everything else — threshold, max weight, rebalance frequency, vol target — comes from
        `nk.PROD_PARAMS`, patched globally by `BOOK`. Production is built by this same function with
        its own counts, so no arm can accidentally be scored in a different book from its baseline.
        """
        return nk.build_arm(prices, ratio, sector_mask=mask, sector_cap=SECTOR_CAP, span=span, **BOOK)

    def benchmark_portfolio(aum=1e8):
        """Real STOXX 600 index as a fully-invested one-asset Portfolio (background context)."""
        b = load_benchmark("STOXX 600").rename({"close": "STOXX 600"})
        cash = b.select("date").with_columns(pl.lit(aum).alias("STOXX 600"))
        return Portfolio(cashposition=cash, prices=b, aum=aum)

    def first_active(pf):
        """First date a portfolio actually trades (nonzero return) — strategies warm up, the index doesn't."""
        r = pf.returns.filter(pl.col("returns").fill_null(0.0) != 0.0)
        return r["date"].min() if r.height else pf.returns["date"].min()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Novelty subset ceiling test — can we beat production?

    Production's sentiment signal counts every article about a company the same. An event that is
    reprinted, syndicated and translated fifty times therefore weighs fifty times as much as one that
    was reported once. This notebook asks whether discounting that repetition helps: every article is
    scored for **novelty** — `1 − mean cosine similarity to the n nearest earlier headlines about the
    same company inside a lookback window` — and that score, rather than a flat count of 1, drives the
    daily sentiment count.

    Three kinds of line appear below:

    - **Production (BASELINE, t-stat ≥ 2.5)** — Yukka's production counts via `load_counts`, drawn as
      the black dashed curve. **The bar to beat.**
    - **Novel (kNN, window, n)** — one arm per cell of the novelty grid, light → dark teal. §1 chooses
      which row or column of the (window × n) grid is overlaid, and whether novelty enters as a
      **filter** (keep only articles scoring above θ) or as a **weight** (keep everything, each article
      contributing its own novelty).
    - **STOXX 600** — the index, background context, not a target.

    Every arm, production included, is backtested through the *same* production sentiment book, so only
    the counts differ and any Sharpe gap is attributable to the signal rather than to the portfolio.
    This is a **ceiling test**: θ and the grid cell are chosen in-sample, so the numbers answer "how far
    could this idea go at best", not "what should we trade".
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1 · Data & universe

    The controls here drive every section below. **Universe top-N** and **span** are honey's own
    strategy parameters (production runs 200 and 19); **θ** is the novelty cutoff, used wherever novelty
    enters as a filter; **overlay** picks which axis of the (window × n) grid becomes the set of curves
    while the other axis stays fixed — the full product is far more arms than one chart can carry.
    Nothing is loaded until **Load & compare** is pressed.

    The two callouts that follow report the universe and the date coverage actually loaded. Everything
    is scored from **2024-07-01**, where the article data starts and therefore where the novelty arms
    can first trade; production's Sharpe over its *full* history is quoted alongside as a reminder that
    this window is a markedly better-than-average stretch of market.
    """)
    return


@app.cell
def _():
    _rows = knn_manifest()
    _windows = [w for w, _h, _p, _q in _rows] or ["—"]
    _ns = sorted(_rows[-1][3].n_values) if _rows else [5]
    top_n_ui = mo.ui.slider(20, 400, step=20, value=200, label="Universe top-N")
    span_ui = mo.ui.slider(1, 42, step=1, value=19, label="Robust-signal span (EWM)")
    theta_ui = mo.ui.slider(0.0, 0.6, step=0.02, value=0.30, label="kNN novelty θ (novel = novelty > θ)")
    # how the novelty score enters the daily counts: a hard threshold, or a per-article weight
    use_ui = mo.ui.radio(["filter (θ)", "weight (no θ)"], value="filter (θ)", label="Novelty enters as", inline=True)
    # the (window x n) grid is too big to draw at once: fix one axis, overlay the other
    overlay_ui = mo.ui.radio(["window", "n"], value="window", label="Overlay kNN arms across", inline=True)
    fix_n_ui = mo.ui.dropdown([str(n) for n in _ns], value=str(_ns[0]), label="…at n =")
    fix_w_ui = mo.ui.dropdown(_windows, value=_windows[-1], label="…at window =")
    load_button = mo.ui.run_button(label="Load & compare")
    mo.vstack(
        [
            mo.hstack([top_n_ui, span_ui, theta_ui], justify="start"),
            mo.hstack([overlay_ui, fix_n_ui, fix_w_ui], justify="start"),
            mo.hstack([use_ui], justify="start"),
            mo.md(f"*{len(_windows)} windows x {len(_ns)} n built — only the fixed axis' selector applies.*"),
            load_button,
        ]
    )
    return (
        fix_n_ui,
        fix_w_ui,
        load_button,
        overlay_ui,
        span_ui,
        theta_ui,
        top_n_ui,
        use_ui,
    )


@app.cell
def _(fix_n_ui, fix_w_ui, load_button, overlay_ui, span_ui, top_n_ui):
    mo.stop(not load_button.value, mo.callout(mo.md("Press **Load & compare**."), kind="info"))

    # honey handles universe / PIT / prices / returns.
    #
    # Prices are cut to ARTICLE_DATA_START because the arms have very different signal histories:
    # production counts reach back to 2015, kNN novelty only to 2024-07. On the full panel the kNN
    # arms hold no position for ~83% of the dates and those zero-return days land in their Sharpe
    # denominator (~0.4x), so an untruncated comparison scores production over eleven years against
    # novelty over two. §3 hides this via `first_active`, but §7 reads the panel directly.
    prices_full = load_prices(top_n_ui.value)
    prices = prices_full.filter((pl.col("date") >= nk.ARTICLE_DATA_START) & (pl.col("date") < nk.ARTICLE_DATA_END))
    returns = load_returns(top_n_ui.value).filter(pl.col("date").is_between(nk.ARTICLE_DATA_START, nk.ARTICLE_DATA_END))
    prod_long = load_counts(top_n_ui.value)  # production t-stat >= 2.5 counts, RIC-keyed, PIT-masked

    # yukka_id -> RIC, for mapping the scores files onto the price panel's identifiers.
    # Use the repository's assets (deduped by RIC), NOT Index.STOXX600.assets — the SDK dedups
    # constituents by ISIN and 844/1066 rows have a null ISIN, collapsing the list to ~200 stale
    # delisted names that resolve almost none of the live universe (see YukkaRepository.assets).
    _assets = YukkaRepository(index=Index.STOXX600).assets
    yukka_to_ric = {a.yukka_id: a.ric for a in _assets if a.yukka_id}
    universe = [c for c in prices.columns if c != "date"]

    _overlay = overlay_ui.value
    _fixed = int(fix_n_ui.value) if _overlay == "window" else fix_w_ui.value
    # The scores parquets are self-contained, so nothing here reads the article dump any more.
    with mo.status.spinner(title="Loading kNN novelty scores…"):
        knn_sc = knn_slice(_overlay, _fixed, yukka_to_ric, universe)  # {arm: scores}; {} if none built

    # Point-in-time Financials membership on the price spine — the `sector_mask` production's own
    # build passes. Inert while SECTOR_CAP is None, but every arm gets it, so turning the cap on
    # applies to production and novelty together.
    sector_mask = load_sector_mask(prices_full)

    # Production over its whole history — the article window is only the recent slice of it, and that
    # slice happens to be a good one, so keep the long-run number visible as the honest reference.
    _pf_full = build_arm(prices_full, sentiment_ratio(prod_long), span_ui.value, sector_mask)
    prod_longrun = nk.window_sharpe(_pf_full, prices_full["date"].min(), prices_full["date"].max())

    _knn_note = (
        f" · kNN: **{len(knn_sc)}** arms overlaying **{_overlay}** at fixed **{_fixed}**"
        if knn_sc
        else " · kNN arms **off** (run novelty_knn_model.py)"
    )
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"Universe **{len(universe)}** RICs · "
                    f"returns from **{returns['date'].min()}** · production to "
                    f"**{prod_long['date'].max()}**{_knn_note}."
                ),
                kind="success",
            ),
            mo.callout(
                mo.md(
                    f"Everything below is scored on **{prices['date'].min()} → {prices['date'].max()}** "
                    f"({prices.height} trading days), the window every arm can trade. For scale: production "
                    f"over its full history (**{prices_full['date'].min()} → {prices_full['date'].max()}**, "
                    f"{prices_full.height} days) earns Sharpe **{prod_longrun:.2f}** — the article window is a "
                    f"markedly better-than-average stretch, so treat every Sharpe here as a recent-window "
                    f"number, not a long-run one."
                ),
                kind="info",
            ),
        ]
    )
    return (
        knn_sc,
        prices,
        prod_long,
        returns,
        sector_mask,
        universe,
        yukka_to_ric,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2 · Signals & information coefficient

    Every arm takes the same path: daily counts → `sentiment_ratio` (Laplace-smoothed positive share) →
    `_robust_signal` (EW-median location and scale, lagged one day so nothing is forward-looking,
    clipped at ±4.2, EW-smoothed over `span`, then z-scored across the cross-section) → `Signal`. Only
    the counts entering that path differ between arms.

    The table is the mean information coefficient — rank correlation between the signal and the next
    *h* trading days of PIT-masked returns — at each horizon, one row per arm. IC is measured on the
    *robust* signal, the one that actually gets traded, so §2 and §3 describe the same object.
    **A novelty arm is only promising where its IC exceeds production's row.** The two charts underneath
    are the IC-decay curves for production and for the last novelty arm of the slice, side by side, so
    the shape of the decay can be compared and not only its level.
    """)
    return


@app.cell
def _(knn_sc, prod_long, span_ui, theta_ui, use_ui):
    _span = span_ui.value
    arms = {PROD_LABEL: arm_signal(prod_long, _span)}
    # one kNN embedding-novelty arm per cell of the selected grid slice. Either the score
    # is thresholded at the live theta, or it weights each article directly (§1 toggle) — see
    # `nk.weighted_long`. Weighting has no theta, so the label drops it.
    _weighted = use_ui.value.startswith("weight")
    for _arm, _scores in knn_sc.items():
        _long = nk.weighted_long(_scores) if _weighted else nk.threshold_long(_scores, theta_ui.value)
        arms[_arm + (" [weighted]" if _weighted else "")] = arm_signal(_long, _span)
    return (arms,)


@app.cell
def _(arms, returns):
    _rows = []
    for _label, (_ratio, _sig) in arms.items():
        _rows.append({"signal": _label, **{f"h={h}": round(_sig.mean_ic(returns, h=h), 4) for h in HORIZONS}})
    mo.ui.table(pl.DataFrame(_rows), selection=None)
    return


@app.cell
def _(arms, returns):
    # IC decay, production against the headline novelty arm (the last kNN cell built). The key is read
    # off `arms` rather than `knn_sc` because weighted mode appends a "[weighted]" suffix to the label.
    # With no scores files there is no novelty arm to compare, so production is shown alone.
    _knn = [k for k in arms if k.startswith("Novel (kNN")]
    _figs = []
    for _label in [PROD_LABEL, *_knn[-1:]]:
        _f = arms[_label][1].plots.ic_decay(returns, horizons=HORIZONS)
        _f.update_layout(title=f"IC decay — {_label}", height=320)
        _figs.append(_f)
    mo.hstack(_figs, widths=[1] * len(_figs))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3 · Backtest — Sharpe, CAGR, cumulative return

    Every arm from §2 — **Production** plus one **kNN** arm per cell of the slice §1 selected — and the
    **STOXX 600** benchmark. Every arm runs through **the production sentiment book**: `nk.build_arm`,
    which is `sent_params` from `sentiment_momentum_top200.py` parameter for parameter — hold each name
    whose robust cross-sectional z exceeds **1.15**, equal-weight capped at **10%** per name with the
    residual in cash, rebalanced every **10** days, then scaled to a **20%** vol target. Only the counts
    differ between arms, so a Sharpe gap can only come from the signal. Everything is scored over one
    **common window** (via `Portfolio.truncate`) so no arm gets a head start; Sharpe and CAGR come from
    `Portfolio.stats`.

    The table under the chart reports each arm on two windows: *Sharpe (vs benchmark)* over the days the
    benchmark file also covers — the window the chart draws — and *Sharpe (arms' window)*, which runs to
    the end of the price panel and is the column the novelty-vs-production question turns on. The last
    column is that arm's gap to production.

    The kNN curves (light → dark teal) differ in **one** parameter — whichever axis §1 is sweeping.
    Every scores file was built with `max_cand=0`, so the window is the true lookback for every firm,
    not a candidate cap that silently bites hardest on the busiest names.

    Read the spread between the curves as the value of that parameter. If they sit on top of each other,
    the parameter is not what limits the signal; switch the overlay in §1 and check the other axis
    before concluding the novelty idea itself is exhausted.
    """)
    return


@app.cell
def _(arms, prices, sector_mask, span_ui):
    _span = span_ui.value
    # every kNN arm the previous cell built, whatever it named them, plus production
    _shown = {k for k in arms if k.startswith("Novel (kNN")} | {PROD_LABEL}
    pfs = {}
    with mo.status.spinner(title="Backtesting the arms through the production book…"):
        for _label, (_ratio, _sig) in arms.items():
            if _label not in _shown:
                continue
            pfs[_label] = build_arm(prices, _ratio, _span, sector_mask)
    # TWO common windows, because the arms and the benchmark do not cover the same days.
    #
    # `window` — the arms' own: every competing arm comes out of the same pipeline on the same price
    # panel, so the window they agree on is the honest one for "does novelty beat production", and it
    # runs to the end of the price data.
    #
    # `bench_window` — that window intersected with the benchmark's coverage. The benchmark is a
    # hand-maintained static file (`cache/benchmark_stoxx600.parquet`) that nothing in the repo
    # refreshes, and it currently ends 17 trading days behind the panel. Letting it into the `min()`
    # above would silently amputate the tail of *every* backtest (it cost production 0.10 Sharpe), but
    # comparing an arm against it over different days is equally wrong — so the benchmark comparison
    # gets its own window, and every arm is reported on that window too. Same days, or no comparison.
    _start = max(first_active(pf) for pf in pfs.values())
    _end = min(pf.returns["date"].max() for pf in pfs.values())
    window = (_start, _end)

    pfs["STOXX 600 (benchmark)"] = benchmark_portfolio()
    bench_window = (_start, min(_end, pfs["STOXX 600 (benchmark)"].returns["date"].max()))
    portfolios = {k: v.truncate(*window) for k, v in pfs.items() if "benchmark" not in k}
    portfolios_bench = {k: v.truncate(*bench_window) for k, v in pfs.items()}
    return bench_window, portfolios, portfolios_bench, window


@app.cell
def _(bench_window, portfolios, portfolios_bench, window):
    _styles = {PROD_LABEL: PROD_STYLE}
    _knn = [k for k in portfolios if k.startswith("Novel (kNN")]
    for _i, _arm in enumerate(_knn):  # light -> dark teal along the overlaid axis
        _styles[_arm] = (knn_ramp(_i, len(_knn)), "solid")
    _styles["STOXX 600 (benchmark)"] = (C_BENCH, "solid")
    _prod_sh = portfolios[PROD_LABEL].stats.sharpe()["returns"]
    _same = window == bench_window  # benchmark covers the arms' window — one window suffices

    def _sharpe(pf):
        _s = pf.stats.sharpe()["returns"]
        return float("nan") if _s is None else float(_s)

    # The chart draws every series on the benchmark window, so the curves a reader compares by eye
    # start and end together — including the benchmark. The table adds the arms' own longer window,
    # which is the number the novelty-vs-production question is decided on.
    _fig = go.Figure()
    _rows = []
    for _label, _pf in portfolios_bench.items():
        _r = _pf.returns
        _nav = np.cumprod(1.0 + np.nan_to_num(_r["returns"].to_numpy())) - 1.0
        _color, _dash = _styles[_label]
        _fig.add_trace(
            go.Scatter(
                x=_r["date"].to_list(),
                y=_nav,
                mode="lines",
                name=_label,
                line={"color": _color, "dash": _dash, "width": 3 if "benchmark" in _label else 2},
            )
        )
        _arm_pf = portfolios.get(_label)
        _rows.append(
            {
                "signal": _label,
                "Sharpe (vs benchmark)": f"{_sharpe(_pf):.2f}",
                "CAGR": f"{_pf.stats.cagr()['returns']:.1%}",
                "cum. return": f"{_nav[-1]:.1%}",
                "Sharpe (arms' window)": "—" if _arm_pf is None else f"{_sharpe(_arm_pf):.2f}",
                "vs prod (Sharpe)": "—" if _arm_pf is None else f"{_sharpe(_arm_pf) - _prod_sh:+.2f}",
            }
        )
    _fig.update_layout(
        title=(
            f"Cumulative return, common window {bench_window[0]} … {bench_window[1]} (no head start)"
            + ("" if _same else f" · arms alone run to {window[1]}")
        ),
        yaxis={"title": "cumulative return", "tickformat": ".0%"},
        hovermode="x unified",
        height=440,
    )
    mo.vstack(
        [
            _fig,
            mo.ui.table(pl.DataFrame(_rows), selection=None),
            mo.md(
                f"*Every series above — arms and benchmark — is measured over the identical "
                f"**{bench_window[0]} … {bench_window[1]}**, so the benchmark comparison is like for like. "
                + (
                    "The benchmark covers the arms' full window, so the two Sharpe columns agree.*"
                    if _same
                    else f"The benchmark file stops at **{bench_window[1]}** while the arms trade to "
                    f"**{window[1]}**; *Sharpe (arms' window)* uses those extra days, which only the "
                    f"arms have, and is the column the novelty-vs-production question turns on.*"
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4 · Optimizing the kNN threshold θ

    θ only re-thresholds the *precomputed* per-article novelty scores (`novel = novelty > θ`), so
    searching it is cheap — a rebuild, never a re-scoring. One **Optuna** TPE study per arm maximises
    that arm's Sharpe over the **same common window** as §3, through the same production book
    (`nk.build_arm`). The scatter below shows every trial, a star marks each arm's best θ, and the black
    dashed line is production's Sharpe — the bar to beat.

    This section is only meaningful in **filter (θ)** mode. With novelty entering as a weight there is
    no threshold to optimise — §1's toggle removes the parameter rather than tuning it.

    **One study per arm in the slice**, each with its own θ\*. θ\* is *not* comparable across arms: a
    longer **window** pulls the nearest neighbours closer, so novelty shifts down and the same θ filters
    harder, while a larger **n** averages in weaker matches, so novelty shifts up and the same θ filters
    more loosely (§5 shows both moves firm by firm). What *is* comparable is each arm's best achievable
    Sharpe — that is the question "does moving this axis buy anything once θ is tuned for it".

    **In-sample.** θ is tuned on the whole evaluation window, so θ\* is an optimistic *ceiling*, not a
    validated parameter — read it as "how far can threshold tuning push the kNN arm", not a tradeable setting.
    Comparing per-arm ceilings is a fair comparison of ceilings, not of tradeable strategies.
    """)
    return


@app.cell
def _(knn_sc):
    n_trials_ui = mo.ui.slider(10, 200, step=10, value=40, label="Optuna trials (per arm)")
    optimize_button = mo.ui.run_button(label="Optimize θ (all arms)")
    _cost = f"{len(knn_sc)} studies" if knn_sc else "no arms built"
    mo.hstack([n_trials_ui, optimize_button, mo.md(f"*{_cost}*")], justify="start")
    return n_trials_ui, optimize_button


@app.cell
def _(
    knn_sc,
    n_trials_ui,
    optimize_button,
    prices,
    sector_mask,
    span_ui,
    window,
):
    mo.stop(
        not knn_sc,
        mo.callout(
            mo.md("kNN scores not built — run `novelty_knn_model.py` first to enable θ optimization."), kind="warn"
        ),
    )
    mo.stop(not optimize_button.value, mo.callout(mo.md("Press **Optimize θ** to run the search."), kind="info"))

    _span = span_ui.value

    def _sharpe_at(scores, theta):
        """Common-window Sharpe of the kNN arm built from `scores` at this θ (-1e9 if degenerate)."""
        _ratio, _ = arm_signal(nk.threshold_long(scores, theta), _span)
        _pf = build_arm(prices, _ratio, _span, sector_mask).truncate(window[0], window[1])
        _sh = _pf.stats.sharpe()["returns"]
        return -1e9 if (_sh is None or np.isnan(_sh)) else float(_sh)  # steer TPE away from degenerate θ

    # one independent TPE study per arm in the slice; same seed so the θ sequences are comparable
    studies = {}
    for _variant, _scores in knn_sc.items():
        _study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        with mo.status.spinner(title=f"Optuna [{_variant}]: {n_trials_ui.value} trials…"):
            _study.optimize(
                lambda t, _s=_scores: _sharpe_at(_s, t.suggest_float("theta", 0.0, 0.6)),
                n_trials=n_trials_ui.value,
            )
        studies[_variant] = _study
    return (studies,)


@app.cell
def _(portfolios, studies):
    _prod_sh = portfolios[PROD_LABEL].stats.sharpe()["returns"]

    _fig = go.Figure()
    _rows = []
    for _i, (_variant, _study) in enumerate(studies.items()):
        _c = knn_ramp(_i, len(studies))
        _t = pl.DataFrame(
            {"theta": [t.params["theta"] for t in _study.trials], "Sharpe": [t.value for t in _study.trials]}
        ).sort("theta")
        _bt, _bs = _study.best_params["theta"], _study.best_value
        _fig.add_trace(
            go.Scatter(
                x=_t["theta"],
                y=_t["Sharpe"],
                mode="markers",
                name=_variant,
                marker={"color": _c, "size": 6, "opacity": 0.55},
            )
        )
        _fig.add_trace(
            go.Scatter(
                x=[_bt],
                y=[_bs],
                mode="markers",
                name=f"θ* {_variant}",
                marker={"color": _c, "size": 14, "symbol": "star", "line": {"color": "#111111", "width": 1}},
            )
        )
        _rows.append(
            {
                "kNN arm": _variant,
                "θ*": f"{_bt:.3f}",
                "Sharpe @ θ*": f"{_bs:.2f}",
                "vs production": f"{_bs - _prod_sh:+.2f}",
                "beats production": "yes" if _bs > _prod_sh else "no",
            }
        )
    _fig.add_hline(y=_prod_sh, line={"color": "#111111", "dash": "dash"}, annotation_text="production Sharpe")
    _fig.update_layout(
        title=f"kNN Sharpe vs θ, one study per arm in the slice (production {_prod_sh:.2f})",
        xaxis={"title": "θ  (is_novel = novelty > θ)"},
        yaxis={"title": "Sharpe (common window)"},
        hovermode="closest",
        height=460,
    )

    _best_v, _best_st = max(studies.items(), key=lambda kv: kv[1].best_value)
    _verdict = "beats" if _best_st.best_value > _prod_sh else "does not beat"
    mo.vstack(
        [
            _fig,
            mo.ui.table(pl.DataFrame(_rows), selection=None),
            mo.callout(
                mo.md(
                    f"Best in this slice: **{_best_v}** at θ\\*={_best_st.best_params['theta']:.3f} → Sharpe "
                    f"**{_best_st.best_value:.2f}** vs production **{_prod_sh:.2f}** "
                    f"(Δ {_best_st.best_value - _prod_sh:+.2f}) — **{_verdict}** production. If the ceilings are "
                    f"flat across the swept axis, that axis was not what was limiting the signal."
                ),
                kind="success" if _best_st.best_value > _prod_sh else "neutral",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5 · What the swept axis actually does, firm by firm

    Both axes move novelty in a guaranteed direction, so the *size* of the move is the whole story:

    - **window** — a longer window only *adds* candidates, so the nearest neighbours can only get
      closer and **novelty is non-increasing in the window**.
    - **n** — averaging over more neighbours pulls in progressively weaker matches, so the mean
      similarity falls and **novelty is non-decreasing in n**.

    The point is that neither move is **uniform across the cross-section**. A quiet firm with three
    articles a day has almost nothing to add as the window grows, and runs out of neighbours long before
    n=100; a firm with three thousand a day moves on both. Since the strategy ranks firms against each
    other, a parameter that shifts some firms and not others is a systematic tilt, not just noise.

    Below: one line chart per selected firm, showing daily mean novelty at each end of the swept axis;
    a table sampling the same numbers date by date with their **spread** (the gap between the extremes);
    and a scatter of that spread against how busy each firm is — the direct read on who the parameter
    actually moves.
    """)
    return


@app.cell
def _(knn_sc):
    mo.stop(len(knn_sc) < 2, mo.callout(mo.md("Needs ≥2 kNN arms in the slice to compare."), kind="warn"))
    # rank firms by article volume — both axes bite in proportion to how busy a firm is
    _vol = (
        next(iter(knn_sc.values()))
        .group_by("entity")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .get_column("entity")
        .to_list()
    )
    # default to a spread of the cross-section: busiest, upper-middle, median, quietest
    _picks = [_vol[0], _vol[len(_vol) // 4], _vol[len(_vol) // 2], _vol[-1]]
    firms_ui = mo.ui.multiselect(_vol, value=list(dict.fromkeys(_picks)), label="Firms")
    n_dates_ui = mo.ui.slider(4, 20, step=2, value=8, label="Sample dates")
    mo.hstack([firms_ui, n_dates_ui], justify="start")
    return firms_ui, n_dates_ui


@app.cell
def _(firms_ui, knn_sc):
    # daily mean novelty per firm per arm, plus that day's article count
    _per_variant = [
        sc.filter(pl.col("entity").is_in(firms_ui.value))
        .group_by("entity", "date")
        .agg(pl.col("novelty").mean().alias(v), pl.len().alias("n_articles"))
        for v, sc in knn_sc.items()
    ]
    daily = _per_variant[0]
    for _df in _per_variant[1:]:
        daily = daily.join(_df.drop("n_articles"), on=["entity", "date"], how="inner")
    daily = daily.sort("entity", "date")
    return (daily,)


@app.cell
def _(daily, firms_ui, knn_sc):
    mo.stop(daily.is_empty(), mo.callout(mo.md("No overlapping dates for the selected firms."), kind="warn"))
    _figs = []
    for _firm in firms_ui.value:
        _d = daily.filter(pl.col("entity") == _firm)
        if _d.is_empty():
            continue
        _f = go.Figure()
        for _i, _v in enumerate(knn_sc):
            _f.add_trace(
                go.Scatter(
                    x=_d["date"].to_list(),
                    y=_d[_v].to_list(),
                    mode="lines",
                    name=_v,
                    line={"color": knn_ramp(_i, len(knn_sc)), "width": 1.6},
                )
            )
        _f.update_layout(
            title=f"{_firm} — daily mean novelty ({_d['n_articles'].mean():.0f} articles/day)",
            yaxis={"title": "mean novelty"},
            hovermode="x unified",
            height=260,
            margin={"t": 40, "b": 30},
        )
        _figs.append(_f)
    mo.vstack(_figs)
    return


@app.cell
def _(daily, knn_sc, n_dates_ui):
    mo.stop(daily.is_empty(), mo.md(""))
    _vars = list(knn_sc)
    # evenly spaced sample dates per firm, so the table spans the whole window rather than one burst
    _sampled = (
        daily.with_columns(
            pl.col("date").rank("ordinal").over("entity").alias("_r"), pl.len().over("entity").alias("_n")
        )
        .filter((pl.col("_r") - 1) % (pl.col("_n") // n_dates_ui.value).clip(lower_bound=1) == 0)
        .drop("_r", "_n")
        .group_by("entity")
        .head(n_dates_ui.value)
        .sort("entity", "date")
    )
    _first, _last = _vars[0], _vars[-1]
    _tbl = _sampled.select(
        "entity",
        "date",
        pl.col("n_articles").alias("articles"),
        *[pl.col(v).round(3).alias(v) for v in _vars],
        (pl.col(_last) - pl.col(_first)).abs().round(3).alias("spread"),
    )
    mo.vstack(
        [
            mo.md(
                f"Per-firm, per-day mean novelty at each end of the swept axis. **spread** is the gap "
                f"between the extremes (`{_first}` vs `{_last}`) — the amount the parameter moves that "
                f"firm on that day. It is near zero for firms too quiet for the parameter to bite."
            ),
            mo.ui.table(_tbl, selection=None, page_size=20),
        ]
    )
    return


@app.cell
def _(daily, knn_sc):
    mo.stop(daily.is_empty() or len(knn_sc) < 2, mo.md(""))
    _vars = list(knn_sc)
    _first, _last = _vars[0], _vars[-1]
    # one row per firm: how far the swept axis moved it, against how busy it is
    _summary = (
        daily.group_by("entity")
        .agg(
            pl.col("n_articles").mean().round(1).alias("articles/day"),
            *[pl.col(v).mean().round(3).alias(v) for v in _vars],
            (pl.col(_last) - pl.col(_first)).abs().mean().round(3).alias("spread"),
        )
        .sort("articles/day", descending=True)
    )
    _fig = go.Figure(
        go.Scatter(
            x=_summary["articles/day"].to_list(),
            y=_summary["spread"].to_list(),
            mode="markers+text",
            text=_summary["entity"].to_list(),
            textposition="top center",
            marker={"color": C_KNN, "size": 11},
        )
    )
    _fig.update_layout(
        title=f"How far the swept axis moves each firm, vs its volume ({_first} vs {_last})",
        xaxis={"title": "articles / day", "type": "log"},
        yaxis={"title": "mean novelty spread across the swept axis"},
        height=380,
    )
    mo.vstack([mo.ui.table(_summary, selection=None), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6 · How many articles actually reach the sentiment count

    Every arm feeds `sentiment_ratio` a daily `(positive, neutral, negative)` count built from the
    articles that survive `novelty > θ`. So θ is not only a quality filter — it is a **sample-size
    knob**, and the three hyperparameters move that sample size in different directions:

    - **θ up** → fewer articles, monotonically.
    - **n up** → novelty shifts up, so *more* articles clear any fixed θ.
    - **window up** → novelty shifts down, so *fewer* articles clear it.

    That matters because the counts go through a Laplace ratio: a firm-day with three surviving
    articles produces a far noisier ratio than one with three hundred, and thinning the sample can
    hurt more than the cleaner signal helps. Each curve below is one arm of the §1 slice, the dashed
    vertical line is the live θ, and the table reads the curves off at exactly that θ.

    This section describes the **filter (θ)** path only. In **weight** mode nothing is ever discarded,
    so survival at a threshold is not what limits the sample; the equivalent quantity there is the total
    novelty mass, which §7 shows.
    """)
    return


@app.cell
def _(knn_sc, theta_ui):
    mo.stop(not knn_sc, mo.callout(mo.md("Load a slice in §1 first."), kind="warn"))
    _grid = np.linspace(0.0, 0.6, 31)
    _fig = go.Figure()
    _rows = []
    for _i, (_arm, _sc) in enumerate(knn_sc.items()):
        # one sort per arm, then every theta is a searchsorted — avoids 31 passes over ~9M rows
        _sorted = np.sort(_sc["novelty"].to_numpy())
        _kept = len(_sorted) - np.searchsorted(_sorted, _grid, side="right")
        _fig.add_trace(
            go.Scatter(
                x=_grid,
                y=_kept / 1e6,
                mode="lines",
                name=_arm.replace("Novel (kNN, ", "").rstrip(")"),
                line={"color": knn_ramp(_i, len(knn_sc)), "width": 2},
            )
        )
        _at = len(_sorted) - np.searchsorted(_sorted, theta_ui.value, side="right")
        _rows.append(
            {
                "arm": _arm.replace("Novel (kNN, ", "").rstrip(")"),
                "all on-topic": f"{len(_sorted):,}",
                f"kept at θ={theta_ui.value:.2f}": f"{_at:,}",
                "share kept": f"{_at / len(_sorted):.1%}",
            }
        )
    _fig.add_vline(x=theta_ui.value, line={"color": "#111111", "dash": "dash"}, annotation_text="live θ")
    _fig.update_layout(
        title="Articles reaching the daily sentiment count, as θ tightens",
        xaxis={"title": "θ"},
        yaxis={"title": "articles kept (millions)"},
        hovermode="x unified",
        height=400,
    )
    mo.vstack([_fig, mo.ui.table(pl.DataFrame(_rows), selection=None)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7 · Sharpe and sample size over the whole (window x n) grid

    §3 can only show one slice at a time. This evaluates **every cell of the grid** and puts both quantities on
    one surface: height is Sharpe, colour is how much article evidence fed it. A high ridge sitting on a
    dark (evidence-poor) patch is a Sharpe built on a thin sample — the cells most likely to be noise.
    Every cell is backtested through the production book (`nk.build_arm`), so the surface is directly
    comparable to production's own Sharpe, drawn as the reference plane.

    **The §1 toggle changes what this surface means.**

    - **Filter (θ)** — each cell needs a θ, and the choice confounds the axes: at a *fixed* θ the cells
      differ in both lookback and filter strength at once, because the same θ keeps a very different
      share of the flow at small and large n (§6 shows by how much). *Matched keep-rate* instead picks
      each cell's own θ so every cell trades the same fraction of articles, isolating the lookback —
      that is the comparison worth reading. Colour is then the θ each cell needed, or, at a fixed θ,
      the article count.
    - **Weight** — there is no θ to pick, so both modes collapse to the same surface and the θ controls
      are ignored. Every article is kept and contributes its own novelty, so the sample size that
      matters is not a count but the **total novelty mass** Σ novelty — the effective number of
      articles after each is discounted by how derivative it is. That is what colour shows.
    """)
    return


@app.cell
def _(use_ui):
    _rows = knn_manifest()
    _cells = len(_rows) * len(_rows[-1][3].n_values) if _rows else 0
    surface_mode_ui = mo.ui.radio(
        ["matched keep-rate", "fixed θ"], value="matched keep-rate", label="θ per cell", inline=True
    )
    keep_ui = mo.ui.slider(0.05, 0.9, step=0.05, value=0.4, label="target keep-rate (matched mode)")
    surface_button = mo.ui.run_button(label=f"Evaluate all {_cells} cells")
    _wtd = use_ui.value.startswith("weight")
    mo.vstack(
        [
            mo.hstack(
                [surface_button] if _wtd else [surface_mode_ui, keep_ui, surface_button],
                justify="start",
            ),
            mo.md(
                f"*One backtest per cell — {_cells} cells, roughly {_cells * 1.4:.0f} s."
                + (" Novelty is a **weight** (§1), so there is no θ to choose.*" if _wtd else "*")
            ),
        ]
    )
    return keep_ui, surface_button, surface_mode_ui


@app.cell
def _(
    keep_ui,
    prices,
    prod_long,
    sector_mask,
    span_ui,
    surface_button,
    surface_mode_ui,
    theta_ui,
    universe,
    use_ui,
    yukka_to_ric,
):
    mo.stop(not surface_button.value, mo.callout(mo.md("Press **Evaluate all … cells**."), kind="info"))
    _manifest = knn_manifest()
    _ns = sorted(_manifest[-1][3].n_values)
    _weighted = use_ui.value.startswith("weight")
    _matched = (not _weighted) and surface_mode_ui.value == "matched keep-rate"
    _cells = []
    with mo.status.progress_bar(total=len(_manifest) * len(_ns), title="Backtesting the grid…") as _bar:
        for _w, _h, _path, _p in _manifest:
            for _n in _ns:
                _sc = knn_scores(_path, _n, yukka_to_ric, universe)
                if _weighted:
                    # No threshold exists, so θ is reported as nan rather than faked as 0.0.
                    _th, _long = float("nan"), nk.weighted_long(_sc)
                else:
                    # matched mode: the theta whose keep-rate hits the target is just a quantile
                    _th = float(_sc["novelty"].quantile(1 - keep_ui.value)) if _matched else theta_ui.value
                    _long = nk.threshold_long(_sc, _th)
                # Filter mode sums 0/1 flags, so this is an article count; weighted mode sums the
                # novelty scores themselves, so it is the effective (duplicate-discounted) count.
                _kept = float(_long.select(pl.sum_horizontal("positive", "neutral", "negative")).sum().item())
                _ratio, _ = arm_signal(_long, span_ui.value)
                _pf = build_arm(prices, _ratio, span_ui.value, sector_mask)
                _s = _pf.stats.sharpe()["returns"]
                _cells.append(
                    {
                        "window": _w,
                        "window_h": _h,
                        "n": _n,
                        "theta": round(_th, 3),
                        "articles": _kept,
                        "sharpe": float("nan") if _s is None or _s != _s else float(_s),
                    }
                )
                _bar.update()
    grid_cells = pl.DataFrame(_cells).sort("window_h", "n")
    # Production over the identical panel, span and book — the plane the surface has to clear.
    grid_prod = nk.window_sharpe(
        build_arm(prices, sentiment_ratio(prod_long), span_ui.value, sector_mask),
        prices["date"].min(),
        prices["date"].max(),
    )
    return grid_cells, grid_prod


@app.cell
def _(grid_cells, grid_prod, keep_ui, prices, surface_mode_ui, use_ui):
    _wins = grid_cells.unique("window", maintain_order=True).sort("window_h")["window"].to_list()
    _ns = sorted(grid_cells["n"].unique().to_list())

    def _panel(col, scale=1.0):
        """One (window x n) matrix out of the long cell table, in the plot's row/column order."""
        return [
            [grid_cells.filter((pl.col("window") == w) & (pl.col("n") == n))[col][0] / scale for n in _ns]
            for w in _wins
        ]

    _z = _panel("sharpe")
    _axes = {
        "xaxis": {"title": "n", "tickvals": list(range(len(_ns))), "ticktext": [str(n) for n in _ns]},
        "yaxis": {"title": "window", "tickvals": list(range(len(_wins))), "ticktext": _wins},
    }
    _weighted = use_ui.value.startswith("weight")
    # Matched mode holds the article count constant by construction, so colouring by it would be a flat
    # surface carrying no information; what varies there is the theta each cell needs to hit the target.
    # Weighted mode has no theta at all, and its "count" is the summed novelty — which does vary by cell.
    _matched = (not _weighted) and surface_mode_ui.value == "matched keep-rate"
    if _weighted:
        _mode = "novelty as weight (no θ)"
        _c, _clabel = _panel("articles", 1e6), "novelty mass Σ (M)"
    elif _matched:
        _mode = f"matched keep-rate {keep_ui.value:.0%}"
        _c, _clabel = _panel("theta"), "θ needed"
    else:
        _mode = "fixed θ"
        _c, _clabel = _panel("articles", 1e6), "articles (M)"

    _f1 = go.Figure(
        go.Surface(
            z=_z,
            x=list(range(len(_ns))),
            y=list(range(len(_wins))),
            surfacecolor=_c,
            colorscale="Teal",
            colorbar={"title": _clabel},
            name="novelty grid",
        )
    )
    # Production as a flat reference plane at its own Sharpe — the surface is only interesting above it.
    _f1.add_trace(
        go.Surface(
            z=[[grid_prod] * len(_ns) for _ in _wins],
            x=list(range(len(_ns))),
            y=list(range(len(_wins))),
            colorscale=[[0, "rgba(120,120,120,0.35)"], [1, "rgba(120,120,120,0.35)"]],
            showscale=False,
            name=f"production ({grid_prod:.2f})",
            hovertemplate=f"production Sharpe {grid_prod:.2f}<extra></extra>",
        )
    )
    _f1.update_layout(
        title=f"Sharpe over the grid, coloured by {_clabel} — {_mode} · grey plane = production {grid_prod:.2f}",
        scene=_axes | {"zaxis": {"title": "Sharpe"}},
        height=560,
    )
    _f2 = go.Figure(go.Surface(z=_c, x=list(range(len(_ns))), y=list(range(len(_wins))), colorscale="Teal"))
    _f2.update_layout(
        title=f"{_clabel} over the grid — {_mode}",
        scene=_axes | {"zaxis": {"title": _clabel}},
        height=520,
    )

    _best = grid_cells.sort("sharpe", descending=True).head(1)
    _spread = grid_cells["sharpe"].max() - grid_cells["sharpe"].min()
    _se = nk.sharpe_se(prices.height)
    _beat = grid_cells.filter(pl.col("sharpe") > grid_prod).height
    _table = grid_cells.drop("window_h", *(["theta"] if _weighted else []))
    mo.vstack(
        [
            _f1,
            _f2,
            mo.ui.table(_table, selection=None, page_size=12),
            mo.callout(
                mo.md(
                    f"Best cell **{_best['window'][0]}, n={_best['n'][0]}** at Sharpe "
                    f"**{_best['sharpe'][0]:.2f}** against production's **{grid_prod:.2f}** on the same panel, "
                    f"span and book — **{_beat}/{grid_cells.height}** cells clear it. Spread across the grid is "
                    f"**{_spread:.2f}**, against a whole-window Sharpe standard error of about **±{_se:.2f}**. "
                    + (
                        "Every article is kept and weighted by its own novelty, so there is no filter "
                        "strength to confound the axes: the whole surface is the lookback, and colour is "
                        "how much novelty mass each cell's discounting leaves behind."
                        if _weighted
                        else "Every cell trades the same share of articles here, so any relief is the lookback "
                        "itself — and the colour shows how differently each cell has to be calibrated to "
                        "get there."
                        if _matched
                        else "At a fixed θ the cells differ in filter strength as well as lookback; switch "
                        "to matched keep-rate to separate the two."
                    )
                ),
                kind="info" if _spread > _se else "warn",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
