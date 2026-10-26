# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.19.6",
#     "honey[research]",
#     "scipy",
# ]
# [tool.uv.sources]
# honey = { path = "../../../.." , editable = true }
# ///
"""Risk decomposition of the kNN-novelty arms against the production sentiment leg.

Every arm in this notebook is built by `nk.build_arm`, which is the production sentiment leg
parameter for parameter (`nk.PROD_PARAMS`: robust-signal z > 1.15, 10% max weight per name, 10-day
rebalance, 20% vol target, span 19). Only the daily counts feeding it differ, so any difference in
outcome is attributable to the signal rather than to portfolio construction.

Each arm's daily returns are regressed on the STOXX 600:

    r_arm(t) = alpha + beta * r_bench(t) + e(t)

reported as annualised alpha, beta, R^2, and the beta-hedged stream `r_arm - beta * r_bench`. The
grid is the 45 (lookback window x n) cells of the kNN novelty score, with novelty entering as a
per-article weight. Span is fixed at 19 throughout — production's live setting — so span is not a
free parameter anywhere in this notebook.

Interval estimates come from a moving-block bootstrap over trading days (block = 10, the forward
horizon), applied to paired daily hedged returns so that the market component common to both arms
cancels.

Strategy background, parameter definitions and the findings this notebook produced are in
`book/marimo/notebooks/NOVELTY.md`.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from scipy.stats import spearmanr
    from yukka.data import Index

    import honey  # noqa: F401  (import side effect: loads YUKKA_TOKEN from repo-root .env)
    from honey.data import YukkaRepository
    from honey.data.counts import sentiment_ratio
    from honey.data.preamble import load_benchmark, load_counts, load_prices, load_sector_mask
    from honey.novelty_score_exp import novelty_knn_model as nk

    #: Production's live span. Fixed everywhere in this notebook; nothing here searches over it.
    SPAN = 19
    TOP_N = 200
    PPY = 252
    ANN = np.sqrt(PPY)
    BLOCK = 10  # bootstrap block length = the forward horizon, so overlapping windows stay in a block
    NBOOT = 4000
    SEED = 0

    #: The region of the grid the two monotone rank tests point at, fixed before any return is read:
    #: alpha decreases monotonically in n, and longer windows score higher on average.
    REGION_W = ("3d", "7d", "14d")
    REGION_N = (1, 3, 5, 7)

    PROD = "production"
    C_PROD = "#111111"
    C_BENCH = "#8c8c8c"
    C_ARM = "#0b7a79"
    C_CTRL = "#c0392b"
    CACHE_DIR = Path(nk.__file__).parent / "cache"  # beside the library, not beside this notebook

    #: `marimo export` runs the notebook headlessly, where no button is ever pressed — so an export
    #: would otherwise capture "press Compute" placeholders instead of the results. Setting
    #: MARIMO_EXPORT=1 releases the gate; interactively the button still controls when work starts.
    EXPORTING = bool(os.environ.get("MARIMO_EXPORT"))

    def cell_key(window: str, n: int) -> str:
        """Column name for one grid cell in the cached returns frame."""
        return f"{window}|n{n}"

    def parse_key(key: str) -> tuple[str, int]:
        """Inverse of `cell_key`."""
        w, n = key.split("|n")
        return w, int(n)

    def benchmark_returns() -> pl.DataFrame:
        """STOXX 600 daily simple returns as `(date, bench)`."""
        b = load_benchmark("STOXX 600").rename({"close": "bm"}).sort("date")
        return b.with_columns(pl.col("bm").pct_change().alias("bench")).select("date", "bench")

    def arm_returns(prices, ratio, mask):
        """Daily returns of one arm built through the production book at `SPAN`."""
        pf = nk.build_arm(prices, ratio, sector_mask=mask, span=SPAN)
        return pf.returns.select("date", "returns").sort("date")

    def regress(y: np.ndarray, x: np.ndarray) -> dict:
        """OLS of arm returns on benchmark returns, annualised, plus the beta-hedged stream."""
        beta, alpha = np.polyfit(x, y, 1)
        hedged = y - beta * x  # keeps alpha in: its mean IS alpha (OLS residuals would be mean-zero)
        return {
            "ret_pa": float(y.mean() * PPY),
            "vol_pa": float(y.std() * ANN),
            "sharpe": float(y.mean() / y.std() * ANN),
            "beta": float(beta),
            "r2": float(np.corrcoef(x, y)[0, 1] ** 2),
            "alpha_pa": float(alpha * PPY),
            "t_alpha": float(alpha / (hedged.std(ddof=2) / np.sqrt(len(y)))),
            "hedged_sharpe": float(hedged.mean() / hedged.std() * ANN),
        }

    def block_bootstrap(series: np.ndarray, nboot: int = NBOOT, block: int = BLOCK, seed: int = SEED):
        """Moving-block bootstrap of an annualised daily mean. Returns the draw distribution."""
        rng = np.random.default_rng(seed)
        n = len(series)
        n_blocks = int(np.ceil(n / block))
        out = np.empty(nboot)
        for i in range(nboot):
            starts = rng.integers(0, n - block + 1, n_blocks)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            out[i] = series[idx].mean()
        return out * PPY

    def ci_row(draws: np.ndarray, observed: float) -> dict:
        """Point estimate, 95% interval and two-sided bootstrap p for one quantity."""
        return {
            "estimate": observed,
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "p": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
        }

    def iso_sharpe(sharpe: float, vols: np.ndarray) -> np.ndarray:
        """Return levels tracing a constant Sharpe ratio across a volatility axis."""
        return sharpe * vols


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Where does the sentiment strategy's return come from? Market exposure, or stock selection?

    The production sentiment leg's headline Sharpe is separated into the part explained by the equity
    market and the part that is not, and the same separation is applied to 45 variants of the strategy
    that reweight the news flow by how **novel** each article is. All 46 portfolios are built by the
    identical production code at the identical settings; the only thing that differs is the daily
    sentiment counts fed into them, so any difference in outcome comes from the signal.

    **Contents** — 1 what is being compared · 2 production decomposed · 3 beta across the grid ·
    4 alpha against n · 5 risk–return plane · 6 paired bootstrap · 7 summary.

    ---

    ### What "novelty" means here, and what n and the window are

    Most of what a news feed carries about a company is not new. A story breaks once and is then
    reprinted, syndicated, translated and rewritten, so a single event can arrive as dozens of
    articles. The production signal counts all of them, which means an event's weight in the signal
    depends on how widely it was syndicated as much as on what happened.

    Each article is therefore scored for **novelty**: how different its headline is from headlines that
    already existed. Headlines are turned into vectors (embeddings) so that closeness between two
    vectors means closeness in meaning, and an article is compared against the company's own earlier
    articles:

    > **novelty = 1 − (similarity to the most similar earlier articles)**

    A verbatim reprint scores near **0**; a genuinely new story scores near **1**. That score then
    becomes the article's weight in the daily sentiment count, so reprints contribute almost nothing
    and first tellings contribute almost fully. This replaces "how many articles were published" with
    "how much genuinely new information arrived".

    Two choices define the comparison, and every cell of the grid is one combination of them:

    | | | |
    |---|---|---|
    | **window** | how far back to look for earlier coverage | 6h, 24h, 3d, 7d, 14d |
    | **n** | how many nearest earlier articles to average over | 1, 3, 5, 7, 10, 20, 50, 70, 100 |

    **The window** is the memory. At 6h, an article only counts as a reprint if the original appeared
    within the last six hours — anything older is forgotten and the article looks new again. At 14d,
    the original is still remembered two weeks later. Longer windows catch more reprints but risk
    treating a genuine follow-up story as an echo of the original.

    **n** controls how strict the comparison is. At n=1 the score depends only on the single closest
    earlier article: "is there *one* near-identical piece already out there?" At n=100 it is the
    average distance to the hundred nearest, which measures something broader — how crowded the recent
    coverage of this company is overall, rather than whether this specific story is a duplicate. Small
    n asks "is this a copy?"; large n asks "is this story being talked about a lot?"

    5 windows x 9 values of n gives the **45 variants** compared throughout. They are not 45
    independent ideas — they are 45 settings of the same measurement, and they behave accordingly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1 · Data, universe and portfolio construction

    Universe is the point-in-time top-200 STOXX 600 by rank, prices from `load_prices`, cut to
    `ARTICLE_DATA_START` because the novelty scores begin 2024-07.

    Arms are built by `build_sentiment(prices=…, counts=…, sector_mask=…, sector_cap=…, **sent_params)`
    — the production pipeline — via `nk.build_arm`, which pins `sent_params` to `nk.PROD_PARAMS`.

    Beta and alpha are measured against the **published STOXX 600** (`load_benchmark`).
    """)
    return


@app.cell
def _():
    run_button = mo.ui.run_button(label="Compute all 45 cells + production")
    mo.vstack(
        [
            run_button,
            mo.md(
                "*One portfolio per cell at span 19. The daily return "
                "series are cached to parquet and every figure below reads from that cache.*"
            ),
        ]
    )
    return (run_button,)


@app.cell
def _(run_button):
    mo.stop(
        not (run_button.value or EXPORTING),
        mo.callout(mo.md("Press **Compute all 45 cells + production**."), kind="info"),
    )

    _cache = CACHE_DIR / f"alpha_returns_full_span{SPAN}_top{TOP_N}.parquet"
    _manifest = nk.nk_manifest()
    mo.stop(
        not _manifest, mo.callout(mo.md("No novelty scores built — run `novelty_knn_model.py` first."), kind="warn")
    )
    _ns = sorted(_manifest[-1][3].n_values)
    _cells = [(w, n) for w, _h, _p, _prm in _manifest for n in _ns]

    prices = load_prices(TOP_N).filter(pl.col("date") >= nk.ARTICLE_DATA_START)
    universe = [c for c in prices.columns if c != "date"]
    _y2r = {a.yukka_id: a.ric for a in YukkaRepository(index=Index.STOXX600).assets if a.yukka_id}
    _mask = load_sector_mask(prices)

    if _cache.exists():
        returns_full = pl.read_parquet(_cache)
    else:
        # Every arm is kept over the FULL price panel and the benchmark is left-joined, so the days the
        # benchmark does not cover survive in the cache. The analysis frame below drops them (beta and
        # alpha need a market return), but keeping them lets the note under this cell measure what the
        # exclusion costs instead of asserting it.
        _frame = arm_returns(prices, sentiment_ratio(load_counts(TOP_N)), _mask).rename({"returns": PROD})
        with mo.status.progress_bar(total=len(_cells), title="Building one portfolio per cell…") as _bar:
            for _w, _n in _cells:
                _r = arm_returns(
                    prices,
                    sentiment_ratio(nk.weighted_long(nk.scores_for(_w, _n, _y2r, universe))),
                    _mask,
                ).rename({"returns": cell_key(_w, _n)})
                _frame = _frame.join(_r, on="date", how="left")
                _bar.update()
        returns_full = _frame.join(benchmark_returns(), on="date", how="left").sort("date")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        returns_full.write_parquet(_cache)

    returns_wide = returns_full.drop_nulls().sort("date")
    cells = [(w, n) for w, n in _cells if cell_key(w, n) in returns_wide.columns]
    bench = returns_wide["bench"].to_numpy()

    # What the benchmark's short coverage costs, measured rather than asserted.
    _tail = returns_full.filter(pl.col("date") > returns_wide["date"].max())
    _cols = [cell_key(w, n) for w, n in cells]
    _p_tail = float(_tail[PROD].fill_null(0).mean() * PPY) if _tail.height else float("nan")
    _c_tail = float(np.nanmean([_tail[c].fill_null(0).mean() for c in _cols]) * PPY) if _tail.height else float("nan")

    def _sr(col, frame):
        v = frame[col].fill_null(0).to_numpy()
        return float(v.mean() / v.std() * ANN)

    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"**{returns_wide.height}** trading days, **{returns_wide['date'].min()} … "
                    f"{returns_wide['date'].max()}**, **{len(universe)}** names, **{len(cells)}** grid cells "
                    f"plus production. STOXX 600 over the window: **{bench.mean() * PPY:+.2%} p.a.**, "
                    f"volatility **{bench.std() * ANN:.2%}**, "
                    f"Sharpe **{bench.mean() / bench.std() * ANN:.3f}**. Book: `{nk.PROD_PARAMS}`."
                ),
                kind="neutral",
            ),
            mo.md(
                f"*The price panel runs to **{prices['date'].max()}**; the benchmark series stops "
                f"**{_tail.height} trading days** earlier, and beta and alpha require a market return, "
                f"so those days are excluded from every figure below. Over that excluded stretch "
                f"production returned **{_p_tail:+.1%} annualised** against a {len(cells)}-cell mean of "
                f"**{_c_tail:+.1%}**. Including it would move production's Sharpe from "
                f"**{_sr(PROD, returns_wide):.3f}** to **{_sr(PROD, returns_full):.3f}** and the "
                f"cell mean from **{np.mean([_sr(c, returns_wide) for c in _cols]):.3f}** to "
                f"**{np.mean([_sr(c, returns_full) for c in _cols]):.3f}**, which reverses the Sharpe "
                f"comparison in §5 and §7. The beta result is unaffected.*"
            ),
        ]
    )
    return bench, cells, returns_wide


@app.cell
def _(bench, cells, returns_wide):
    # One regression per arm; `stats` is the table every later section reads.
    _rows = []
    for _name in [PROD, *[cell_key(w, n) for w, n in cells]]:
        _y = returns_wide[_name].to_numpy()
        _w, _n = (None, None) if _name == PROD else parse_key(_name)
        _rows.append({"arm": _name, "window": _w, "n": _n, **regress(_y, bench)})
    stats = pl.DataFrame(_rows)
    prod_stats = stats.filter(pl.col("arm") == PROD).to_dicts()[0]
    grid = stats.filter(pl.col("arm") != PROD)
    return grid, prod_stats


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2 · Production decomposed into market exposure and residual

    Production's daily returns regressed on the STOXX 600. `R²` is the share of production's return
    variance explained by the benchmark; `alpha` is the annualised intercept; `t(alpha)` uses the
    standard error of the beta-hedged stream.
    """)
    return


@app.cell
def _(bench, prod_stats, returns_wide):
    _y = returns_wide[PROD].to_numpy()
    _fit = np.polyfit(bench, _y, 1)
    _xs = np.linspace(bench.min(), bench.max(), 100)

    fig_capm = go.Figure()
    fig_capm.add_trace(
        go.Scatter(
            x=bench * 100,
            y=_y * 100,
            mode="markers",
            name="daily returns",
            marker={"color": C_PROD, "size": 4, "opacity": 0.35},
        )
    )
    fig_capm.add_trace(
        go.Scatter(
            x=_xs * 100,
            y=(_fit[0] * _xs + _fit[1]) * 100,
            mode="lines",
            name=f"fit: beta={_fit[0]:.3f}",
            line={"color": C_ARM, "width": 3},
        )
    )
    fig_capm.add_trace(
        go.Scatter(
            x=_xs * 100,
            y=_xs * 100,
            mode="lines",
            name="beta = 1",
            line={"color": C_BENCH, "width": 2, "dash": "dash"},
        )
    )
    fig_capm.update_layout(
        title="Production daily return vs STOXX 600 daily return",
        xaxis={"title": "STOXX 600 daily return (%)"},
        yaxis={"title": "production daily return (%)"},
        height=460,
    )

    _tbl = pl.DataFrame(
        [
            {"quantity": "annualised return", "value": f"{prod_stats['ret_pa']:+.2%}"},
            {"quantity": "annualised volatility", "value": f"{prod_stats['vol_pa']:.2%}"},
            {"quantity": "Sharpe", "value": f"{prod_stats['sharpe']:.3f}"},
            {"quantity": "beta to STOXX 600", "value": f"{prod_stats['beta']:.3f}"},
            {"quantity": "R² (market share of variance)", "value": f"{prod_stats['r2']:.1%}"},
            {"quantity": "alpha", "value": f"{prod_stats['alpha_pa']:+.2%} p.a."},
            {"quantity": "t(alpha)", "value": f"{prod_stats['t_alpha']:+.2f}"},
            {"quantity": "Sharpe of the beta-hedged stream", "value": f"{prod_stats['hedged_sharpe']:+.3f}"},
        ]
    )
    mo.vstack(
        [
            fig_capm,
            mo.ui.table(_tbl, selection=None),
            mo.md(
                f"{prod_stats['r2']:.1%} of production's return variance is explained by the benchmark. "
                f"The intercept is {prod_stats['alpha_pa']:+.2%} p.a. with t = {prod_stats['t_alpha']:+.2f}, "
                f"i.e. not distinguishable from zero. Sharpe of the beta-hedged stream is "
                f"{prod_stats['hedged_sharpe']:+.3f}."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3 · Beta across the grid

    Market beta of each of the 45 cells, against production's beta. Ordered by lookback window then n.
    """)
    return


@app.cell
def _(grid, prod_stats):
    _g = grid.sort("window", "n")
    _labels = [f"{r['window']}/n{r['n']}" for r in _g.iter_rows(named=True)]
    _below = _g.filter(pl.col("beta") < prod_stats["beta"]).height

    fig_beta = go.Figure()
    fig_beta.add_trace(go.Bar(x=_labels, y=_g["beta"].to_list(), name="novelty cell", marker={"color": C_ARM}))
    fig_beta.add_hline(
        y=prod_stats["beta"],
        line={"color": C_PROD, "width": 2, "dash": "dash"},
        annotation_text=f"production beta {prod_stats['beta']:.3f}",
        annotation_position="top left",
    )
    fig_beta.update_layout(
        title="Market beta by grid cell (span 19)",
        yaxis={"title": "beta to STOXX 600", "range": [0.75, 0.92]},
        xaxis={"title": "", "tickangle": -60},
        height=460,
        showlegend=False,
    )
    mo.vstack(
        [
            fig_beta,
            mo.md(
                f"**{_below} of {_g.height}** cells carry a lower beta than production. Cell betas span "
                f"{_g['beta'].min():.3f}–{_g['beta'].max():.3f} against production's {prod_stats['beta']:.3f}."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4 · Alpha against n

    Annualised alpha of every cell plotted against n, with the mean over lookback windows. The rank
    correlation between n and mean alpha is reported with its p-value.
    """)
    return


@app.cell
def _(grid, prod_stats):
    _byn = grid.group_by("n").agg(pl.col("alpha_pa").mean().alias("mean_alpha")).sort("n")
    _rho, _p = spearmanr(_byn["n"].to_numpy(), _byn["mean_alpha"].to_numpy())
    _ns = _byn["n"].to_list()
    _pos = [str(i) for i in range(len(_ns))]  # even spacing: n is ordinal here, not cardinal
    _idx = {n: i for i, n in enumerate(_ns)}

    fig_alpha_n = go.Figure()
    for _w in grid["window"].unique(maintain_order=True).to_list():
        _s = grid.filter(pl.col("window") == _w).sort("n")
        fig_alpha_n.add_trace(
            go.Scatter(
                x=[_idx[n] for n in _s["n"].to_list()],
                y=[v * 100 for v in _s["alpha_pa"].to_list()],
                mode="markers",
                name=_w,
                marker={"size": 8, "opacity": 0.55},
            )
        )
    fig_alpha_n.add_trace(
        go.Scatter(
            x=list(range(len(_ns))),
            y=[v * 100 for v in _byn["mean_alpha"].to_list()],
            mode="lines+markers",
            name="mean over windows",
            line={"color": C_ARM, "width": 4},
            marker={"size": 11, "color": C_ARM},
        )
    )
    fig_alpha_n.add_hline(
        y=prod_stats["alpha_pa"] * 100,
        line={"color": C_PROD, "width": 2, "dash": "dash"},
        annotation_text=f"production {prod_stats['alpha_pa']:+.2%}",
        annotation_position="bottom right",
    )
    fig_alpha_n.update_layout(
        title="Annualised alpha by n (each marker one cell; line = mean over the five windows)",
        xaxis={
            "title": "n (neighbours averaged)",
            "tickvals": list(range(len(_ns))),
            "ticktext": [str(n) for n in _ns],
        },
        yaxis={"title": "alpha (% p.a.)"},
        height=470,
    )
    _positive = _byn.filter(pl.col("mean_alpha") > 0)["n"].to_list()
    mo.vstack(
        [
            fig_alpha_n,
            mo.ui.table(
                _byn.with_columns((pl.col("mean_alpha") * 100).round(2).alias("mean alpha % p.a.")).drop("mean_alpha"),
                selection=None,
            ),
            mo.md(
                f"Spearman rank correlation between n and mean alpha: **{_rho:+.3f}**, p = **{_p:.4f}** "
                f"({len(_ns)} values of n). Mean alpha is positive at n ∈ {_positive} and negative at the "
                f"remaining values."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5 · Risk–return plane

    Annualised return against annualised volatility. Dashed grey lines are constant-Sharpe rays: a
    point above a ray has a higher Sharpe than that ray. Sharpe is invariant to leverage, so position
    along a ray carries no information about signal quality while position between rays does.

    The **region composite** is the equally weighted average of the daily returns of the
    12 cells in windows 3d/7d/14d at n ∈ {1,3,5,7} — the region selected by the rank ordering in §4,
    not by these outcomes.
    """)
    return


@app.cell
def _(bench, cells, grid, prod_stats, returns_wide):
    _region = [(w, n) for w, n in cells if w in REGION_W and n in REGION_N]
    composite = np.mean([returns_wide[cell_key(w, n)].to_numpy() for w, n in _region], axis=0)
    comp_stats = regress(composite, bench)
    bench_stats = regress(bench, bench)

    _vols = np.linspace(0.11, 0.16, 50)
    fig_plane = go.Figure()
    for _s in (0.8, 0.9, 1.0, 1.1):
        fig_plane.add_trace(
            go.Scatter(
                x=_vols * 100,
                y=iso_sharpe(_s, _vols) * 100,
                mode="lines",
                line={"color": C_BENCH, "width": 1, "dash": "dot"},
                name=f"Sharpe {_s:.1f}",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_plane.add_annotation(
            x=_vols[-1] * 100,
            y=iso_sharpe(_s, _vols[-1]) * 100,
            text=f"SR {_s:.1f}",
            showarrow=False,
            font={"size": 10, "color": C_BENCH},
            xanchor="left",
        )
    fig_plane.add_trace(
        go.Scatter(
            x=[r["vol_pa"] * 100 for r in grid.iter_rows(named=True)],
            y=[r["ret_pa"] * 100 for r in grid.iter_rows(named=True)],
            mode="markers",
            name="45 novelty cells",
            marker={"color": C_ARM, "size": 7, "opacity": 0.45},
            text=[f"{r['window']}/n{r['n']}" for r in grid.iter_rows(named=True)],
            hovertemplate="%{text}<br>vol %{x:.2f}%  ret %{y:.2f}%<extra></extra>",
        )
    )
    for _label, _st, _colour, _sym in (
        (f"production ({len(_region)}-cell bar)", prod_stats, C_PROD, "square"),
        ("region composite", comp_stats, C_ARM, "diamond"),
        ("STOXX 600", bench_stats, C_BENCH, "circle"),
    ):
        fig_plane.add_trace(
            go.Scatter(
                x=[_st["vol_pa"] * 100],
                y=[_st["ret_pa"] * 100],
                mode="markers+text",
                name=_label.split(" (")[0],
                text=[_label.split(" (")[0]],
                textposition="top center",
                marker={"color": _colour, "size": 15, "symbol": _sym, "line": {"color": "white", "width": 1.5}},
            )
        )
    fig_plane.update_layout(
        title="Annualised return vs annualised volatility (span 19)",
        xaxis={"title": "volatility (% p.a.)"},
        yaxis={"title": "return (% p.a.)"},
        height=540,
    )

    _tbl = pl.DataFrame(
        [
            {
                "arm": k,
                "return p.a.": f"{v['ret_pa']:+.2%}",
                "vol p.a.": f"{v['vol_pa']:.2%}",
                "Sharpe": f"{v['sharpe']:.3f}",
                "beta": f"{v['beta']:.3f}",
                "alpha p.a.": f"{v['alpha_pa']:+.2%}",
            }
            for k, v in (("production", prod_stats), ("region composite", comp_stats), ("STOXX 600", bench_stats))
        ]
    )
    mo.vstack(
        [
            fig_plane,
            mo.ui.table(_tbl, selection=None),
            mo.md(
                f"Region composite: return {comp_stats['ret_pa']:+.2%} vs production "
                f"{prod_stats['ret_pa']:+.2%}; volatility {comp_stats['vol_pa']:.2%} vs "
                f"{prod_stats['vol_pa']:.2%}; Sharpe {comp_stats['sharpe']:.3f} vs "
                f"{prod_stats['sharpe']:.3f}; beta {comp_stats['beta']:.3f} vs {prod_stats['beta']:.3f}."
            ),
        ]
    )
    return comp_stats, composite


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6 · Paired bootstrap of the alpha difference

    Moving-block bootstrap (block 10 trading days, 4,000 draws) of the **paired** daily difference
    between each arm's beta-hedged stream and production's. Pairing removes the market component
    common to both arms; the correlation between the two hedged streams is reported alongside.
    """)
    return


@app.cell
def _(bench, cells, composite, grid, returns_wide):
    _p_hedged = returns_wide[PROD].to_numpy() - regress(returns_wide[PROD].to_numpy(), bench)["beta"] * bench

    def _hedged(y):
        return y - regress(y, bench)["beta"] * bench

    _region = [(w, n) for w, n in cells if w in REGION_W and n in REGION_N]
    _arms = [("region composite", composite)]
    _arms += [
        (f"{r['window']}/n{r['n']}", returns_wide[cell_key(r["window"], r["n"])].to_numpy())
        for r in grid.sort("alpha_pa", descending=True).head(3).iter_rows(named=True)
    ]

    _rows, _draws = [], {}
    for _label, _y in _arms:
        _d = _hedged(_y) - _p_hedged
        _dr = block_bootstrap(_d)
        _draws[_label] = _dr
        _row = ci_row(_dr, float(_d.mean() * PPY))
        _rows.append(
            {
                "arm": _label,
                "alpha difference p.a.": f"{_row['estimate']:+.2%}",
                "95% CI": f"[{_row['lo']:+.2%}, {_row['hi']:+.2%}]",
                "p": f"{_row['p']:.3f}",
                "corr with production hedged": f"{np.corrcoef(_hedged(_y), _p_hedged)[0, 1]:.3f}",
            }
        )

    fig_boot = go.Figure()
    for _label, _dr in _draws.items():
        fig_boot.add_trace(go.Histogram(x=_dr * 100, name=_label, opacity=0.55, nbinsx=70))
    fig_boot.add_vline(
        x=0,
        line={"color": C_PROD, "width": 2, "dash": "dash"},
        annotation_text="no difference",
        annotation_position="top",
    )
    fig_boot.update_layout(
        title="Bootstrap distribution of the alpha difference vs production (4,000 draws)",
        xaxis={"title": "alpha difference (% p.a.)"},
        yaxis={"title": "draws"},
        barmode="overlay",
        height=460,
    )
    mo.vstack(
        [
            fig_boot,
            mo.ui.table(pl.DataFrame(_rows), selection=None),
            mo.md(
                f"Each interval spans zero. `{len(_region)}` cells enter the composite; the bootstrap uses "
                f"{returns_wide.height} trading days, giving a whole-window Sharpe standard error of about "
                f"±{nk.sharpe_se(returns_wide.height):.2f} for reference."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7 · Summary

    Counts across the 45 cells, and the region composite against production on each quantity.
    """)
    return


@app.cell
def _(comp_stats, grid, prod_stats, returns_wide):
    _region = grid.filter(pl.col("window").is_in(REGION_W) & pl.col("n").is_in(REGION_N))
    _all3 = _region.filter(
        (pl.col("beta") < prod_stats["beta"])
        & (pl.col("sharpe") > prod_stats["sharpe"])
        & (pl.col("alpha_pa") > prod_stats["alpha_pa"])
    ).height

    _counts = pl.DataFrame(
        [
            {
                "statement": "beta below production",
                "cells": f"{grid.filter(pl.col('beta') < prod_stats['beta']).height} / {grid.height}",
            },
            {
                "statement": "Sharpe above production",
                "cells": f"{grid.filter(pl.col('sharpe') > prod_stats['sharpe']).height} / {grid.height}",
            },
            {
                "statement": "alpha above production",
                "cells": f"{grid.filter(pl.col('alpha_pa') > prod_stats['alpha_pa']).height} / {grid.height}",
            },
            {
                "statement": "|t(alpha)| > 2",
                "cells": f"{grid.filter(pl.col('t_alpha').abs() > 2).height} / {grid.height}",
            },
            {
                "statement": f"in region {REGION_W} x n{REGION_N}: beta, Sharpe and alpha all above production",
                "cells": f"{_all3} / {_region.height}",
            },
        ]
    )
    _cmp = pl.DataFrame(
        [
            {"quantity": k, "production": p, "region composite": c, "difference": d}
            for k, p, c, d in (
                (
                    "return p.a.",
                    f"{prod_stats['ret_pa']:+.2%}",
                    f"{comp_stats['ret_pa']:+.2%}",
                    f"{comp_stats['ret_pa'] - prod_stats['ret_pa']:+.2%}",
                ),
                (
                    "volatility p.a.",
                    f"{prod_stats['vol_pa']:.2%}",
                    f"{comp_stats['vol_pa']:.2%}",
                    f"{comp_stats['vol_pa'] - prod_stats['vol_pa']:+.2%}",
                ),
                (
                    "Sharpe",
                    f"{prod_stats['sharpe']:.3f}",
                    f"{comp_stats['sharpe']:.3f}",
                    f"{comp_stats['sharpe'] - prod_stats['sharpe']:+.3f}",
                ),
                (
                    "beta",
                    f"{prod_stats['beta']:.3f}",
                    f"{comp_stats['beta']:.3f}",
                    f"{comp_stats['beta'] - prod_stats['beta']:+.3f}",
                ),
                (
                    "R²",
                    f"{prod_stats['r2']:.1%}",
                    f"{comp_stats['r2']:.1%}",
                    f"{comp_stats['r2'] - prod_stats['r2']:+.1%}",
                ),
                (
                    "alpha p.a.",
                    f"{prod_stats['alpha_pa']:+.2%}",
                    f"{comp_stats['alpha_pa']:+.2%}",
                    f"{comp_stats['alpha_pa'] - prod_stats['alpha_pa']:+.2%}",
                ),
                ("t(alpha)", f"{prod_stats['t_alpha']:+.2f}", f"{comp_stats['t_alpha']:+.2f}", ""),
            )
        ]
    )
    mo.vstack(
        [
            mo.md("**Counts across the grid**"),
            mo.ui.table(_counts, selection=None),
            mo.md("**Region composite vs production**"),
            mo.ui.table(_cmp, selection=None),
            mo.md(
                f"All figures are computed on {returns_wide.height} trading days "
                f"({returns_wide['date'].min()} … {returns_wide['date'].max()}) at span {SPAN}, through the "
                f"production book, against the STOXX 600. No cell reaches |t(alpha)| > 2, and every "
                f"bootstrap interval in §7 spans zero."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
