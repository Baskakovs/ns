# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.6",
#     "honey[research]",
#     "scipy",
# ]
# [tool.uv.sources]
# honey = { path = "../../../.." , editable = true }
# ///
"""Does removing the dispersion factor change the novelty ceiling?

`novelty_dispersion_check.py` established that `disp_n` — the length of the mean neighbour vector —
is a real factor inside the raw novelty score, grows with n, and varies systematically across firms
(rho = -0.767 against article flow at n=100, well past the 0.15 gate). This notebook asks the only
question that follows: with it removed, does anything about the ceiling change?

Every arm is built exactly as in `novelty_subset_ceiling.py` — same point-in-time top-200 universe,
same `sentiment_ratio` -> `_robust_signal` path, same book via `nk.build_arm` at `nk.PROD_PARAMS`,
same window. **The only thing that differs between the two arms is which score feeds the counts**,
and both scores come from the *same* scan file, so they cannot differ in article coverage.

Scope: 6h and 24h. The full 5-window grid was priced at ~17-21h of additional compute and cut. That
means the -0.800 Spearman(n, alpha) from `novelty_alpha_decomposition.py` — which averages alpha
across all five windows — is not the right reference here, so §3 **rebaselines it**: the raw score's
Spearman is recomputed on these two windows and the normalised score is judged against that.

    uv run --with marimo marimo edit book/marimo/notebooks/novelty/novelty_ceiling_normalised.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from scipy.stats import spearmanr
    from yukka.data import Index

    import honey  # noqa: F401  (import side effect: loads YUKKA_TOKEN from repo-root .env)
    from honey.data import YukkaRepository
    from honey.data.counts import sentiment_ratio
    from honey.data.preamble import load_benchmark, load_counts, load_prices, load_returns, load_sector_mask
    from honey.novelty_score_exp import dispersion_scan as ds
    from honey.novelty_score_exp import novelty_knn_model as nk
    from honey.signals import Signal

    # aliased off the leading underscore: marimo treats top-level `_names` as cell-private
    from honey.strategy.sentiment import _robust_signal as robust_signal

    SPAN = 19
    TOP_N = 200
    CLIP_C = nk.PROD_PARAMS["clip_c"]
    HORIZONS = [1, 5, 10, 21, 63]
    N_VALUES = (1, 3, 5, 7, 10, 20, 50, 70, 100)
    PPY, ANN = 252, np.sqrt(252)
    PROD = "Production (t-stat >= 2.5)"
    C_RAW, C_NORM, C_PROD = "#c0392b", "#0ea5a4", "#8a8f96"

    def arm_signal(counts_long, span=SPAN):
        """(ratio_wide, Signal) for one arm — the identical path every ceiling-notebook arm takes."""
        ratio = sentiment_ratio(counts_long)
        cols = [c for c in ratio.columns if c != "date"]
        return ratio, Signal(robust_signal(ratio, cols, span=span, clip_c=CLIP_C))

    def regress(y: np.ndarray, x: np.ndarray) -> dict:
        """OLS of arm returns on benchmark returns, annualised, plus the beta-hedged stream.

        Copied in spirit from `novelty_alpha_decomposition.py` so alpha here means what it means
        there — the hedged stream keeps alpha in, so its mean *is* alpha.
        """
        beta, alpha = np.polyfit(x, y, 1)
        hedged = y - beta * x
        return {
            "sharpe": float(y.mean() / y.std() * ANN),
            "beta": float(beta),
            "alpha_pa": float(alpha * PPY),
            "t_alpha": float(alpha / (hedged.std(ddof=2) / np.sqrt(len(y)))),
        }

    def cell_label(window: str, n: int, kind: str) -> str:
        """Stable name for one (window, n, score) arm."""
        return f"{kind}|{window}|n{n}"


@app.cell
def _():
    mo.md("""
    # The novelty ceiling, with dispersion removed

    `raw = 1 - disp_n * cos(article, mean neighbour direction)` carries a factor that describes the
    *neighbours*, not the article. `norm = 1 - cos(...)` removes it. The gate notebook showed the
    factor is large (31% compression at n=100) and firm-dependent (rho = **-0.767** against article
    flow), so it can rotate a cross-sectional ranking rather than merely shift its level.

    Everything below is production's own book. Only the score feeding the counts differs.
    """)
    return


@app.cell
def _():
    # No load button: unlike the ceiling notebook this one is cheap once the panels exist
    # (`ds.panel_for` streams, so all 36 arms cost ~20s), and a gated cell would export empty.
    prices_full = load_prices(TOP_N)
    prices = prices_full.filter((pl.col("date") >= nk.ARTICLE_DATA_START) & (pl.col("date") < nk.ARTICLE_DATA_END))
    returns = load_returns(TOP_N).filter(pl.col("date").is_between(nk.ARTICLE_DATA_START, nk.ARTICLE_DATA_END))
    prod_long = load_counts(TOP_N)
    universe = [c for c in prices.columns if c != "date"]
    # repository assets, not Index.STOXX600.assets — see the note in novelty_subset_ceiling.py
    id_map = {a.yukka_id: a.ric for a in YukkaRepository(index=Index.STOXX600).assets if a.yukka_id}
    sector_mask = load_sector_mask(prices_full)
    built = [(w, h) for w, h, _p, _prm in ds.manifest()]
    mo.stop(
        not built,
        mo.callout(mo.md("No dispersion scans. Run `dispersion_scan.py` first."), kind="warn"),
    )
    mo.callout(
        mo.md(
            f"Universe **{len(universe)}** RICs · **{prices['date'].min()} → {prices['date'].max()}** "
            f"({prices.height} trading days) · windows **{', '.join(w for w, _ in built)}** · "
            f"book `{nk.PROD_PARAMS}`"
        ),
        kind="success",
    )
    return built, id_map, prices, prod_long, returns, sector_mask, universe


@app.cell
def _(built, id_map, prices, prod_long, returns, sector_mask, universe):
    # Every arm, both scores. `ds.panel_for` aggregates inside a streaming scan, so a cell costs
    # ~0.5s instead of materialising 38.9M article rows.
    bench_df = load_benchmark("STOXX 600").rename({"close": "bm"}).sort("date")
    bench_df = bench_df.with_columns(pl.col("bm").pct_change().alias("bench")).select("date", "bench")

    _ic_rows = []
    _ratio_p, _sig_p = arm_signal(prod_long)
    _pf_p = nk.build_arm(prices, _ratio_p, sector_mask=sector_mask, span=SPAN)
    _arms = {PROD: _pf_p.returns.select("date", "returns").sort("date")}
    _ic_rows.append(
        {
            "arm": PROD,
            "kind": "prod",
            "window": None,
            "n": None,
            **{f"h={h}": round(_sig_p.mean_ic(returns, h=h), 4) for h in HORIZONS},
        }
    )

    with mo.status.progress_bar(total=len(built) * len(N_VALUES) * 2) as _bar:
        for _w, _ in built:
            for _n in N_VALUES:
                for _kind in ("raw", "norm"):
                    _panel = ds.panel_for(_w, _n, _kind, id_map, universe)
                    _ratio, _sig = arm_signal(_panel)
                    _pf = nk.build_arm(prices, _ratio, sector_mask=sector_mask, span=SPAN)
                    _key = cell_label(_w, _n, _kind)
                    _arms[_key] = _pf.returns.select("date", "returns").sort("date")
                    _ic_rows.append(
                        {
                            "arm": _key,
                            "kind": _kind,
                            "window": _w,
                            "n": _n,
                            **{f"h={h}": round(_sig.mean_ic(returns, h=h), 4) for h in HORIZONS},
                        }
                    )
                    _bar.update()

    ic_tbl = pl.DataFrame(_ic_rows)

    # one wide return panel, then one regression per arm
    _wide = None
    for _k, _df in _arms.items():
        _r = _df.rename({"returns": _k})
        _wide = _r if _wide is None else _wide.join(_r, on="date", how="inner")
    _wide = _wide.join(bench_df, on="date", how="inner").drop_nulls()
    _b = _wide["bench"].to_numpy()
    stats = pl.DataFrame(
        [
            {
                "arm": _k,
                "kind": "prod" if _k == PROD else _k.split("|")[0],
                "window": None if _k == PROD else _k.split("|")[1],
                "n": None if _k == PROD else int(_k.split("|n")[1]),
                **regress(_wide[_k].to_numpy(), _b),
            }
            for _k in _arms
        ]
    )
    prod_row = stats.filter(pl.col("arm") == PROD).to_dicts()[0]
    grid = stats.filter(pl.col("arm") != PROD)
    return grid, ic_tbl, prod_row


@app.cell
def _(ic_tbl, prod_row):
    _p = ic_tbl.filter(pl.col("kind") == "prod")
    _summary = pl.concat(
        [
            _p.select(pl.lit("production").alias("score"), *[pl.col(f"h={h}") for h in HORIZONS]),
            *[
                ic_tbl.filter(pl.col("kind") == k)
                .select(*[pl.col(f"h={h}").mean().round(4) for h in HORIZONS])
                .select(pl.lit(f"{k} (mean of cells)").alias("score"), *[pl.col(f"h={h}") for h in HORIZONS])
                for k in ("raw", "norm")
            ],
            *[
                ic_tbl.filter(pl.col("kind") == k)
                .select(*[pl.col(f"h={h}").max().round(4) for h in HORIZONS])
                .select(pl.lit(f"{k} (best cell)").alias("score"), *[pl.col(f"h={h}") for h in HORIZONS])
                for k in ("raw", "norm")
            ],
        ]
    )
    _beat = ic_tbl.filter((pl.col("kind") != "prod") & (pl.col("h=1") > _p["h=1"][0]))
    mo.vstack(
        [
            mo.md(f"""
            ## 1 · IC by horizon

            Production is the bar. A novelty arm is only interesting where it clears production's row.
            Production's Sharpe on this window is **{prod_row["sharpe"]:.3f}**.
            """),
            mo.ui.table(_summary, selection=None),
            mo.callout(
                mo.md(
                    f"Cells beating production's h=1 IC: **{_beat.filter(pl.col('kind') == 'raw').height}** raw · "
                    f"**{_beat.filter(pl.col('kind') == 'norm').height}** normalised, of "
                    f"{ic_tbl.filter(pl.col('kind') == 'raw').height} each."
                ),
                kind="neutral",
            ),
        ]
    )
    return


@app.cell
def _(grid, prod_row):
    _fig = go.Figure()
    for _k, _c in (("raw", C_RAW), ("norm", C_NORM)):
        _g = grid.filter(pl.col("kind") == _k).sort("n")
        for _w in _g["window"].unique(maintain_order=True):
            _s = _g.filter(pl.col("window") == _w)
            _fig.add_scatter(
                x=[str(v) for v in _s["n"]],
                y=_s["sharpe"],
                mode="lines+markers",
                name=f"{_k} {_w}",
                line={"color": _c, "dash": "solid" if _w == "24h" else "dot"},
            )
    _fig.add_hline(
        y=prod_row["sharpe"],
        line={"color": C_PROD, "dash": "dash"},
        annotation_text=f"production {prod_row['sharpe']:.2f}",
    )
    _fig.update_layout(
        title="Sharpe per cell", xaxis_title="n", yaxis_title="Sharpe", height=380, margin={"t": 46, "b": 30}
    )
    mo.vstack([mo.md("## 2 · Backtest Sharpe across the grid"), _fig])
    return


@app.cell
def _(grid):
    # §3 — the pre-committed test, rebaselined onto the windows actually scanned.
    _out = {}
    for _k in ("raw", "norm"):
        _byn = grid.filter(pl.col("kind") == _k).group_by("n").agg(pl.col("alpha_pa").mean()).sort("n")
        _rho, _p = spearmanr(_byn["n"].to_numpy(), _byn["alpha_pa"].to_numpy())
        _out[_k] = (float(_rho), float(_p), _byn)
    rho_raw, rho_norm = _out["raw"][0], _out["norm"][0]
    attenuated = rho_norm > -0.40
    return attenuated, rho_norm, rho_raw


@app.cell
def _(attenuated, grid, rho_norm, rho_raw):
    _fig = go.Figure()
    for _k, _c in (("raw", C_RAW), ("norm", C_NORM)):
        _byn = grid.filter(pl.col("kind") == _k).group_by("n").agg(pl.col("alpha_pa").mean()).sort("n")
        _fig.add_scatter(
            x=[str(v) for v in _byn["n"]], y=_byn["alpha_pa"] * 100, mode="lines+markers", name=_k, line={"color": _c}
        )
    _fig.update_layout(
        title="Mean annualised alpha against n",
        xaxis_title="n",
        yaxis_title="alpha % p.a.",
        height=340,
        margin={"t": 46, "b": 30},
    )
    mo.vstack(
        [
            mo.md("""
            ## 3 · Spearman(n, mean alpha) — the pre-committed test

            The n-trend in alpha was the symptom attributed to dispersion. **Pre-committed:** the
            correction is doing what the derivation predicts only if this attenuates to worse than
            **-0.40**. The published **-0.800** averages alpha over five windows; only two are scanned
            here, so the raw score is recomputed on the same two and *that* is the reference.
            """),
            _fig,
            mo.callout(
                mo.md(
                    f"Spearman(n, mean alpha): raw **{rho_raw:+.3f}** (rebaselined; published 5-window "
                    f"figure was -0.800) → normalised **{rho_norm:+.3f}**.  \n"
                    + (
                        "**Attenuated past -0.40 — the correction behaves as the derivation predicts.**"
                        if attenuated
                        else "**Not attenuated past -0.40 — the pre-committed test is not met.**"
                    )
                ),
                kind="success" if attenuated else "warn",
            ),
        ]
    )
    return


@app.cell
def _(grid, ic_tbl, prod_row, rho_norm, rho_raw):
    _r = grid.filter(pl.col("kind") == "raw")
    _n = grid.filter(pl.col("kind") == "norm")
    _p_ic = ic_tbl.filter(pl.col("kind") == "prod")["h=1"][0]
    _norm_beat = ic_tbl.filter((pl.col("kind") == "norm") & (pl.col("h=1") > _p_ic)).height
    _raw_beat = ic_tbl.filter((pl.col("kind") == "raw") & (pl.col("h=1") > _p_ic)).height
    _better = float(_n["sharpe"].mean()) - float(_r["sharpe"].mean())
    verdict = (
        "improved"
        if (_better > 0.10 and _norm_beat > _raw_beat)
        else "made it worse"
        if _better < -0.10
        else "was neutral"
    )
    mo.vstack(
        [
            mo.md("## 4 · Did it change the picture?"),
            mo.ui.table(
                pl.DataFrame(
                    [
                        {
                            "quantity": "mean Sharpe across cells",
                            "raw": round(float(_r["sharpe"].mean()), 3),
                            "normalised": round(float(_n["sharpe"].mean()), 3),
                        },
                        {
                            "quantity": "best cell Sharpe",
                            "raw": round(float(_r["sharpe"].max()), 3),
                            "normalised": round(float(_n["sharpe"].max()), 3),
                        },
                        {
                            "quantity": "mean alpha p.a.",
                            "raw": round(float(_r["alpha_pa"].mean()), 4),
                            "normalised": round(float(_n["alpha_pa"].mean()), 4),
                        },
                        {"quantity": "cells beating production IC (h=1)", "raw": _raw_beat, "normalised": _norm_beat},
                        {
                            "quantity": "Spearman(n, mean alpha)",
                            "raw": round(rho_raw, 3),
                            "normalised": round(rho_norm, 3),
                        },
                    ]
                ),
                selection=None,
            ),
            mo.callout(
                mo.md(
                    f"**Normalising {verdict}.** Mean cell Sharpe moves "
                    f"{float(_r['sharpe'].mean()):.3f} → {float(_n['sharpe'].mean()):.3f} against production's "
                    f"{prod_row['sharpe']:.3f}; cells clearing production's h=1 IC go {_raw_beat} → {_norm_beat}; "
                    f"Spearman(n, alpha) goes {rho_raw:+.3f} → {rho_norm:+.3f}."
                ),
                kind="neutral",
            ),
        ]
    )
    return


@app.cell
def _(prices):
    mo.callout(
        mo.md(f"""
        **Read rank and direction, not levels.**

        - {prices.height} trading days. Whole-window Sharpe carries a standard error of roughly
          **±0.70**, which is larger than most differences in §2.
        - The cells are **{len(N_VALUES)} settings of one measurement across 2 windows**, not
          independent trials. Neighbouring n values share most of their neighbours.
        - Only 6h and 24h are scanned; the 5-window surface was cut on cost. The -0.800 reference is
          therefore rebaselined in §3 rather than quoted.
        - Both scores come from the same scan file, so raw and normalised see identical articles. The
          recomputed raw differs from `novelty_scores_w*.parquet` by ~1.3e-3 per article because the
          scan renormalises embeddings to exactly unit length; panel sums differ by ~0.08%.
        """),
        kind="warn",
    )
    return


if __name__ == "__main__":
    app.run()
