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
"""Is the kNN novelty score contaminated by neighbour dispersion — and does it matter?

The score is `1 - mean(top-n cosine similarities to earlier same-company headlines)`. With unit-norm
embeddings that mean is a dot product against the *mean vector* of those n neighbours, and a mean
vector is short whenever the neighbours disagree:

    raw  = 1 - disp_n * cos(article, mean direction)
    disp_n = || (1/n) * sum of the n nearest earlier embeddings ||
    norm = 1 - cos(article, mean direction)

`disp_n` depends only on how much the *neighbours* disagree, not on the article being scored, so it
is a factor the score should not carry. It is 1 at n=1 and falls as n grows.

That it is < 1 is arithmetic, not a finding. A uniform contamination only shifts the level and
cannot tilt a cross-sectional ranking, so the question this notebook answers is whether `disp_n`
**varies systematically across firms**. §4 is the gate for doing anything about it.

Inputs come from `dispersion_scan.py`; run that first:

    uv run src/honey/novelty_score_exp/dispersion_scan.py

Open with the project environment:

    uv run --with marimo marimo edit book/marimo/notebooks/novelty/novelty_dispersion_check.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl

    import honey  # noqa: F401  (import side effect: loads YUKKA_TOKEN from repo-root .env)
    from honey.novelty_score_exp import dispersion_scan as ds

    #: Where the cached per-window check tables live (gitignored, regenerable).
    CACHE = Path(__file__).resolve().parents[4] / "src" / "honey" / "novelty_score_exp" / "cache"
    C_POS, C_NEG, C_MUT = "#0ea5a4", "#c0392b", "#8a8f96"
    #: Gate threshold: below this the contamination cannot tilt a cross-sectional ranking enough to matter.
    GATE_RHO = 0.15
    #: The gate is judged at the deep end of the n grid, where dispersion is largest.
    GATE_N_MIN = 20

    def windows() -> list[tuple[str, float, Path, object]]:
        """Built dispersion parquets, shortest window first."""
        return ds.manifest()

    def _check_tables(path: Path, n_values: tuple[int, ...]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Aggregate one window's per-article scan into the three small tables the checks need.

        Reads lazily and streams: the scan files hold ~39M rows each, which is far more than a
        notebook should ever hold in memory.
        """
        lf = pl.scan_parquet(path)
        by_n_rows, daily_rows = [], []
        for n in n_values:
            d, nm = ds.disp_col(n), ds.norm_col(n)
            # fill_nan(None) is load-bearing: unscored articles are float NaN, and in polars NaN is
            # NOT null -- mean() propagates it and drop_nulls() does not remove it, so without this
            # every aggregate below comes back NaN.
            dc, nc = pl.col(d).fill_nan(None), pl.col(nm).fill_nan(None)
            raw = (1.0 - dc * (1.0 - nc)).alias("raw")
            base = lf.select("date", dc.alias("disp"), nc.alias("norm"), raw).drop_nulls()
            agg = base.select(
                pl.col("disp").mean().alias("mean_disp"),
                pl.col("disp").std().alias("sd_disp"),
                pl.col("raw").mean().alias("mean_raw"),
                pl.col("norm").mean().alias("mean_norm"),
                pl.len().alias("rows"),
            ).collect(engine="streaming")
            by_n_rows.append({"n": n, **agg.to_dicts()[0]})

            # cross-sectional Spearman per day, at article level: does disp actually reach the score?
            rho = (
                base.with_columns(
                    r_raw=pl.col("raw").rank().over("date"),
                    r_disp=pl.col("disp").rank().over("date"),
                )
                .group_by("date")
                .agg(rho=pl.corr("r_raw", "r_disp"))
                .select(pl.col("rho").mean())
                .collect(engine="streaming")
                .item()
            )
            daily_rows.append({"n": n, "mean_rho": rho})

        firm = (
            lf.group_by("entity")
            .agg(
                *[pl.col(ds.disp_col(n)).fill_nan(None).mean().alias(f"disp_n{n}") for n in n_values],
                pl.len().alias("articles"),
                pl.col("date").n_unique().alias("days"),
            )
            .with_columns(arts_per_day=pl.col("articles") / pl.col("days"))
            .collect(engine="streaming")
        )
        return pl.DataFrame(by_n_rows), pl.DataFrame(daily_rows), firm

    def check_tables(label: str, path: Path, n_values: tuple[int, ...]) -> tuple[pl.DataFrame, ...]:
        """`_check_tables` with a parquet cache, so reopening the notebook is instant."""
        CACHE.mkdir(parents=True, exist_ok=True)
        paths = [CACHE / f"dispcheck_{label}_{k}.parquet" for k in ("byn", "daily", "firm")]
        if all(p.exists() for p in paths):
            return tuple(pl.read_parquet(p) for p in paths)
        tables = _check_tables(path, n_values)
        for t, p in zip(tables, paths, strict=True):
            t.write_parquet(p)
        return tables


@app.cell
def _():
    mo.md("""
    # Neighbour dispersion in the kNN novelty score

    `raw = 1 - disp_n * cos(article, mean neighbour direction)`. The factor `disp_n` is the length
    of the mean neighbour vector — it measures how much the *neighbours* disagree with each other
    and has nothing to do with the article being scored.

    That `disp_n < 1` is arithmetic, not a finding. A contamination that is the same for every firm
    shifts the level and cannot tilt a cross-sectional ranking. **§4 is the gate:** it only matters
    if `disp_n` varies systematically across firms.
    """)
    return


@app.cell
def _():
    _built = windows()
    mo.stop(
        not _built,
        mo.callout(
            mo.md("No dispersion scans found. Run `uv run src/honey/novelty_score_exp/dispersion_scan.py` first."),
            kind="warn",
        ),
    )
    window_ui = mo.ui.dropdown(
        {f"{lab} ({hrs:g}h)": lab for lab, hrs, _p, _prm in _built},
        value=f"{_built[len(_built) // 2][0]} ({_built[len(_built) // 2][1]:g}h)",
        label="Lookback window",
    )
    window_ui
    return (window_ui,)


@app.cell
def _(window_ui):
    _row = next(r for r in windows() if r[0] == window_ui.value)
    label, path, params = _row[0], _row[2], _row[3]
    n_values = tuple(sorted(params.n_values))
    by_n, daily, firm = check_tables(label, path, n_values)
    mo.md(f"**{params.describe()}** · {by_n['rows'].max():,} scored articles · {firm.height} companies")
    return by_n, daily, firm, label, n_values


@app.cell
def _(by_n):
    # CHECK 1 — the correctness test for the whole compute stage.
    _d1 = float(by_n.filter(pl.col("n") == 1)["mean_disp"][0])
    _raw1 = float(by_n.filter(pl.col("n") == 1)["mean_raw"][0])
    _norm1 = float(by_n.filter(pl.col("n") == 1)["mean_norm"][0])
    _err_d, _err_s = abs(_d1 - 1.0), abs(_raw1 - _norm1)
    check1 = (_err_d < 1e-5) and (_err_s < 1e-5)
    mo.vstack(
        [
            mo.md("""
            ## 1 · Implementation check

            **Criterion:** `disp_n` at n=1 must be 1.0 to floating-point tolerance (a single neighbour
            *is* its own mean, and vectors are renormalised on load), and `norm` must equal `raw` at
            n=1. If either fails the neighbour gather is wrong and nothing below is meaningful.
            """),
            mo.callout(
                mo.md(
                    f"`|mean disp_1 - 1|` = **{_err_d:.2e}** · `|mean raw_1 - mean norm_1|` = **{_err_s:.2e}**"
                    + ("" if check1 else "  \n**FAILED — stop. Do not read §2-§4.**")
                ),
                kind="success" if check1 else "danger",
            ),
        ]
    )
    return (check1,)


@app.cell
def _(by_n, check1):
    mo.stop(not check1, mo.callout(mo.md("§1 failed — the remaining sections are not meaningful."), kind="danger"))
    _d = by_n.sort("n")
    _mono = bool(np.all(np.diff(_d["mean_disp"].to_numpy()) <= 1e-9))
    _fig = go.Figure(go.Scatter(x=_d["n"], y=_d["mean_disp"], mode="lines+markers", line={"color": C_POS}))
    _fig.update_layout(
        title="Mean disp_n by neighbour count",
        xaxis_title="n",
        yaxis_title="mean disp",
        xaxis_type="log",
        height=300,
        margin={"t": 46, "b": 30},
    )
    _lo = float(_d["mean_disp"].min())
    mo.vstack(
        [
            mo.md("""
            ## 2 · Does contamination grow with n?

            **Criterion:** mean `disp_n` must decline monotonically in n. If it does not, `disp_n` is
            not behaving as the derivation says and the correction has no basis.
            """),
            _fig,
            mo.callout(
                mo.md(
                    f"Monotonically declining: **{'yes' if _mono else 'no'}** · "
                    f"falls from **1.000** at n=1 to **{_lo:.3f}** at n={int(_d['n'].max())}, i.e. the raw "
                    f"score is compressed by **{(1 - _lo) * 100:.0f}%** at the deep end."
                ),
                kind="success" if _mono else "danger",
            ),
        ]
    )
    return


@app.cell
def _(check1, daily):
    mo.stop(not check1, mo.md(""))
    _d = daily.sort("n")
    _rho = _d["mean_rho"].to_numpy()
    _mag = np.abs(np.nan_to_num(_rho))
    _inversions = int(np.sum(np.diff(_mag) < -1e-6))
    _ok = _inversions <= 1
    _fig = go.Figure(
        go.Bar(x=[str(v) for v in _d["n"]], y=_rho, marker_color=[C_NEG if v < 0 else C_POS for v in _rho])
    )
    _fig.update_layout(
        title="Mean per-day cross-sectional Spearman(raw, disp) by n",
        xaxis_title="n",
        yaxis_title="rho",
        height=300,
        margin={"t": 46, "b": 30},
    )
    mo.vstack(
        [
            mo.md("""
            ## 3 · Does it reach the score?

            **Criterion:** the correlation must be ~0 at n=1 — where `disp` is constant at 1, so it
            cannot correlate with anything — and grow in magnitude with n. One inversion allowed.
            Measured across articles within each day, then averaged over days.
            """),
            _fig,
            mo.callout(
                mo.md(
                    f"rho at n=1 = **{_rho[0]:+.3f}** · at n={int(_d['n'].max())} = **{_rho[-1]:+.3f}** · "
                    f"inversions in |rho|: **{_inversions}**"
                ),
                kind="success" if _ok else "warn",
            ),
        ]
    )
    return


@app.cell
def _(check1, firm, n_values):
    mo.stop(not check1, mo.md(""))
    # CHECK 4 / THE GATE. A uniform disp shifts the level; only cross-firm variation tilts a ranking.
    from scipy.stats import spearmanr

    _rows = []
    _arts = firm["arts_per_day"].to_numpy()
    for n in n_values:
        _v = firm[f"disp_n{n}"].to_numpy()
        _m = np.isfinite(_v) & np.isfinite(_arts)
        # disp is exactly 1 for every firm at n=1, so the correlation is undefined, not zero
        _const = np.ptp(_v[_m]) == 0 if _m.sum() else True
        _rho_arts = np.nan if _const or _m.sum() <= 2 else spearmanr(_v[_m], _arts[_m]).statistic
        _rows.append(
            {
                "n": n,
                "cross-firm sd": round(float(np.nanstd(_v)), 4),
                "rho vs articles/day": None if _rho_arts != _rho_arts else round(float(_rho_arts), 3),
            }
        )
    tilt = pl.DataFrame(_rows)
    _deep = tilt.filter(pl.col("n") >= GATE_N_MIN)
    gate_rho = float(np.nanmax(np.abs(_deep["rho vs articles/day"].to_numpy())))
    return gate_rho, tilt


@app.cell
def _(check1, firm, gate_rho, tilt):
    mo.stop(not check1, mo.md(""))
    _fig = go.Figure()
    _fig.add_bar(x=[str(v) for v in tilt["n"]], y=tilt["cross-firm sd"], marker_color=C_POS, name="cross-firm sd")
    _fig.update_layout(
        title="Cross-firm dispersion of disp_n (the quantity that can tilt a ranking)",
        xaxis_title="n",
        yaxis_title="sd across firms",
        height=300,
        margin={"t": 46, "b": 30},
    )
    mo.vstack(
        [
            mo.md(f"""
            ## 4 · Does it create a cross-sectional tilt? — **the gate**

            **Criterion:** `disp_n` must vary across firms *and* line up with a firm characteristic,
            with `|rho| >= {GATE_RHO}` at n >= {GATE_N_MIN}. A uniform contamination cannot tilt a
            cross-sectional ranking, so without this the correction cannot change any result.

            Dollar volume is **not available** — `IndexConstituent` carries only name/isin/ric/yukka_id
            and the price panel is close-only — so the size proxy here is raw article flow, which the
            spec names as the other leg. Market-cap rank needs a slow API ladder and is not run by default.
            """),
            _fig,
            mo.ui.table(tilt, selection=None),
            mo.callout(
                mo.md(
                    f"Max `|rho|` against articles/day at n >= {GATE_N_MIN}: **{gate_rho:.3f}** "
                    f"(threshold {GATE_RHO}) over {firm.height} firms."
                ),
                kind="success" if gate_rho >= GATE_RHO else "warn",
            ),
        ]
    )
    return


@app.cell
def _(check1, gate_rho, label, tilt):
    _passed = check1 and gate_rho >= GATE_RHO
    _sd_deep = float(tilt.filter(pl.col("n") >= GATE_N_MIN)["cross-firm sd"].max())
    mo.callout(
        mo.md(
            f"## Verdict — window {label}\n\n"
            + (
                f"**Gate PASSES.** `disp_n` varies across firms (sd up to **{_sd_deep:.3f}** at "
                f"n >= {GATE_N_MIN}) and tracks article flow at `|rho|` = **{gate_rho:.3f}**. The "
                f"contamination is not uniform, so it can tilt the cross-section and correcting it "
                f"can change the result. Proceed to `novelty_ceiling_normalised.py`.\n"
                if _passed
                else f"**Gate FAILS.** `|rho|` = **{gate_rho:.3f}** against article flow at "
                f"n >= {GATE_N_MIN}, below the {GATE_RHO} threshold. `disp_n` is real and grows with n, "
                f"but it does not vary across firms in a way that lines up with firm characteristics — "
                f"so it shifts the level of the score without tilting the cross-section. **The "
                f"correction cannot change a cross-sectional result. Stop here.**\n"
            )
        ),
        kind="success" if _passed else "warn",
    )
    return


if __name__ == "__main__":
    app.run()
