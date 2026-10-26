"""Measure how far an article-exclusion rule actually moves the production signal.

The novelty and event-exclusion experiments both discard large fractions of the article base and
both come back near-null. This module quantifies why: it pushes each candidate count panel through
the *production* signal chain and compares the result with the all-articles baseline, alongside a
random drop at the same rate as a control.

The control is the point. A filter that displaces the signal no more than random thinning of the
same severity is contributing sampling noise, not information, whatever it does at article level.

    uv run python -m honey.novelty_score_exp.invariance --window 168h --n 5

Findings and the mechanism are written up in `book/marimo/notebooks/PIPELINE.md`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from honey.data.counts import sentiment_ratio
from honey.novelty_score_exp import novelty_knn_model as nk
from honey.strategy.sentiment import _robust_signal

#: Production signal parameters. Taken from `nk.PROD_PARAMS` so the two cannot drift apart.
SPAN: Final[int] = int(nk.PROD_PARAMS["span"])
CLIP_C: Final[float] = float(nk.PROD_PARAMS["clip_c"])
THRESHOLD: Final[float] = float(nk.PROD_PARAMS["threshold"])

#: Minimum names in a cross-section before its rank correlation is worth computing.
MIN_NAMES: Final[int] = 20


@dataclass(frozen=True)
class Displacement:
    """How far one arm's signal sits from the baseline's."""

    arm: str
    kept: float  # share of baseline article mass surviving the rule
    pooled_r: float  # correlation over every (date, name) cell
    rank_rho: float  # median per-date cross-sectional rank correlation
    basket_overlap: float  # mean Jaccard of the `z > THRESHOLD` baskets

    def row(self) -> str:
        """One formatted table row."""
        return (
            f"{self.arm:<32} {self.kept:>7.1%} {self.pooled_r:>9.3f} {self.rank_rho:>9.3f} {self.basket_overlap:>9.1%}"
        )


def baseline_long(scores: pl.DataFrame) -> pl.DataFrame:
    """Count every article, the way production does — the arm every other arm is measured against.

    Args:
        scores: Per-article `(date, entity, sentiment, novelty)` rows.

    Returns:
        Long `(date, entity, positive, neutral, negative)` counts.
    """
    return (
        scores.group_by("date", "entity")
        .agg(
            (pl.col("sentiment") > 0).sum().alias("positive"),
            (pl.col("sentiment") == 0).sum().alias("neutral"),
            (pl.col("sentiment") < 0).sum().alias("negative"),
        )
        .sort("date", "entity")
    )


def article_mass(long: pl.DataFrame) -> int:
    """Total counted articles in a long count panel — the denominator for a retention rate.

    Args:
        long: Long count panel.

    Returns:
        Sum of the three count columns.
    """
    return int(long.select(pl.col("positive").sum() + pl.col("neutral").sum() + pl.col("negative").sum()).item())


def production_signal(long: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Run a count panel through the production signal chain.

    Args:
        long: Long `(date, entity, positive, neutral, negative)` counts.

    Returns:
        The wide signal panel and its entity column names.
    """
    ratio_wide = sentiment_ratio(long)
    cols = [c for c in ratio_wide.columns if c != "date"]
    return _robust_signal(ratio_wide, cols, SPAN, CLIP_C), cols


def compare(
    arm: str,
    kept: float,
    base: pl.DataFrame,
    base_cols: list[str],
    other: pl.DataFrame,
    other_cols: list[str],
) -> Displacement:
    """Compare one arm's signal panel with the baseline's on the dates and names they share.

    Args:
        arm: Label for the arm.
        kept: Share of baseline article mass the arm retained.
        base: Baseline wide signal panel.
        base_cols: Baseline entity columns.
        other: The arm's wide signal panel.
        other_cols: The arm's entity columns.

    Returns:
        The three displacement measures for this arm.
    """
    common = [c for c in base_cols if c in other_cols]
    joined = base.join(other.select("date", *common), on="date", how="inner", suffix="_x")
    a = joined.select(common).to_numpy()
    b = joined.select([f"{c}_x" for c in common]).to_numpy()

    both = np.isfinite(a) & np.isfinite(b)
    pooled = float(np.corrcoef(a[both], b[both])[0, 1]) if both.sum() > 2 else float("nan")

    rhos, overlaps = [], []
    for i in range(a.shape[0]):
        row = np.isfinite(a[i]) & np.isfinite(b[i])
        if row.sum() >= MIN_NAMES:
            x, y = a[i][row], b[i][row]
            rhos.append(np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1])
        in_a, in_b = set(np.flatnonzero(a[i] > THRESHOLD)), set(np.flatnonzero(b[i] > THRESHOLD))
        if in_a or in_b:
            overlaps.append(len(in_a & in_b) / len(in_a | in_b))
    return Displacement(
        arm=arm,
        kept=kept,
        pooled_r=pooled,
        rank_rho=float(np.median(rhos)) if rhos else float("nan"),
        basket_overlap=float(np.mean(overlaps)) if overlaps else float("nan"),
    )


def random_drop(scores: pl.DataFrame, keep: float, seed: int) -> pl.DataFrame:
    """Drop articles uniformly at random, ignoring novelty — the control arm.

    Args:
        scores: Per-article rows.
        keep: Fraction to retain.
        seed: RNG seed, so a reported number can be reproduced.

    Returns:
        Long counts over the surviving articles.
    """
    rng = np.random.default_rng(seed)
    mask = pl.Series("keep", rng.random(scores.height) < keep)
    return baseline_long(scores.filter(mask))


def within_cell_tilt(scores: pl.DataFrame, min_articles: int = 4) -> pl.DataFrame:
    """Novelty difference between positive and negative articles *inside* each `(date, entity)` cell.

    This is the whole alpha budget of the weighting path: if novelty were constant within a cell,
    the weights would factor out of the ratio entirely and only the Laplace prior would notice.

    Args:
        scores: Per-article rows with novelty clipped.
        min_articles: Smallest cell worth measuring.

    Returns:
        Single-row frame of the tilt's mean, sd, median absolute size, and the within-cell sd it
        must be judged against.
    """
    cells = (
        scores.drop_nulls("novelty")
        .group_by("date", "entity")
        .agg(
            n=pl.len(),
            nov_pos=pl.col("novelty").filter(pl.col("sentiment") > 0).mean(),
            nov_neg=pl.col("novelty").filter(pl.col("sentiment") < 0).mean(),
            nov_sd=pl.col("novelty").std(),
        )
        .filter(pl.col("n") >= min_articles)
    )
    diff = (cells["nov_pos"] - cells["nov_neg"]).drop_nulls()
    return pl.DataFrame(
        {
            "cells": [cells.height],
            "tilt_mean": [diff.mean()],
            "tilt_sd": [diff.std()],
            "tilt_abs_median": [diff.abs().median()],
            "within_cell_sd_median": [cells["nov_sd"].median()],
        }
    )


def load_scores(window: str, n: int, emb_dir: str | Path = nk.EMB) -> pl.DataFrame:
    """Per-article novelty scores for one grid cell, clipped to `[0, 1]`.

    Unlike `nk.scores_for` this applies no RIC mapping or universe filter: the comparison here is
    between arms over the same articles, so the universe cancels and the price panel is not needed.

    Args:
        window: Window label, e.g. `168h`.
        n: Neighbour count.
        emb_dir: Directory holding the scores parquets.

    Returns:
        `(entity, date, sentiment, novelty)` rows.

    Raises:
        FileNotFoundError: If no built scores file carries that window label.
    """
    match = next((r for r in nk.nk_manifest(emb_dir) if r[0] == window), None)
    if match is None:
        have = ", ".join(r[0] for r in nk.nk_manifest(emb_dir)) or "none"
        msg = f"no scores parquet for window {window!r}; built: {have}"
        raise FileNotFoundError(msg)
    col = nk.novelty_col(n)
    return (
        pl.read_parquet(match[2], columns=["entity", "date", "sentiment", col])
        .rename({col: "novelty"})
        .with_columns(nk.CLIP_NOVELTY)
    )


def run(scores: pl.DataFrame, thetas: tuple[float, ...], seeds: tuple[int, ...]) -> list[Displacement]:
    """Measure every arm against the baseline: novelty thresholds, weighting, and random controls.

    Args:
        scores: Per-article rows.
        thetas: Novelty thresholds to test.
        seeds: RNG seeds for the random-drop controls, matched to the middle theta's retention.

    Returns:
        One `Displacement` per arm, thresholds first, then weighting, then the controls.
    """
    base_long = baseline_long(scores)
    total = article_mass(base_long)
    base, base_cols = production_signal(base_long)

    arms: list[tuple[str, pl.DataFrame]] = [(f"novelty > {t:g}", nk.threshold_long(scores, t)) for t in thetas]
    arms.append(("novelty-weighted", nk.weighted_long(scores)))

    out: list[Displacement] = []
    control_keep = None
    for label, long in arms:
        kept = article_mass(long) / total
        if label == f"novelty > {thetas[len(thetas) // 2]:g}":
            control_keep = kept
        sig, cols = production_signal(long)
        out.append(compare(label, kept, base, base_cols, sig, cols))

    for seed in seeds:
        long = random_drop(scores, control_keep or 0.25, seed)
        sig, cols = production_signal(long)
        out.append(compare(f"random drop (seed {seed})", article_mass(long) / total, base, base_cols, sig, cols))
    return out


def main() -> None:
    """Report every arm's displacement from the baseline, plus the within-cell novelty tilt."""
    ap = argparse.ArgumentParser(description="Measure signal displacement under article exclusion.")
    ap.add_argument("--window", default="168h", help="window label of a built scores parquet")
    ap.add_argument("--n", type=int, default=5, help="neighbour count column to score on")
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.10, 0.30, 0.50])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--emb", default=nk.EMB)
    args = ap.parse_args()

    scores = load_scores(args.window, args.n, args.emb)
    print(f"{scores.height:,} article rows | null novelty {scores['novelty'].null_count():,}")
    print(f"production chain: span={SPAN}, clip_c={CLIP_C}, threshold={THRESHOLD}\n")
    print(f"{'arm':<32} {'kept':>7} {'pooled r':>9} {'rank rho':>9} {'overlap':>9}")
    for d in run(scores, tuple(args.thetas), tuple(args.seeds)):
        print(d.row())

    tilt = within_cell_tilt(scores).row(0, named=True)
    print(
        f"\nwithin-cell novelty(pos) - novelty(neg) over {tilt['cells']:,} cells: "
        f"mean {tilt['tilt_mean']:+.4f}, sd {tilt['tilt_sd']:.4f}\n"
        f"  median |tilt| {tilt['tilt_abs_median']:.4f} vs median within-cell sd "
        f"{tilt['within_cell_sd_median']:.4f}"
    )


if __name__ == "__main__":
    main()
