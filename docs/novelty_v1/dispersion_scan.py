"""Neighbour-dispersion scan: the hidden factor inside the kNN novelty score.

    uv run src/honey/novelty_score_exp/dispersion_scan.py --help

`novelty_knn_model` scores an article as ``1 - mean(top-n cosine similarities to earlier
same-company headlines)``. With unit-norm embeddings that mean is a dot product against the **mean
vector** of those n neighbours, and a mean vector is shorter than unit length whenever the
neighbours disagree with each other::

    mean_sim   = disp_n * cos(article, mean direction)
    disp_n     = || (1/n) * sum of the n nearest earlier embeddings ||

So the published score carries a factor that has nothing to do with the article being scored::

    raw_score  = 1 - disp_n * cos(...)          <- what novelty_knn_model writes
    norm_score = 1 - cos(...)                   <- scored against the mean *direction*

`disp_n` is exactly 1 at n=1 and falls as n grows, so the contamination grows with n. This module
computes `disp_n` and `norm_score` per (article, n) for one lookback window.

Two implementation notes that matter:

* **Vectors are renormalised on load.** The store is unit-norm only to ~2.5e-4 (256-dim truncation
  plus float32), so without this `disp_1` would not be exactly 1 and the identity above would hold
  only approximately. The cost is that a recomputed `raw_score` differs from
  `novelty_scores_w*.parquet` by ~1e-4.
* **Only `disp` and `norm` are stored.** `raw` is recoverable exactly — ``raw = 1 - disp*(1 - norm)``
  — so writing it too would double the file for no information. `raw_expr` does the algebra.

On cost: the obvious model is ``N_c^2 * W / T`` per company — quadratic in article count — but
measurement contradicted it. Barclays (2.3M articles) took 1.0 min while heidelbergcement (57k) took
15.8 min, and the slow companies were simply the ones scanned *later*. The real constraint is memory:
holding every company's frame until the end drove 17.7 GB of swap on a 26 GB machine, and once the
box thrashes the 11.9 GB memmap costs more than any arithmetic. Hence per-company shards and a
modest thread count; watch RSS, not article counts.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from honey.novelty_score_exp.novelty_knn_model import (
    BLOCK,
    DUMP,
    EMB,
    KNNParams,
    Store,
    discover_dump,
    load_company,
    load_store,
    params_metadata,
    vectors_for,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def disp_col(n: int) -> str:
    """Column holding neighbour dispersion at neighbour count `n`."""
    return f"disp_n{n}"


def norm_col(n: int) -> str:
    """Column holding the dispersion-normalised novelty at neighbour count `n`."""
    return f"norm_n{n}"


def raw_expr(n: int) -> pl.Expr:
    """`raw_score` rebuilt from the two stored columns: ``raw = 1 - disp * (1 - norm)``.

    Exact, not an approximation: `norm = 1 - cos` gives `cos = 1 - norm`, and `raw = 1 - disp*cos`.
    Storing `raw` as well would double the file and add nothing.
    """
    return (1.0 - pl.col(disp_col(n)) * (1.0 - pl.col(norm_col(n)))).alias(f"raw_n{n}")


def unit(vecs: np.ndarray) -> np.ndarray:
    """Renormalise to exactly unit length, leaving all-zero rows (un-embedded titles) at zero."""
    out = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)


def _disp_block(
    lohi: tuple[int, int],
    start: np.ndarray,
    ix: np.ndarray,
    vecs: np.ndarray,
    n_values: tuple[int, ...],
    disp: np.ndarray,
    norm: np.ndarray,
    has_vec: np.ndarray,
) -> None:
    """Dispersion + normalised novelty for one block of articles, written in place.

    Mirrors `novelty_knn_model._score_block`, but takes `argpartition` rather than `partition`
    because the neighbour *vectors* are needed, not just their similarities. One selection at
    max(n) still serves every n: both the similarity prefix sums and the vector prefix sums are
    cumulative, so each n is a lookup at a different cut point.
    """
    i0, i1 = lohi
    c0 = int(start[i0])
    cands = vecs[c0:i1]
    sims = vecs[i0:i1] @ cands.T
    cand_ix = ix[c0:i1][None, :]
    sims[(cand_ix < start[i0:i1, None]) | (cand_ix >= ix[i0:i1, None])] = -np.inf
    sims[:, ~has_vec[c0:i1]] = -np.inf

    k = min(n_values[-1], sims.shape[1])
    rows = np.arange(i1 - i0)[:, None]
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.argsort(-sims[rows, part], axis=1)  # best first, so a prefix is the top-n
    nb = part[rows, order]
    top = sims[rows, nb]

    valid = np.isfinite(top)
    avail = valid.sum(1)
    np.copyto(top, 0.0, where=~valid)  # padding must not move either running sum
    gathered = cands[nb]
    gathered[~valid] = 0.0

    csum_sim = np.cumsum(top, axis=1)
    csum_vec = np.cumsum(gathered, axis=1)

    for j, n in enumerate(n_values):
        take = np.minimum(n, avail)
        keep = (take > 0) & has_vec[i0:i1]
        sel = np.flatnonzero(keep)
        if not sel.size:
            continue
        cut = take[sel] - 1
        mean_sim = csum_sim[sel, cut] / take[sel]
        mean_vec = csum_vec[sel, cut] / take[sel][:, None]
        d = np.linalg.norm(mean_vec, axis=1)
        disp[i0:i1, j][keep] = d.astype(np.float32)
        # d == 0 only if the neighbours cancel exactly; leave those NaN rather than dividing by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            norm[i0:i1, j][keep] = np.where(d > 0, 1.0 - mean_sim / d, np.nan).astype(np.float32)


def dispersion(
    times: np.ndarray,
    vecs: np.ndarray,
    p: KNNParams,
    workers: int | None = None,
    has_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per article: `(disp, norm)`, each `(len(times), len(p.n_values))`, column j for `sorted(n)[j]`.

    An article with no embedding, or with no earlier neighbour in the window, stays NaN in both —
    the same closed-failure convention `knn_novelty` uses, so "unknown" can never read as "novel".
    """
    n_art = len(times)
    n_vals = tuple(sorted(p.n_values))
    vecs = unit(vecs)
    if has_vec is None:
        has_vec = np.einsum("ij,ij->i", vecs, vecs) > 0
    disp = np.full((n_art, len(n_vals)), np.nan, dtype=np.float32)
    norm = np.full((n_art, len(n_vals)), np.nan, dtype=np.float32)
    if n_art < 2:
        return disp, norm

    ix = np.arange(n_art)
    start = np.searchsorted(times, times - p.window_s, side="left")
    if p.max_cand > 0:
        start = np.maximum(start, ix - p.max_cand)
    bounds = [(i0, min(i0 + BLOCK, n_art)) for i0 in range(0, n_art, BLOCK)]

    def do(b: tuple[int, int]) -> None:
        """Score one `(lo, hi)` block in place."""
        _disp_block(b, start, ix, vecs, n_vals, disp, norm, has_vec)

    # 4, not 10: the gather is memory-bandwidth bound and GIL-holding, so extra threads buy
    # little throughput while multiplying peak RSS by the 26 MB per-block buffers.
    with ThreadPoolExecutor(workers or min(4, len(bounds))) as ex:
        list(ex.map(do, bounds, chunksize=8))
    return disp, norm


def scan_company(entity: str, path: str, store: Store, p: KNNParams, workers: int | None = None) -> pl.DataFrame | None:
    """One company's `(entity, date, sentiment, disp_n*, norm_n*)`, or None if it has no articles."""
    arts = load_company(path)
    if arts.is_empty():
        return None
    times = arts["ts"].dt.epoch("s").to_numpy().astype(float)
    vecs = vectors_for(arts["title"].to_list(), store)
    disp, norm = dispersion(times, vecs, p, workers=workers)
    del vecs  # a big company's vectors are ~2.4 GB; do not hold them while building the frame
    n_sorted = sorted(p.n_values)
    return arts.select(
        pl.lit(entity).alias("entity"),
        pl.col("ts").dt.date().alias("date"),
        "sentiment",
        *[pl.Series(disp_col(n), disp[:, j]) for j, n in enumerate(n_sorted)],
        *[pl.Series(norm_col(n), norm[:, j]) for j, n in enumerate(n_sorted)],
    )


def out_path(emb_dir: str, p: KNNParams) -> Path:
    """Where one window's scan lands, named the way `nk_manifest` names its scores files."""
    return Path(emb_dir) / f"dispersion_w{p.window_label}.parquet"


def parts_dir(emb_dir: str, p: KNNParams) -> Path:
    """Per-company shard directory for one window."""
    return Path(emb_dir) / f"dispersion_w{p.window_label}_parts"


def scan(emb_dir: str, dump_dir: str, p: KNNParams, limit: int | None = None, workers: int | None = None) -> Path:
    """Scan every company for one window and write the stamped parquet. Returns the path.

    Results are written as **one shard per company** and only combined at the end, with
    `sink_parquet` streaming the concatenation. The first version held every company's frame in a
    list and concatenated at the end; on 26 GB that drove the machine into swap (17.7 GB of 19.5 GB
    used), and once thrashing began each company took ~16 min regardless of size — a 57k-row company
    was slower than a 2.3M-row one. Sharding caps memory and makes the scan resumable per company,
    which matters because a full window is hours.
    """
    store = load_store(emb_dir)
    files = discover_dump(dump_dir)
    if limit:
        files = dict(list(files.items())[:limit])
    dest, parts = out_path(emb_dir, p), parts_dir(emb_dir, p)
    parts.mkdir(parents=True, exist_ok=True)
    t0, n_art, n_done = time.time(), 0, 0
    print(f"{p.describe()} -> {dest.name}  ({len(files)} companies)", flush=True)
    for i, (entity, path) in enumerate(files.items(), 1):
        shard = parts / f"{entity.replace(':', '_')}.parquet"
        if shard.exists():
            continue
        df = scan_company(entity, path, store, p, workers=workers)
        if df is not None:
            tmp = shard.with_suffix(".parquet.tmp")  # atomic, so a kill never leaves a partial shard
            df.write_parquet(tmp)
            tmp.replace(shard)
            n_art += df.height
            n_done += 1
            del df
        el = time.time() - t0
        print(f"  [{i:>3}/{len(files)}] {entity:<34} {n_art:>12,} new rows  {el / 60:6.1f}m", flush=True)
    lf = pl.scan_parquet(parts / "*.parquet")
    lf.sink_parquet(dest, metadata=params_metadata(p))  # streamed: never materialises the window
    print(f"wrote {dest}  ({(time.time() - t0) / 60:.1f}m, {n_done} new shards)", flush=True)
    return dest


def main() -> None:
    """CLI entry point: one window per invocation, mirroring `novelty_knn_model.main`."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", default=EMB)
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--window-hours", type=float, nargs="+", default=[6, 24, 72, 168, 336])
    ap.add_argument("--n", type=int, nargs="+", default=[1, 3, 5, 7, 10, 20, 50, 70, 100])
    ap.add_argument("--max-cand", type=int, default=0, help="0 = the window alone bounds the lookback")
    ap.add_argument("--limit", type=int, default=None, help="scan only the first N companies (benchmarking)")
    ap.add_argument("--workers", type=int, default=4, help="threads per company; raise only if RAM is free")
    ap.add_argument("--force", action="store_true", help="rescan windows whose parquet already exists")
    args = ap.parse_args()

    n_values: Sequence[int] = tuple(sorted(set(args.n)))
    for wh in args.window_hours:
        p = KNNParams(window_s=wh * 3600, n_values=tuple(n_values), max_cand=args.max_cand)
        dest = out_path(args.emb, p)
        if dest.exists() and not args.force:
            print(f"skip {dest.name} (exists; --force to rescan)", flush=True)
            continue
        scan(args.emb, args.dump, p, limit=args.limit, workers=args.workers)


def scores_for(
    window: str,
    n: int,
    kind: str,
    id_map: dict[str, str],
    universe: list[str],
    emb_dir: str | Path = EMB,
) -> pl.DataFrame:
    """One grid cell as `(entity, date, sentiment, novelty)`, RIC-mapped and universe-filtered.

    The dispersion twin of `novelty_knn_model.scores_for`. `kind` selects which score the `novelty`
    column carries — ``"raw"`` reconstructs the contaminated score exactly via `raw_expr`, ``"norm"``
    takes the dispersion-normalised one. Both come from the *same* file and therefore the same
    article set, so a raw-vs-normalised comparison cannot be confounded by differing coverage.
    """
    from honey.novelty_score_exp.novelty_knn_model import CLIP_NOVELTY

    if kind not in {"raw", "norm"}:
        msg = f"kind must be 'raw' or 'norm', got {kind!r}"
        raise ValueError(msg)
    row = next(r for r in manifest(emb_dir) if r[0] == window)
    d, m = disp_col(n), norm_col(n)
    novelty = pl.col(m) if kind == "norm" else 1.0 - pl.col(d) * (1.0 - pl.col(m))
    return (
        pl.read_parquet(row[2], columns=["entity", "date", "sentiment", d, m])
        .select("entity", "date", "sentiment", novelty.alias("novelty"))
        .with_columns(pl.col("entity").replace(id_map), CLIP_NOVELTY)
        .filter(pl.col("entity").is_in(universe))
    )


def panel_for(
    window: str,
    n: int,
    kind: str,
    id_map: dict[str, str],
    universe: list[str],
    theta: float | None = None,
    emb_dir: str | Path = EMB,
) -> pl.DataFrame:
    """One grid cell straight to production's `(date, entity, positive, neutral, negative)` panel.

    Same semantics as `novelty_knn_model.weighted_long` (``theta=None``) and `threshold_long`
    (``theta`` set), but aggregated *inside* a streaming scan. Building 36 arms by materialising
    38.9M article rows each would move ~28 GB through memory; grouping in the scan keeps every cell
    at panel size (~100k rows), which is what makes a raw-vs-normalised sweep feasible at all.
    """
    if kind not in {"raw", "norm"}:
        msg = f"kind must be 'raw' or 'norm', got {kind!r}"
        raise ValueError(msg)
    row = next(r for r in manifest(emb_dir) if r[0] == window)
    d, m = disp_col(n), norm_col(n)
    novelty = pl.col(m) if kind == "norm" else 1.0 - pl.col(d) * (1.0 - pl.col(m))
    lf = (
        pl.scan_parquet(row[2])
        .select("entity", "date", "sentiment", novelty.alias("novelty"))
        # NaN is not null in polars: unscored articles must be dropped explicitly or they poison
        # every sum below. This is the same trap the check notebook hit.
        .with_columns(pl.col("novelty").fill_nan(None).clip(0.0, 1.0))
        .drop_nulls("novelty")
        .with_columns(pl.col("entity").replace(id_map))
        .filter(pl.col("entity").is_in(universe))
    )
    if theta is None:  # weighting path: each article contributes its own novelty
        agg = (
            pl.col("novelty").filter(pl.col("sentiment") > 0).sum().alias("positive"),
            pl.col("novelty").filter(pl.col("sentiment") == 0).sum().alias("neutral"),
            pl.col("novelty").filter(pl.col("sentiment") < 0).sum().alias("negative"),
        )
    else:  # threshold path: survivors are counted as production counts them
        lf = lf.filter(pl.col("novelty") > theta)
        agg = (
            (pl.col("sentiment") > 0).sum().alias("positive"),
            (pl.col("sentiment") == 0).sum().alias("neutral"),
            (pl.col("sentiment") < 0).sum().alias("negative"),
        )
    return lf.group_by("date", "entity").agg(*agg).sort("date", "entity").collect(engine="streaming")


def manifest(emb_dir: str | Path = EMB) -> list[tuple[str, float, Path, Any]]:
    """Built dispersion parquets as `(window_label, window_hours, path, KNNParams)`, shortest first.

    The dispersion twin of `novelty_knn_model.nk_manifest`, so a notebook discovers what exists
    rather than hardcoding a grid.
    """
    from honey.novelty_score_exp.novelty_knn_model import read_params

    out = []
    for path in Path(emb_dir).glob("dispersion_w*.parquet"):
        prm = read_params(path)
        if prm is not None:
            out.append((prm.window_label, prm.window_hours, path, prm))
    return sorted(out, key=lambda r: r[1])


if __name__ == "__main__":
    main()
