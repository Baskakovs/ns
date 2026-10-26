"""The blocked, threaded scorer — `core.novelty` at the scale of a 38 GB dump.

This is the implementation half of the one intentional duplication in the package.
:mod:`honey.novelty_score_exp.core.novelty` is the readable definition and this is what
actually runs; ``tests/test_novelty_equivalence.py`` is what licenses them to coexist. Without
that test they are two opinions rather than a spec and an implementation, so **a change here
that cannot be justified against `core/` is a bug, whatever it does to the wall clock**.

The numerics below are moved unchanged from ``archive/novelty_v1/novelty_knn_model.py``, which
produced every published figure. Only the names changed: ``KNNParams.n_values`` is now
``k_values``, the cap test is now :attr:`KNNParams.capped`, and the embedded mask arrives from
:class:`honey.novelty_score_exp.engine.store.Lookup` rather than being re-derived here.

**Scoring is selection-bound, not matmul-bound.** Measured per block: GEMM 6.6 ms, masking
43 ms, ``np.partition`` **163 ms**. Three decisions follow from that one measurement, and
together they are a ~13x speedup:

- ``BLOCK = 256`` articles at a time, so a block's candidate range stays in cache;
- a row-major ``(articles, candidates)`` layout, so each row's partition walks contiguous
  memory;
- **one** partition at ``max(k)`` followed by a cumulative sum, so every `k` in the grid comes
  out of a single selection.

The last is why a wide grid of `k` is nearly free, and why any change reintroducing a per-`k`
selection undoes the whole thing. Measure before restructuring.

**`f` is a reducer over that one selection.** :data:`FAMILY` maps each candidate's name to what
it does with a block's sorted top similarities, and its keys are exactly
:data:`honey.novelty_score_exp.core.functions.FAMILY`'s — the equivalence test is parametrised
over the core registry, so a candidate with no engine form, or a disagreeing one, fails there.

**The clip happens here, before calibration**, because that is where `core` applies it:
:func:`core.novelty.raw_novelty` returns the clipped value and :func:`core.calibration` ranks
those. Calibrating the unclipped column ranks near-reprints by float32 noise below zero — two
articles core sees as tied at 0.0. Measured on the 6h run: 12.0% of raw scores land below zero,
7.2% of calibrated values moved when the order was fixed, and 384,405 articles crossed
`theta = 0.30` one way or the other.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import polars as pl

from honey.novelty_score_exp.engine.calibrate import calibrate_column
from honey.novelty_score_exp.engine.corpus import load_company
from honey.novelty_score_exp.engine.params import KNNParams, calibrated_novelty_col, raw_novelty_col
from honey.novelty_score_exp.engine.store import Store, vectors_for

#: Articles scored per block. See the module docstring: chosen against the partition's cost,
#: not the matmul's.
BLOCK: Final = 256

#: Below this many articles the thread pool costs more than the work it saves — score inline.
THREAD_MIN: Final = 8192


#: What a candidate does with one block's selection: ``(top, avail, k_values) -> (block, len(k))``.
#:
#: ``top`` holds each article's best similarities in descending order, zero-padded past
#: ``avail`` real neighbours. Rows with ``avail == 0`` are ignored by the caller, which owns the
#: empty-reference rule — as `core.novelty` owns it for every `f`.
Reducer = Callable[[np.ndarray, np.ndarray, tuple[int, ...]], np.ndarray]


def _top_k_mean(top: np.ndarray, avail: np.ndarray, k_values: tuple[int, ...]) -> np.ndarray:
    """`nu^(k)` for every `k` at once — the vectorised :func:`core.functions.top_k_mean`.

    One cumulative sum serves the whole grid: the mean of the first `min(k, avail)` entries is
    a prefix sum divided by its length.

    Args:
        top: `(block, depth)` similarities, descending, zero past each row's ``avail``.
        avail: `(block,)` real neighbours per article.
        k_values: Neighbour counts, ascending.

    Returns:
        `(block, len(k_values))` float32 raw novelty, NaN where ``avail`` is 0.
    """
    csum = np.cumsum(top, axis=1)  # prefix sums -> every k's mean in one shot
    out = np.full((top.shape[0], len(k_values)), np.nan, dtype=np.float32)
    for j, k_value in enumerate(k_values):
        take = np.minimum(k_value, avail)  # can't average over more neighbours than exist
        rows = np.flatnonzero(take > 0)
        out[rows, j] = 1.0 - (csum[rows, take[rows] - 1] / take[rows]).astype(np.float32)
    return out


#: The engine form of every candidate in :data:`honey.novelty_score_exp.core.functions.FAMILY`,
#: under the same names. Read-only for the same reason the core registry is.
FAMILY: Final[Mapping[str, Reducer]] = MappingProxyType({"top_k_mean": _top_k_mean})


def _score_block(
    lohi: tuple[int, int],
    start: np.ndarray,
    ix: np.ndarray,
    vecs: np.ndarray,
    k_values: tuple[int, ...],
    out: np.ndarray,
    has_vec: np.ndarray,
    reduce: Reducer,
) -> None:
    """Score one block of articles against its candidate range, writing into *out* in place.

    Row-major `(articles, candidates)` so each row's partition walks contiguous memory;
    non-causal and unembedded pairs are masked to `-inf`. One ``np.partition`` at `max(k)`
    selects for every `k`, and *reduce* — the candidate `f` — turns the selection into scores.

    The masking is where §2.4's two conditions are enforced: ``cand < start`` drops anything
    outside the window and ``cand >= ix`` drops anything not strictly earlier, which is the
    half-open interval `core.reference_set` writes out. Condition (2) needs no mask because the
    dump arrives one file per company, so every candidate already mentions the company.

    Args:
        lohi: The block's `(lo, hi)` article range.
        start: Per-article index of the first candidate still inside the window.
        ix: ``arange(n_articles)``, for the strictly-earlier comparison.
        vecs: `(n, dim)` unit-norm embeddings, time-ordered.
        k_values: Neighbour counts, ascending.
        out: `(n, len(k_values))` raw novelty array, written in place.
        has_vec: `(n,)` embedded mask.
        reduce: The candidate's :data:`Reducer`.
    """
    i0, i1 = lohi
    c0 = int(start[i0])  # `start` is non-decreasing, so i0's candidate window opens first
    sims = vecs[i0:i1] @ vecs[c0:i1].T  # unit vectors -> dot = cosine; (block, candidates)
    cand = ix[c0:i1][None, :]
    sims[(cand < start[i0:i1, None]) | (cand >= ix[i0:i1, None])] = -np.inf  # out-of-window / not earlier
    sims[:, ~has_vec[c0:i1]] = -np.inf  # an unembedded candidate is unknown, not dissimilar
    k = min(k_values[-1], sims.shape[1])  # ascending, so reach the deepest k once
    top = -np.sort(-np.partition(sims, -k, axis=1)[:, -k:], axis=1)  # k best per article, descending
    valid = np.isfinite(top)  # counts real neighbours: honours the window *and* the embedding mask
    avail = valid.sum(1)
    np.copyto(top, 0.0, where=~valid)  # -inf padding contributes nothing to a sum
    scored = (avail > 0) & has_vec[i0:i1]  # unembedded queries stay NaN; empty sets stay 1.0
    out[i0:i1][scored] = reduce(top, avail, k_values)[scored]


def knn_raw_novelty(
    times: np.ndarray,
    vecs: np.ndarray,
    p: KNNParams,
    workers: int | None = None,
    has_vec: np.ndarray | None = None,
) -> np.ndarray:
    """Per article, raw novelty under ``p.f`` — before the clip.

    The vectorised form of :func:`honey.novelty_score_exp.core.novelty.raw_novelty` up to its
    final clip, which :func:`score_arrays` applies; it must agree with it. Every rule that
    function owns is upheld here, by different means:

    - **unembedded query -> NaN, never 1.0.** The array is initialised to NaN and only
      ``has_vec`` rows are raised to 1.0, so a title nobody embedded keeps its NaN. A zero
      vector scores cosine 0 against everything, which reads as "nothing similar came before"
      and would award maximum novelty to scraper residue.
    - **unembedded candidate dropped, not counted as dissimilar.** Masked to `-inf` in
      :func:`_score_block`, so it never enters ``avail`` and never enters the divisor — the
      same resolution of §2.4's `n_c(d)` that `core` reaches by filtering the list.
    - **empty reference set -> exactly 1.0**, from the ``nov[has_vec] = 1.0`` initialisation,
      left standing wherever ``take`` is 0.

    The clip into `[0, 1]` is **not** applied here but in :func:`score_arrays`, so that the
    sub-zero values a reprint produces stay observable in a test.

    Args:
        times: `(n,)` UTC epoch seconds, ascending. Naive datetimes shift an hour across DST,
            which is why :mod:`honey.novelty_score_exp.engine.corpus` guarantees the zone.
        vecs: `(n, dim)` unit-norm embeddings in the same order.
        p: The run's parameters.
        workers: Thread count. Defaults to ``min(12, cpu_count)``.
        has_vec: `(n,)` embedded mask. Defaults to "the row is not all-zero".

    Returns:
        `(len(times), len(p.k_values))` float32; column `j` is raw novelty at
        ``sorted(p.k_values)[j]``.
    """
    n_art = len(times)
    k_vals = tuple(sorted(p.k_values))
    if has_vec is None:
        has_vec = np.einsum("ij,ij->i", vecs, vecs) > 0  # squared norm; cheaper than np.linalg.norm
    nov = np.full((n_art, len(k_vals)), np.nan, dtype=np.float32)
    nov[has_vec] = 1.0  # embedded but nothing earlier in the window -> genuinely novel
    if n_art < 2:
        return nov
    ix = np.arange(n_art)
    # first index still inside the lookback window, then the max_cand cap — both non-decreasing in i.
    start = np.searchsorted(times, times - p.window_s, side="left")
    if p.capped:  # uncapped means the window alone bounds the candidate set
        start = np.maximum(start, ix - p.max_cand)
    bounds = [(i0, min(i0 + BLOCK, n_art)) for i0 in range(0, n_art, BLOCK)]

    reduce = FAMILY[p.f]

    def do(b: tuple[int, int]) -> None:
        """Score one `(lo, hi)` block of articles in place."""
        _score_block(b, start, ix, vecs, k_vals, nov, has_vec, reduce)

    if n_art < THREAD_MIN:  # small company: pool setup would dominate
        for b in bounds:
            do(b)
        return nov
    with ThreadPoolExecutor(workers or min(12, os.cpu_count() or 4)) as ex:
        list(ex.map(do, bounds, chunksize=16))
    return nov


def score_arrays(
    times: np.ndarray,
    vecs: np.ndarray,
    p: KNNParams,
    has_vec: np.ndarray | None = None,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One company's raw and calibrated novelty — the whole pipeline `core` defines, in arrays.

    The pair :func:`core.novelty.raw_novelty` and :func:`core.calibration.calibrated_novelty`
    compute article by article, and the unit the equivalence test compares against them, so
    the *composition* is pinned and not only each step: the clip before calibration is exactly
    the kind of rule that two individually correct steps can still get wrong between them.

    Args:
        times: `(n,)` UTC epoch seconds, ascending.
        vecs: `(n, dim)` unit-norm embeddings in the same order.
        p: The run's parameters.
        has_vec: `(n,)` embedded mask.
        workers: Thread count for the kNN pass.

    Returns:
        ``(raw, calibrated)``, each `(n, len(k_values))` float32 with columns in ascending `k`.
        Raw is clipped into `[0, 1]`; NaN wherever the article was never embedded, in both.
    """
    raw = np.clip(knn_raw_novelty(times, vecs, p, workers=workers, has_vec=has_vec), 0.0, 1.0)  # NaN passes
    calibrated = np.empty_like(raw)
    for j in range(raw.shape[1]):
        calibrated[:, j] = calibrate_column(times, raw[:, j].astype(float), p.window_s)
    return raw, calibrated


def score_company(entity: str, path: str | Path, store: Store, p: KNNParams) -> pl.DataFrame | None:
    """One company's file -> `(article_id, entity, date, sentiment, raw_novelty_k*, calibrated_novelty_k*)`.

    **``article_id`` is the column that makes a score joinable** to the mention that produced
    it, which is the grain `sql/` calls ``mentions``. It costs one string column.

    **Both scales are written.** Raw `nu` is the score; calibrated `nu_tilde` is §sec:calib's
    rank of it within its own reference set, and §3.1 and §3.2 both threshold and weight on the
    latter. Carrying them side by side is what lets the calibration's defining property —
    invariance under any strictly increasing transform of `nu` — be checked on real data.

    Unembedded titles stay null, so a missing vector cannot pass for high novelty downstream.
    `fill_nan(None)` is what puts NULL on disk rather than NaN.

    Args:
        entity: The company, e.g. ``company:allianz``.
        path: Its ``company_*.jsonl`` file.
        store: The embedding store.
        p: The run's parameters.

    Returns:
        One row per `(article, company)` pair, or None where the file holds no on-topic
        articles.
    """
    arts = load_company(path)
    if arts.is_empty():
        return None
    times = arts["ts"].dt.epoch("s").to_numpy().astype(float)
    lookup = vectors_for(arts["title"].to_list(), store)
    raw, calibrated = score_arrays(times, lookup.vectors, p, has_vec=lookup.embedded)
    ordered = sorted(p.k_values)
    return arts.select(
        "article_id",
        pl.lit(entity).alias("entity"),
        pl.col("ts").dt.date().alias("date"),
        "sentiment",
        *[pl.Series(raw_novelty_col(k), raw[:, j]).fill_nan(None) for j, k in enumerate(ordered)],
        *[pl.Series(calibrated_novelty_col(k), calibrated[:, j]).fill_nan(None) for j, k in enumerate(ordered)],
    )
