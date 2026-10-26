"""`nu_tilde` as a column — the vectorised form of :mod:`honey.novelty_score_exp.core.calibration`.

§sec:calib calibrates each article against the raw novelty of its own reference set. Done
article by article that is a nested scan: for every `d`, find `R_c(d;w)` again and rank `d`
within it. This module does it once per `(company, window, k)` instead.

**`nu_tilde` depends only on `(w, k)`** — never on the aggregation route and never on `theta`.
So it is computed as a *column*, beside the raw one, and every `theta` in a sweep becomes a
comparison against a number already on disk. Calibrating inside `filtered` and `weighted` would
implement one quantity twice and recompute it for every point of the grid.

**This reads the score column, not the vectors.** Nothing here embeds anything or touches the
store: the raw novelties are the scorer's output, and calibration is a second pass over them.

## Why a sliding structure rather than a slice per article

Articles arrive in publication order and `start` is non-decreasing, so consecutive reference
sets overlap almost entirely — at a 336h window on the densest company, article `i` and `i + 1`
differ by a handful of members out of 48,561. Re-scanning the slice for each article throws
that away and costs one comparison per `(article, member)` pair.

Measured on `company_barclays.jsonl`, 2,324,613 articles, the pair count is the whole story:

| window | median candidates | pairs | re-scanning |
|---|---|---|---|
| 24h | 3,592 | 8.2e9 | 10.8 s |
| 168h | 24,283 | 54.0e9 | 54.8 s |
| 336h | 48,561 | 106.6e9 | 110.5 s |

Keeping **one** window and sliding it replaces that with `O(n log u)`: ~56 M operations instead
of 106.6 bn at 336h. The structure is a Fenwick tree over the distinct novelty values, which is
what makes "how many members are at or below this score" a prefix sum rather than a scan.

A range join is the wrong shape here and would be far worse, not better: it materialises the
pair table, and at 336h that is 106.6 bn rows for one company. The sliding form materialises
nothing.
"""

from __future__ import annotations

import math

import numpy as np

from honey.novelty_score_exp.core.calibration import EMPTY_REFERENCE
from honey.novelty_score_exp.core.novelty import UNEMBEDDED


class _Fenwick:
    """Counts over value buckets, with prefix sums — a binary indexed tree.

    One instance holds the articles currently inside the window: :meth:`add` enters or leaves
    one, :meth:`at_or_below` asks how many of them sit at or under a given bucket. Both are
    `O(log u)` in the number of distinct novelty values.

    Backed by a plain Python list rather than a numpy array on purpose. Every operation here is
    a scalar read-modify-write from Python, where numpy's per-element indexing overhead
    dominates and a list is several times faster.
    """

    __slots__ = ("tree",)

    def __init__(self, size: int) -> None:
        """Make an empty tree over *size* buckets.

        Args:
            size: Number of distinct values to count over.
        """
        self.tree = [0] * (size + 1)  # 1-indexed; index 0 is unused

    def add(self, bucket: int, delta: int) -> None:
        """Record an article entering (`delta = 1`) or leaving (`delta = -1`) the window.

        Args:
            bucket: The article's value bucket, 0-based.
            delta: +1 to enter, -1 to leave.
        """
        tree, i = self.tree, bucket + 1
        n = len(tree)
        while i < n:
            tree[i] += delta
            i += i & -i

    def at_or_below(self, bucket: int) -> int:
        """How many articles in the window sit at or below *bucket*.

        The `<=` of §sec:calib's indicator: equal values share a bucket, so including the
        bucket itself is what makes a tie count toward the share.

        Args:
            bucket: The query's value bucket, 0-based.

        Returns:
            The count.
        """
        tree, i, total = self.tree, bucket + 1, 0
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total


def calibrate_column(times: np.ndarray, nu: np.ndarray, window_s: float) -> np.ndarray:
    """`nu_tilde` for one company's articles, in publication order.

    Equivalent to calling :func:`honey.novelty_score_exp.core.calibration.calibrated_novelty`
    on every article, and pinned to it by ``tests/test_novelty_calibration.py``. The window is
    the same half-open interval §2.4 writes and
    :func:`honey.novelty_score_exp.engine.scorer.knn_raw_novelty` computes — `start` is the first
    index still inside it — so an article never ranks against itself.

    Unembedded articles carry NaN rather than a score, and NaN enters neither the numerator nor
    the denominator: such a member is never added to the tree, so it is not counted among those
    at or below, and not counted in the window total either. That is §2.4's divisor question
    resolved as `core` resolves it — a member with no novelty of its own cannot be ranked
    against — and it is a reading of the raw column, not a second judgment about what counts as
    embedded.

    Args:
        times: `(n,)` UTC epoch seconds, ascending — the scorer's own time array.
        nu: `(n,)` raw novelty, NaN where the article was never embedded.
        window_s: The lookback `w` in seconds, matching the run that produced *nu*.

    Returns:
        `(n,)` float32 calibrated novelty. NaN wherever *nu* is NaN — an article with no raw
        score has nothing to calibrate — and :data:`EMPTY_REFERENCE` where no scoreable
        antecedent is in reach.
    """
    n = len(nu)
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return out

    # the vectorised form of core.novelty.is_scored, which is the definition; the equivalence
    # test is what keeps the two agreeing. Branches on the sentinel for the same reason the
    # scalar form does — a hardcoded isnan would stop matching if UNEMBEDDED ever changed, and
    # the branch is per column rather than per element, so it costs nothing.
    scoreable = ~np.isnan(nu) if math.isnan(UNEMBEDDED) else nu != UNEMBEDDED
    # bucket by value, so "at or below nu[i]" is a prefix sum. Equal values share a bucket,
    # which is what makes `<=` rather than `<` fall out of the structure.
    distinct = np.unique(nu[scoreable])
    buckets = np.zeros(n, dtype=np.int64)
    buckets[scoreable] = np.searchsorted(distinct, nu[scoreable])
    start = np.searchsorted(times, times - window_s, side="left")

    # Python-level access throughout: these loops are scalar work, where list indexing beats
    # numpy's per-element overhead by several times.
    valid = scoreable.tolist()
    bucket = buckets.tolist()
    first = start.tolist()
    shares = [np.nan] * n

    tree = _Fenwick(len(distinct))
    lo = hi = 0  # the window currently in the tree is [lo, hi)
    members = 0
    for i in range(n):
        while hi < i:  # everything before i has become eligible
            if valid[hi]:
                tree.add(bucket[hi], 1)
                members += 1
            hi += 1
        while lo < first[i]:  # `start` is non-decreasing, so this only ever moves forward
            if valid[lo]:
                tree.add(bucket[lo], -1)
                members -= 1
            lo += 1
        if valid[i]:
            shares[i] = EMPTY_REFERENCE if members == 0 else tree.at_or_below(bucket[i]) / members

    out[:] = shares
    return out
