"""ψ and the similarity on its outputs — §2.3 of `Novelty Paper.md`.

The paper takes embeddings normalised to unit length, under which cosine similarity is just
the inner product:

    s(d, d') = x_d . x_d'          for x_d, x_d' on the unit sphere

That identity is the reason every similarity in the codebase is a dot product and never a
division. It is also a *precondition*, not a fact, so this module checks it rather than
trusting it — see :func:`check_unit_norm`.

An article with no embedding is represented by a zero row, and its similarity to anything is
**NaN, not 0.0**. Zero would read as "unrelated", which is the opposite of the truth: an
unmeasurable article is unknown, and the failure has to stay closed.
"""

from __future__ import annotations

from typing import Final

import numpy as np

#: How far a stored vector may sit from unit norm before it is treated as a defect.
#:
#: Measured over all 11,660,617 vectors in the pilot store: median deviation 1.94e-4, p90
#: 4.00e-4, p99.9 6.28e-4, **worst 8.17e-4**. (Footnote 1 of the paper gives ~2.5e-4, which
#: is close to the median rather than the bound.) 1e-3 clears the measured worst case with
#: room to spare while still catching a vector that is wrong rather than merely imprecise.
NORM_TOL: Final = 1e-3


def is_embedded(vectors: np.ndarray) -> np.ndarray:
    """Which rows carry a real embedding.

    An all-zero row is the store's representation of "this title was never embedded"; no
    genuine embedding is the zero vector, and the pilot store contains no zero rows at all.

    Args:
        vectors: `(n, dim)` array of embeddings, or a single `(dim,)` vector.

    Returns:
        Boolean array of length `n` (or a 0-d array for a single vector).
    """
    return np.einsum("...j,...j->...", vectors, vectors) > 0  # squared norm; cheaper than norm


def check_unit_norm(vectors: np.ndarray, *, tol: float = NORM_TOL) -> None:
    """Raise unless every embedded row is within *tol* of unit norm.

    Deliberately fails rather than renormalising. A vector off unit norm means the store was
    built by something other than the pipeline that is assumed downstream, and silently
    rescaling it would turn a build error into a small, permanent, invisible bias in every
    similarity computed from it.

    Unembedded rows are exempt: they are all-zero by construction, so `‖0‖ = 0` is maximally
    off-tolerance and would otherwise fire on every one of them.

    Args:
        vectors: `(n, dim)` array of embeddings.
        tol: Permitted deviation from 1.0.

    Raises:
        ValueError: If any embedded row deviates by more than *tol*. The message carries the
            offending count and the worst deviation, so the decision to widen the tolerance
            or rebuild the store can be made from the error alone.
    """
    embedded = is_embedded(vectors)
    if not embedded.any():
        return
    norms = np.sqrt(np.einsum("ij,ij->i", vectors[embedded], vectors[embedded]))
    deviation = np.abs(norms - 1.0)
    bad = int((deviation > tol).sum())
    if bad:
        msg = (
            f"{bad:,} of {int(embedded.sum()):,} embedded vectors are further than {tol:g} "
            f"from unit norm (worst {deviation.max():.3e}). The similarity in §2.3 is an inner "
            f"product only under unit norm, so this is a store defect, not a rounding issue."
        )
        raise ValueError(msg)


def similarities(x: np.ndarray, others: np.ndarray) -> np.ndarray:
    """s(d, d') for one article against many — the only place the similarity is defined.

    NaN marks a pair that cannot be compared, in either direction: an unembedded *query* makes
    every similarity NaN, and an unembedded *counterpart* makes its own NaN. Zero would be the
    natural-looking answer and is the wrong one — it reads as "unrelated", and through
    ``1 - s`` that becomes maximum novelty for an article nobody could embed.

    In the normal path :func:`honey.novelty_score_exp.core.novelty.raw_novelty` has already dropped
    unembedded members and returned early on an unembedded query, so a candidate function never
    has to reason about NaN. This stays correct for direct callers regardless.

    Args:
        x: The query article's embedding, ``(dim,)``.
        others: ``(n, dim)`` embeddings to compare against.

    Returns:
        ``(n,)`` similarities, NaN wherever the pair cannot be compared.
    """
    others = np.atleast_2d(others)
    if others.size == 0:
        return np.empty(0, dtype=np.float64)
    if not is_embedded(x):
        return np.full(len(others), np.nan)
    return np.where(is_embedded(others), others @ x, np.nan)


def similarity(x: np.ndarray, y: np.ndarray) -> float:
    """s(d, d') for one pair — §2.1's form of the same thing.

    Delegates to :func:`similarities` rather than repeating the arithmetic. Keeping two
    implementations meant keeping two copies of the unembedded rule as well, and they diverged:
    the pair form returned NaN for an unembedded query while the vectorised form returned 0.0.

    Args:
        x: One article's embedding.
        y: The other's.

    Returns:
        Their inner product, or NaN if either is unembedded.
    """
    return float(similarities(x, np.atleast_2d(y))[0])
