"""The candidate family `f` — §2.5 of `Novelty Paper.md`.

§2.5 defines, for a neighbour count `k`,

    nu^(k)_c(d) = 1 - (1 / min{k, n_c(d)}) * sum_{j=1}^{min{k, n_c(d)}} s_{c,(j)}(d)

where `s_{c,(j)}(d)` is the `j`-th largest similarity between `d` and the members of
`R_c(d;w)`. At `k = 1` this is @allan1998's nearest-neighbour score — `d` against the single
member most like it, no other member entering. At `k > 1` it is the top-`k` mean, measuring how
much of what `d` reports is carried by its `k` closest antecedents together; @vuurens2016 found
the mean over three neighbours detected first stories better than the distance to one.

**Every `f` here is pure maths under a precondition.** The vectors it is handed are all real
embeddings and the reference set is non-empty, because
:func:`honey.novelty_score_exp.core.novelty.raw_novelty` has already established both. That is what
keeps these functions to their arithmetic: the unembedded query, the unembedded member, the
empty set and the clip back into `[0, 1]` are properties of the composition, not of any
candidate, so they are written once in the wrapper rather than once per `f`. Adding N3 or N5-N7
therefore costs only the formula.

**No `centroid`.** Appendix 12 shows the centroid score equals `nu^(k)` up to a multiplicative
factor, so it is a reading of the top-`k` mean rather than a candidate beside it.

The paper writes the family as `nu^(k)` while Appendices 11 and 12 switch to `n`. The code uses
`k` throughout — `n_c(d)` is the *size* of the reference set in §2.4, so reusing it for the
neighbour count would collide.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import numpy as np

from honey.novelty_score_exp.core.similarity import similarities

#: What a candidate function looks like: query vector, reference vectors, neighbour count.
#:
#: Both arrays are real embeddings and ``reference_vectors`` is non-empty — see the module
#: docstring on why that is the wrapper's guarantee rather than each function's problem.
Candidate = Callable[[np.ndarray, np.ndarray, int], float]


def top_k_mean(x: np.ndarray, reference_vectors: np.ndarray, k: int) -> float:
    """`nu^(k)` — one minus the mean of the `k` largest similarities.

    Takes **vectors rather than similarities**, as §2 does: the function owns the comparison, so
    a candidate that needs something other than a plain cosine ordering can have it.

    The divisor `min{k, n_c(d)}` is not a rule this function applies. It is what taking the last
    `k` of a shorter array already means, which is why there is no branch for a reference set
    smaller than `k`. It reads as `min(k, #embedded members)` rather than `min(k, |R_c(d;w)|)`
    because the wrapper drops unembedded members before calling — the resolution §2.4 leaves
    open, decided in one place.

    Sorts in full. :mod:`honey.novelty_score_exp.engine.scorer` partitions once at `max(k)`
    instead, which is a ~13x speedup at scale and the reason a wide grid of `k` is nearly free;
    the two must agree, which is what the equivalence test is for.

    Args:
        x: The query article's embedding, `(dim,)`. A real embedding, not a zero row.
        reference_vectors: `(n, dim)` embeddings of `R_c(d;w)`, all real and `n >= 1`.
        k: The neighbour count. `k >= 1`, which :func:`novelty.raw_novelty` enforces rather than
            this function — see the module docstring on why a candidate validates nothing.

    Returns:
        `1 - mean of the min(k, n) largest similarities`. Can fall a little outside `[0, 1]`
        in float32 — the wrapper clips, this does not.
    """
    ordered = np.sort(similarities(x, reference_vectors))
    return 1.0 - float(np.mean(ordered[-min(k, len(ordered)) :]))


#: Every candidate `f`, by the name a scoring run records it under (``KNNParams.f``).
#:
#: **The engine keeps a registry with exactly these keys**,
#: :data:`honey.novelty_score_exp.engine.scorer.FAMILY`, and the equivalence test is
#: parametrised over this one — so a candidate added here without an engine form, or with one
#: that disagrees, fails the suite rather than a run. Read-only, so nothing can register a
#: candidate at runtime that the tests never saw.
FAMILY: Final[Mapping[str, Candidate]] = MappingProxyType({"top_k_mean": top_k_mean})

#: The candidate a run uses unless it names another; the one every published figure used.
DEFAULT_F: Final = "top_k_mean"
