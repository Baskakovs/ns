"""Raw novelty `nu_c(d)` — §2.1 of `Novelty Paper.md`, and the contract every candidate runs under.

§2.1 asks for

    nu_c(d) = f(x_d, {x_{d'} : d' in R_c(d)}) in [0, 1]

small when `d` restates prior coverage of `c` and large when it introduces something unlike it.
Reaching that number is four choices, and this module is where they compose:

    phi   which text represents the article        §2.2, core.representation
    psi   how that text becomes a vector           §2.3, supplied by the caller
    R     what the article is compared against     §2.4, core.reference_set
    f     how the similarities become one number   §2.5, core.functions

**This is the single owner of the rules a score has to uphold.** None of them is a property of
any `f` — they are properties of the composition — so writing them here is what lets every
candidate be its arithmetic and nothing else:

    reject k < 1                                              §2.5, k in N
    build R_c(d;w)                                            §2.4
    represent and embed the query
      query has no vector    -> NaN, return                   never 1.0
    embed the members, drop those with no vector              unknown, not dissimilar
      nothing left           -> 1.0, return                   §2.5, empty reference set
    nu = f(x_d, vectors, k)                                   f sees only valid, non-empty input
    clip nu into [0, 1]                                       §2.1's codomain

Three consequences are worth naming, because each is a rule that stops being a rule.

**The divisor question disappears.** §2.4 writes `n_c(d) = |R_c(d;w)|`, which leaves open what
to do about a member whose title was never embedded. Dropping them *before* the call makes
`min(k, len(vectors))` mean `min(k, #embedded)` automatically, so no candidate has to know.

**The empty-set rule is checked after the drop, not before.** A reference set of three articles
none of which is embedded is empty for this purpose, and must score 1.0 by the same reasoning
§2.5 gives for a genuinely empty set: no earlier account of `c` is within reach.

**The clip sits outside `f`.** Cosine in float32 exceeds 1.0 for a near-exact reprint — measured
at 1.000790 on a real stored vector, so `1 - s` is -0.00079 — and an unclipped negative is
worse than cosmetic: through §3.2's weight route it *subtracts* from a sentiment bucket, and any
fractional power of it is NaN, which poisons a whole day's counts silently.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Iterable
from typing import Final

import numpy as np

from honey.novelty_score_exp.core.functions import DEFAULT_F, FAMILY, Candidate
from honey.novelty_score_exp.core.reference_set import reference_set
from honey.novelty_score_exp.core.representation import Article, phi_head
from honey.novelty_score_exp.core.similarity import is_embedded

#: What an article with no embedding scores.
#:
#: **NaN, never 1.0.** A missing vector means the title was never embedded, which says nothing
#: about whether the article repeats anything. 1.0 would say the opposite — maximum novelty —
#: and would hand the strongest possible signal to scraper residue and parse failures. 39,077
#: rows of the pilot store are in this state.
UNEMBEDDED: Final = math.nan


#: What an article with no reachable prior coverage scores.
#:
#: §2.5 fixes this at exactly 1.0: with `R_c(d;w)` empty, no earlier account of `c` is within
#: reach, so nothing can have been restated. Reached when the window holds nothing, and equally
#: when it holds only articles whose titles were never embedded.
EMPTY_REFERENCE: Final = 1.0


def is_scored(nu: float) -> bool:
    """Whether a novelty can be ranked or weighted — False for :data:`UNEMBEDDED`.

    **The one definition of "this article has a usable novelty".** Everything downstream of the
    score asks this question: :func:`calibration._calibrate` to decide what is in
    `G_hat_c`'s divisor, :func:`aggregation.scoreable` to decide what a day's counts are taken
    over, and :func:`engine.calibrate.calibrate_column` through its vectorised `~isnan` sibling,
    which the equivalence test pins to this one. Three call sites, one rule.

    **Not the same question as :func:`similarity.is_embedded`, and they must not be merged.**
    That one reads *vectors* and asks whether the store holds a real row; this one reads
    *scores* and asks whether a novelty came out. The first produces the second — an unembedded
    article is why a novelty is NaN — but they sit at different stages, and conflating them
    would let a zero vector stand for an unscoreable article, which is how a missing embedding
    turns into maximum novelty rather than no answer.

    **Defined against :data:`UNEMBEDDED` rather than against NaN**, so the sentinel and the
    test for it cannot come apart. Written as a plain ``nu != UNEMBEDDED`` they would: NaN
    compares unequal to itself, so that reads every unscoreable article as scored. The second
    clause is what covers a NaN sentinel, and the first is what keeps this correct if
    :data:`UNEMBEDDED` is ever changed to a value that *does* compare — the dead branch today is
    the one that makes the dependency real rather than decorative.

    Args:
        nu: A raw or calibrated novelty.

    Returns:
        Whether it carries information.
    """
    return not (nu == UNEMBEDDED or (math.isnan(nu) and math.isnan(UNEMBEDDED)))


def raw_novelty(
    article: Article,
    company: str,
    window: dt.timedelta,
    corpus: Iterable[Article],
    embed: Callable[[str], np.ndarray],
    *,
    f: Candidate = FAMILY[DEFAULT_F],
    k: int = 1,
    represent: Callable[[Article], str] = phi_head,
) -> float:
    """Raw novelty `nu_c(d)` — how much of what *article* reports about *company* is already in the window.

    "Raw" as against :func:`honey.novelty_score_exp.core.calibration.calibrated_novelty`, which ranks
    this number within its own reference set. The two are never called plain "novelty" anywhere
    in the package: they share a dtype and a range, so the name is the only thing that says which
    scale a number is on.

    Args:
        article: The article being scored, `d`.
        company: The company it is scored for, `c`. Novelty is a property of the pair: §2.1 is
            explicit that one article can be a first report about one company and a restatement
            about another.
        window: The lookback, `w`.
        corpus: The articles `R_c(d;w)` is drawn from.
        embed: `psi` — text to unit-norm vector. A zero row means "never embedded"; it is the
            caller's store, so `core` stays free of I/O.
        f: The candidate, `nu^(k)`. Defaults to :data:`functions.FAMILY`'s default entry; the
            engine selects the same function by name, through ``KNNParams.f``.
        k: The neighbour count. `k = 1` is the nearest-neighbour score.
        represent: `phi`. Defaults to :func:`representation.phi_head`, the only one the corpus
            can supply — `phi_comp` needs span offsets with a body, `phi_full` needs body text
            the trading window does not have.

    Returns:
        A novelty in `[0, 1]`; :data:`EMPTY_REFERENCE` where nothing prior is reachable, and
        :data:`UNEMBEDDED` where the article itself has no vector.

    Raises:
        ValueError: If `k < 1`. §2.5 defines the family over `k in N`, and a candidate is pure
            maths under a precondition rather than a validator — `top_k_mean` would take a mean
            over an empty slice and return NaN, which is this module's answer for an unembedded
            article and would be indistinguishable from one. Rejecting here keeps `k`'s domain
            with the other rules no `f` should have to know.
    """
    if k < 1:
        msg = f"k must be at least 1, got {k}: §2.5 defines the candidate family over k in N"
        raise ValueError(msg)

    x = embed(represent(article))
    if not is_embedded(x):
        return UNEMBEDDED

    members = reference_set(article, company, window, corpus)
    vectors = [embed(represent(member)) for member in members]
    embedded = [vector for vector in vectors if is_embedded(vector)]
    if not embedded:
        return EMPTY_REFERENCE

    return float(np.clip(f(x, np.stack(embedded), k), 0.0, 1.0))
