"""Calibrated novelty, `nu_tilde = G_hat_c(nu)` — §sec:calib of `Novelty Paper.md`.

A raw novelty is not comparable across companies or across time. The scale a company's scores
occupy depends on how densely it is covered: a thinly covered name's reference set holds few
and distant antecedents, so its raw novelty sits high, while a heavily covered one is compared
against a crowd and sits low. A single absolute cutoff `theta` therefore means a different
thing for each name and each week, which is what §sec:calib replaces with a relative boundary.
@montague2001 state the general case — raw scores from different sources are not on a
comparable scale and must be normalised before being combined or thresholded — and
@manmatha2001 map a raw score to its position within a fitted distribution, the mechanism used
here.

Write `G_hat_c` for the empirical distribution of raw novelty over `d`'s reference set,

    G_hat_c(u) = (1 / |R_c(d)|) * sum_{d' in R_c(d)} 1{ nu_c(d') <= u }

and set `nu_tilde_c(d) = G_hat_c(nu_c(d))` — **the share of that reference set whose own
novelty does not exceed `d`'s**.

Two properties follow directly, and both are tested rather than assumed:

- `nu_tilde` takes values in multiples of `1 / |R_c(d)|`. It is a rank, so its resolution is
  set by how many antecedents there are to rank against.
- `nu_tilde` is **invariant under any strictly increasing transformation of `nu`**. Only the
  ordering of the raw scores enters, never their spacing. That is the property that makes the
  calibration worth doing — it is what severs `theta` from whatever scale the embedding model
  happens to produce — and it is the strongest test available for this module.

**This reads a score column, not vectors.** Each member of `R_c(d;w)` needs its own `nu`,
computed against *its* own window, so calibration is a second pass over the output of
:mod:`honey.novelty_score_exp.core.novelty` — no re-embedding and no re-scoring.
:mod:`honey.novelty_score_exp.engine.calibrate` does it once per `(window, k)` and hands back a
column; calibrating inside each aggregation route would implement one quantity twice and
recompute it for every `theta` in a sweep.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Mapping, Sequence

from honey.novelty_score_exp.core.novelty import EMPTY_REFERENCE, UNEMBEDDED, is_scored
from honey.novelty_score_exp.core.reference_set import reference_set
from honey.novelty_score_exp.core.representation import Article


def _calibrate(nu: float, reference_novelties: Sequence[float]) -> float:
    """`G_hat_c(nu)` — the share of *reference_novelties* that do not exceed *nu*.

    **Private: the arithmetic alone.** :func:`calibrated_novelty` is the entry point, and it is
    the one that knows which scores these are. Called directly, nothing ties *reference_novelties*
    to `R_c(d;w)` — any list of numbers would be accepted — which is exactly the looseness the
    public signature was changed to remove, so the underscore is load-bearing rather than
    stylistic. This is to :func:`calibrated_novelty` what :func:`functions.top_k_mean` is to
    :func:`novelty.raw_novelty`: a formula its caller supplies the meaning for.

    **NaN members are dropped here, in both numerator and denominator.** NaN is
    :data:`novelty.UNEMBEDDED`: a member whose title was never embedded has no novelty of its
    own, so it cannot be ranked against. That resolves §sec:calib's `1 / |R_c(d)|` the way
    :func:`novelty.raw_novelty` resolves §2.4's `n_c(d)`, and it is a *reading* of the raw column
    rather than a second judgment about what counts as embedded — that decision was made once,
    when the score was produced.

    Owning the rule here rather than requiring a filtered sequence is what lets the scalar and
    vectorised forms be compared like with like: both take the reference set as it is.

    The comparison is `<=`, not `<`. An article tying with every member of its reference set
    calibrates to 1.0 rather than 0.0, which is the right way round: ties mean `d` is at least
    as novel as everything it is ranked against.

    Args:
        nu: The raw novelty of the article being calibrated, `nu_c(d)`.
        reference_novelties: The raw novelty of each member of `R_c(d;w)`, each computed
            against its own window, **NaN included**. Order is irrelevant — only counts matter.

    Returns:
        The share in `[0, 1]`, in multiples of one over the number of scoreable members;
        :data:`EMPTY_REFERENCE` where there is nothing to rank against.
    """
    scores = [s for s in reference_novelties if is_scored(s)]
    if not scores:
        return EMPTY_REFERENCE
    return sum(s <= nu for s in scores) / len(scores)


def calibrated_novelty(
    article: Article,
    company: str,
    window: dt.timedelta,
    corpus: Iterable[Article],
    raw: Mapping[str, float],
) -> float:
    """`nu_tilde_c(d)` — *article* ranked against the raw novelty of its own reference set.

    **The reference set is built, not supplied.** The first four arguments are deliberately the
    ones :func:`novelty.raw_novelty` takes, and for the same reason: `nu_tilde` ranks `d` within the
    very set `nu` was computed over, so the arguments that determine that set must be the same
    arguments. Calling :func:`reference_set.reference_set` rather than accepting a list of
    scores means the window, the half-open bound and §2.4's condition (2) are inherited from the
    one place that defines them, instead of being restated by every caller.

    The fifth argument is where this differs from `novelty`, and it is the whole character of
    §sec:calib: calibration reads a **score column**, not vectors. Each member needs its own
    `nu`, already computed against its own window, so this is a second pass over the scorer's
    output — nothing here embeds anything or touches the store.

    `k` is absent on purpose. `nu_tilde` depends only on `(w, k)`, and `k` is already fixed by
    *which column* `raw` was read from; taking it as well would be an unused argument free to
    contradict the data. The cost is that `raw` must come from one column of one run, which
    nothing here can check.

    Args:
        article: The article being calibrated, `d`.
        company: The company it is calibrated for, `c`.
        window: The lookback, `w`. Must match the run that produced *raw*.
        corpus: The articles `R_c(d;w)` is drawn from.
        raw: `article_id` -> raw novelty, one column of a scoring run. A missing id reads as
            NaN, which is the same answer an unembedded article already carries, so a gap in
            the column and a gap in the store are handled alike.

    Returns:
        The calibrated novelty, or NaN where *article* has no raw score of its own — there is
        nothing to place in a distribution.
    """
    nu = raw.get(article.article_id, UNEMBEDDED)
    if not is_scored(nu):
        return UNEMBEDDED
    members = reference_set(article, company, window, corpus)
    return _calibrate(nu, [raw.get(member.article_id, math.nan) for member in members])
