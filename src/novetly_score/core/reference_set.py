"""The prior coverage an article is scored against — §2.4 of `Novelty Paper.md`.

Every novelty score in the paper is relative to a reference set, and §2.4 constrains it twice:

    R_c(d;w)  subset of  { d' : t(d') < t(d) }        (1) computable at t(d)
    R_c(d;w)  subset of  { d' : c in C(d') }          (2) both are accounts of c

(1) is what makes the score usable live — no article published later can enter it. (2) is what
makes every similarity in it a comparison between two accounts of the same company.

Within those, §2.4 argues the set should be **contiguous in time** rather than a sample:
@leskovec2009 found phrase volume decaying exponentially with a half-life of roughly eight
hours, so the `k` most recent articles are the `k` most likely antecedents for every `k`, and a
set with a gap is improved by closing it and dropping its oldest members instead. Bounding by a
duration `w` gives that for free.

What remains open is how far back `w` should reach, and §2.4 frames it as a two-sided error. Too
short and every antecedent predates the set, so a restatement is compared only against articles
carrying none of what it repeats and scores as novel — a false positive. Too long and the set
reaches the previous instance of a recurring event: a quarterly report is more similar to last
quarter's than to anything else about the company, so a report repeating nothing is scored as a
restatement — a false negative.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Sequence

from honey.novelty_score_exp.core.representation import Article


def mentions_company(article: Article, company: str) -> bool:
    """§2.4's condition (2) — whether *article* is an account of *company*.

    Spelled out as a predicate rather than left to the data's shape. In the dump it holds by
    construction, because articles arrive in per-company files; a merged corpus would drop it
    silently, and nothing downstream would notice that novelty had started comparing an account
    of one company against an account of another.

    **Membership only — tone plays no part.** Condition (2) is `c in C(d')`, so an article that
    names *company* without characterising it is still a valid comparison: it is an account of
    the company, which is all a candidate has to be. :func:`counts.counts` applies a second,
    narrower rule to the articles it reduces — it skips a pair whose :func:`counts.tone` is None
    — and the two must not be conflated. A tone-less pair contributes to no bucket in §3 while
    remaining in `R_c(d;w)` in §2.4.

    Args:
        article: The candidate article.
        company: The company being scored for.

    Returns:
        Whether the article mentions the company.
    """
    return company in article.mentions


def _accounts_of(
    company: str,
    corpus: Iterable[Article],
    within: Callable[[dt.datetime], bool],
) -> Sequence[Article]:
    """The accounts of *company* published at an instant *within* admits, in publication order.

    Every set in this module is that sentence with a different *within*, so the two rules they
    share live here: :func:`mentions_company` decides what counts as an account, and the result
    is ordered by publication time. Holding them once is what stops `R_c(d;w)` and `A_{c,t}`
    from drifting apart — they are different intervals over the same corpus, not different
    notions of membership.

    The interval itself stays at the call site. It is the only thing §2.4 and §3 disagree about
    and the thing each docstring argues over, so a reader checking either against the paper
    should find the bound written out rather than passed in.

    Args:
        company: The company the set is about, `c`.
        corpus: The articles to draw from.
        within: Whether a publication instant falls in the set's interval.

    Returns:
        The members, in publication order. Empty where nothing qualifies.
    """
    return sorted(
        (candidate for candidate in corpus if within(candidate.published) and mentions_company(candidate, company)),
        key=lambda candidate: candidate.published,
    )


def reference_set(
    article: Article,
    company: str,
    window: dt.timedelta,
    corpus: Iterable[Article],
) -> Sequence[Article]:
    """`R_c(d;w)` — the prior accounts of *company* within *window* of *article*.

    The interval is **half-open**, `t(d) - w <= t(d') < t(d)`, exactly as §2.4 writes it. Two
    consequences of the strict upper bound worth knowing: *article* cannot be its own
    antecedent, and neither can anything published at the same instant. Simultaneous publication
    is common in syndicated feeds, so that excludes some genuine antecedents — but admitting
    them would make the score depend on tie order, which is not a property of the corpus.

    No candidate cap. §2.4 bounds the set by time alone; the cap in
    :class:`honey.novelty_score_exp.engine.params.KNNParams` exists to make a run fit and is not
    part of the definition.

    Members with no embedding are **kept**. Dropping them is
    :func:`honey.novelty_score_exp.core.novelty.raw_novelty`'s job, because that is where the
    divisor is decided, and doing it here would leave two places that know the rule.

    Args:
        article: The article being scored, `d`.
        company: The company being scored for, `c`.
        window: The lookback duration, `w`.
        corpus: The articles to draw from.

    Returns:
        The members, in publication order. Empty where nothing qualifies.
    """
    lower = article.published - window
    return _accounts_of(company, corpus, lambda ts: lower <= ts < article.published)


def daily_set(company: str, day: dt.date, corpus: Iterable[Article]) -> Sequence[Article]:
    """`A_{c,t}` — the articles about *company* published on *day*, §3.

    **Day means the UTC calendar date of publication.** §3 defines `t` as the interval from the
    previous trading day's close to its own, which is not what this does and not what produced
    any published figure. The deviation is deliberate and recorded: weekends are retained here
    and through the smoothing, then dropped where the signal meets the price panel, so the EW
    recursions run on an unbroken daily grid. Adopting §3's reading would roll weekend and
    holiday coverage into the next trading day and move ~8% of matches.

    Args:
        company: The company, `c`.
        day: The UTC calendar date, `t`.
        corpus: The articles to draw from.

    Returns:
        The members, in publication order.
    """
    return _accounts_of(company, corpus, lambda ts: ts.date() == day)


def echo_set(
    article: Article,
    company: str,
    horizon: dt.timedelta,
    corpus: Iterable[Article],
) -> Sequence[Article]:
    """`E_c(d;τ)` — the later articles that reach *article* by repeatedly following π_c, §3.3.

    The one set that looks **forward**, and therefore the one that cannot feed a live signal
    without a lag: §3.3 says outright that `E_c(d;τ)` is known only at `t(d) + τ`. Whatever
    builds this has to own that hazard, because every other set here is causal by construction
    and a reader will assume this one is too.

    It also needs a tie-break that the backward sets do not. π_c is an `argmax` returning an
    *identity* rather than a value, so where two candidates are equally similar the answer
    depends on which is picked — unlike the top-k mean, where ties are value-identical and drop
    out. §3.3 does not specify one.

    Args:
        article: The article whose echo is wanted, `d`.
        company: The company, `c`.
        horizon: The confirmation horizon, `τ`.
        corpus: The articles to draw from.

    Raises:
        NotImplementedError: Always. §3.3 says nothing in the originator route is implemented.
    """
    msg = (
        "echo_set is §3.3's forward set: unimplemented, and it needs both an explicit lag and a "
        "tie-break for π_c's argmax before it can be"
    )
    raise NotImplementedError(msg)
