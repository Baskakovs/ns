"""Tone, and the counts a day's articles reduce to — §3 of `Novelty Paper.md`.

§3 defines, for a company `c` and a trading day `t`:

    A_{c,t} = the articles about c published on t
    P_{c,t} = |{ d in A_{c,t} : tone(d) = +1 }|
    N_{c,t} = |{ d in A_{c,t} : tone(d) = -1 }|
    S_{c,t} = (P + 1) / (P + N + 2)

This module owns `tone`, `P` and `N`. It deliberately does **not** own `S`: the Laplace ratio
is already `honey.data.counts`, which is what production trades, and `build_portfolio` applies
it to the counts handed to it. Restating the formula here would create a second definition of
the one expression the whole comparison rests on.

The three aggregation routes of §3.1-3.3 differ *only* in what each article contributes, so
they are one function here parameterised by a weight — not three near-identical count blocks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Final, NamedTuple

from honey.novelty_score_exp.core.representation import Article, Mention

#: The tonality values §3 admits.
POSITIVE: Final = 1
NEUTRAL: Final = 0
NEGATIVE: Final = -1


class Counts(NamedTuple):
    """`(P, N)` and the neutral bucket, for one `(company, day)`.

    Floats rather than ints because the weight route sums novelty rather than articles (§3.2).
    ``neutral`` is carried because the panel shape production consumes has three columns, though
    `S` ignores it.
    """

    positive: float
    neutral: float
    negative: float


def summed_sentiment(mentions: Sequence[Mention]) -> int | None:
    """Sum the tonalities of every mention of one company in one article.

    Named for what it returns. A mention with no polarity contributes 0 rather than voiding the
    others; a company named with no sentiment-bearing mention at all has no sum, which is not
    the same as a sum of zero.

    Args:
        mentions: Every mention of the company in the article.

    Returns:
        The summed tonalities, or None where the company is named but not characterised.
    """
    if not mentions:
        return None
    return sum(m.sentiment for m in mentions if m.sentiment is not None)


def tone(mentions: Sequence[Mention]) -> int | None:
    """§3's `tone(d)`: the **sign** of :func:`summed_sentiment`.

    One article yields one tone per company however many times it refers to that company, which
    is what makes §3's `P` a count of *articles* rather than of occurrences. An article
    referring to a company three times as positive and once as negative is positive — once.

    None where the company is named but not characterised. That is **not** neutral: it is 8.6%
    of the feed overall and 54% for the busiest names, so returning 0 would be a large and quiet
    distortion.

    Takes the mentions rather than `(article, company)` because they already *are* that pair's
    evidence, and delegates the summation rather than repeating it. Taking the records rather
    than their tonalities is what lets a future rule weigh a coreference differently from the
    company's name without changing this signature — see :class:`representation.Mention`.

    Args:
        mentions: Every mention of the company in the article.

    Returns:
        -1, 0 or +1, or None where the company is named but not characterised.
    """
    total = summed_sentiment(mentions)
    if total is None:
        return None
    if total > 0:
        return POSITIVE
    if total < 0:
        return NEGATIVE
    return NEUTRAL


def counts(
    company: str,
    articles: Iterable[Article],
    weight: Callable[[Article, str], float] | None = None,
) -> Counts:
    """Reduce one company's coverage to `(P, N)`, each article contributing *weight*.

    The single counting function behind all three routes of §3, which differ only in what an
    article contributes. Production passes no weight and every article counts 1; the filter
    route passes `1{nu_tilde >= theta}`; the weight route passes `nu_tilde` itself.

    An article that does not mention *company*, or that names it without characterising it,
    contributes to no bucket. The latter is **not** neutral — a tone-less pair is 8.6% of the
    feed overall and 54% for the busiest names, so counting it as 0 would be a large and quiet
    distortion.

    Args:
        company: The company to count for.
        articles: The articles to reduce, normally one day's.
        weight: What each article contributes, as a function of the pair. Defaults to 1.

    Returns:
        The three buckets.

    Note:
        *weight* is a function of the pair alone, deliberately. The obvious generalisation is to
        pass the tone as well — ``weight(article, company, polarity)`` — which would let an article
        contribute differently by amplitude not only direction.
        Kept out for simplicity until a route needs it; adding the parameter later is a
        signature change here and in the two call sites that build a weight, and nothing else.
    """
    weigh = weight if weight is not None else (lambda _a, _c: 1.0)
    positive = neutral = negative = 0.0
    for article in articles:
        mentions = article.mentions_of(company)
        if mentions is None:
            continue
        polarity = tone(mentions)
        if polarity is None:
            continue
        value = weigh(article, company)
        if polarity > 0:
            positive += value
        elif polarity < 0:
            negative += value
        else:
            neutral += value
    return Counts(positive, neutral, negative)
