"""The three ways a novelty score reaches a signal — §3.1-3.3 of `Novelty Paper.md`.

§3 reduces a day's coverage of a company to `(P, N)` and then to
`S_{c,t} = (P + 1) / (P + N + 2)`. The three routes below change **only what an article
contributes to that count**, never the count's shape and never `S`:

    production   §3     every article counts 1
    filtered     §3.1   1{nu_tilde >= theta}        a repeated article does not enter at all
    weighted     §3.2   nu_tilde                    every signed article enters, graded
    originator   §3.3   gamma_{c,o(d)}              not implemented

So they are one :func:`counts.counts` call parameterised by a weight rather than three
near-identical reductions. §3.1 and §3.2 differ in what they assume about repetition:
filtering treats it as a boundary — @ke2019 split newswire articles into fresh and stale and
found the return response larger for fresh ones — while weighting treats it as graded, which
@fedyk2023 support in finding prices responded more to articles recombining several earlier
pieces than to reprints of one. A threshold gives a recombination and a reprint the same weight
of zero.

**No route calibrates.** `nu_tilde` arrives as a column, computed once per `(w, k)` by
:mod:`honey.novelty_score_exp.engine.calibrate`. Calibrating inside `filtered` would implement
one quantity twice and recompute it for every `theta` in a sweep; as a column, each `theta` is
a comparison.

**And no route defines `S`.** `core/` hands over `(P, N)` and production's `build_portfolio`
applies the Laplace ratio, the robust signal, the book, the per-name cap and the vol target
unchanged. That division is §10.5's requirement: hold everything but the map from `A_{c,t}` to
`S_{c,t}` fixed, so a difference in the result can only come from the map.

## One corpus for every arm

An article whose title was never embedded has no `nu_tilde`. It therefore cannot pass a
threshold and cannot carry a weight — but §3's production count would still include it, since
tone does not depend on an embedding.

**Every route here drops it, production included.** Otherwise the baseline would trade a
slightly larger corpus than the novelty arms, and a difference in Sharpe would have two
possible causes: the map doing its work, or the arms quietly trading a different sample. The
effect is small — 99.89% of `es_metadata` titles are embedded — but it is not nothing, and the
comparison is the entire point of the exercise. :func:`scoreable` is the shared rule, applied
once, and :func:`production` takes the same `calibrated` column as the others solely so that it
can apply it.
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from honey.novelty_score_exp.core.counts import Counts, counts
from honey.novelty_score_exp.core.novelty import is_scored
from honey.novelty_score_exp.core.reference_set import daily_set
from honey.novelty_score_exp.core.representation import Article


class CorpusShrunk(UserWarning):
    """Raised by :func:`scoreable` when a day loses articles for want of a novelty.

    Its own class so it can be silenced or escalated on its own:
    ``warnings.simplefilter("error", CorpusShrunk)`` to make an unscoreable article fatal, or
    ``"once"`` to keep a full run readable — at 229 companies over two years this fires per
    company-day, which is a lot of lines if every one is printed.

    Worth escalating rather than silencing when a published figure is at stake. The drop is
    supposed to be a rounding error — 99.89% of `es_metadata` titles are embedded — so a day
    that loses a large share is far more likely to be a stale store cache or a missing scores
    column than a genuine gap. Both of those are silent: lookups miss, vectors come back zero,
    novelty comes back NaN, and the arms quietly trade a thinner corpus.
    """


#: What a route contributes for an article that carries no calibrated novelty.
#:
#: Nothing. Such an article is dropped before the count rather than weighted zero, so it is
#: absent from every route alike — see the module docstring on why production drops it too.
NO_CONTRIBUTION: Final = 0.0


def scoreable(
    articles: Iterable[Article],
    calibrated: Mapping[str, float],
    context: str = "",
) -> Sequence[Article]:
    """The articles that carry a calibrated novelty — the corpus every route shares.

    Applied once, by each route, so the four arms are compared on identical samples. An article
    is dropped where *calibrated* has no entry for it or holds NaN; those are the same
    condition reached two ways — a gap in the column, and
    :data:`honey.novelty_score_exp.core.novelty.UNEMBEDDED` recorded in it.

    **Warns whenever it drops anything**, with the count and the share. Dropping is correct —
    see the module docstring on why every route including production does it — but it is also
    the one way this code can quietly change what is being traded, so it should never happen
    silently. A day that loses a few per mille is the expected coverage gap; a day that loses a
    tenth of its articles is a defect somewhere upstream, and the two are indistinguishable
    from the counts alone.

    Args:
        articles: The day's articles, `A_{c,t}`.
        calibrated: `article_id` -> `nu_tilde`, one column of a calibrated run.
        context: What this day is, for the warning — normally `"<company> on <date>"`. A
            warning that cannot say which company-day shrank is most of the way to useless.

    Returns:
        Those with a usable `nu_tilde`, in the order given.

    Warns:
        CorpusShrunk: If any article is dropped.
    """
    day = list(articles)
    kept = [a for a in day if is_scored(calibrated.get(a.article_id, math.nan))]
    dropped = len(day) - len(kept)
    if dropped:
        where = f" for {context}" if context else ""
        warnings.warn(
            f"dropped {dropped} of {len(day)} articles{where} ({dropped / len(day):.1%}): "
            f"no calibrated novelty. Every route drops these, so the arms stay comparable, "
            f"but a large share points at a stale store cache or a missing scores column "
            f"rather than a genuine coverage gap.",
            CorpusShrunk,
            stacklevel=2,
        )
    return kept


def production(
    company: str,
    day: dt.date,
    corpus: Iterable[Article],
    calibrated: Mapping[str, float],
) -> Counts:
    """§3's counts — every article contributes 1. The baseline the other routes are read against.

    Takes *calibrated* without using its values, which looks redundant and is not: it is how
    this route drops exactly the articles the novelty routes cannot score, so all four arms see
    one corpus.

    Args:
        company: The company, `c`.
        day: The UTC calendar date, `t`.
        corpus: The articles `A_{c,t}` is drawn from.
        calibrated: `article_id` -> `nu_tilde`, used only to decide what is scoreable.

    Returns:
        `(P, neutral, N)` for the day.
    """
    return counts(company, scoreable(daily_set(company, day, corpus), calibrated, f"{company} on {day}"))


def filtered(
    company: str,
    day: dt.date,
    corpus: Iterable[Article],
    calibrated: Mapping[str, float],
    theta: float,
) -> Counts:
    """§3.1's `P^theta`, `N^theta` — only articles at or above *theta* are counted.

    The comparison is `>=`, as §3.1 writes it, so an article exactly at the threshold is kept.

    Removing an article entirely rather than discounting it is the route's claim: the tonality
    of a repeated article should not enter the signal at all. That is also its weakness, which
    §3.2 exists to address — a recombination and a verbatim reprint both fall below the line and
    both get weight zero.

    Args:
        company: The company, `c`.
        day: The UTC calendar date, `t`.
        corpus: The articles `A_{c,t}` is drawn from.
        calibrated: `article_id` -> `nu_tilde`.
        theta: The novelty cutoff, in `[0, 1]` on the calibrated scale.

    Returns:
        `(P^theta, neutral, N^theta)` for the day.
    """

    def weight(article: Article, _company: str) -> float:
        """§3.1's summand: 1 where the article clears *theta*, nothing where it does not."""
        return 1.0 if calibrated[article.article_id] >= theta else NO_CONTRIBUTION

    return counts(company, scoreable(daily_set(company, day, corpus), calibrated, f"{company} on {day}"), weight)


def weighted(
    company: str,
    day: dt.date,
    corpus: Iterable[Article],
    calibrated: Mapping[str, float],
) -> Counts:
    """§3.2's `P^w`, `N^w` — every signed article enters, weighted by its novelty.

    The counts become novelty mass rather than article counts, which is why
    :class:`counts.Counts` carries floats. §3.2 notes that this route depends on `nu_tilde` at
    every article while §3.1 depends on it only through whether it clears `theta` — so the
    calibration of §sec:calib bears on this route everywhere and on the filter route only at
    the boundary.

    Args:
        company: The company, `c`.
        day: The UTC calendar date, `t`.
        corpus: The articles `A_{c,t}` is drawn from.
        calibrated: `article_id` -> `nu_tilde`.

    Returns:
        `(P^w, neutral, N^w)` for the day.
    """

    def weight(article: Article, _company: str) -> float:
        """§3.2's summand: the article's own calibrated novelty."""
        return calibrated[article.article_id]

    return counts(company, scoreable(daily_set(company, day, corpus), calibrated, f"{company} on {day}"), weight)


def originator(
    company: str,
    day: dt.date,
    corpus: Iterable[Article],
    calibrated: Mapping[str, float],
    theta: float,
    tau: dt.timedelta,
    min_echoes: int,
) -> Counts:
    """§3.3's `P^s`, `N^s` — only first reports are counted.

    An article originates about `c` when it is novel against what preceded it **and** echoed by
    what follows: `nu_tilde >= theta` and `|E_c(d;tau)| >= min_echoes`. The second condition is
    what the first cannot do alone — translation artefacts, off-topic mentions and poor
    reproductions all score as novel and none is a first report.

    Raises rather than returning, deliberately. A `pass` here would return None, and a None in
    a count panel is an arm that is silently empty rather than one that failed.

    Three things are missing before this can be written, and none is a matter of effort:

    - `E_c(d;tau)` is a **forward** set, known only at `t(d) + tau`, so this route cannot feed
      a live signal without an explicit lag. §3.3 says so outright and then gives the live form
      as a per-outlet origination rate `gamma_{c,o}` used in place of `nu_tilde`.
    - `pi_c` is an `argmax` returning an *identity*, so equally similar candidates need a
      tie-break that §3.3 does not specify.
    - `gamma_{c,o}` needs the outlet, `o(d)`, which
      :data:`honey.novelty_score_exp.engine.corpus.DUMP_SCHEMA` does not currently parse —
      ``source_name`` is in the dump but not in the schema.

    Args:
        company: The company, `c`.
        day: The UTC calendar date, `t`.
        corpus: The articles `A_{c,t}` is drawn from.
        calibrated: `article_id` -> `nu_tilde`.
        theta: The novelty cutoff.
        tau: The confirmation horizon.
        min_echoes: `q`, the minimum size of `E_c(d;tau)`.

    Raises:
        NotImplementedError: Always. See §3.3.
    """
    msg = (
        "originator is §3.3's route: unimplemented. It needs a lag for the forward echo set, a "
        "tie-break for pi_c's argmax, and source_name in engine.corpus.DUMP_SCHEMA for o(d)"
    )
    raise NotImplementedError(msg)
