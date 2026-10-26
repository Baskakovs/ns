"""What an article is, and which of its text is compared — §2.2 of `Novelty Paper.md`.

φ is the first of the score's four choices: it reduces an article to the text that goes to the
embedding model. §2.2's argument for taking as little as possible is not about cost. Company
background and standing disclaimers recur across every article about a company whatever that
article reports, so a map admitting them raises the measured similarity of two pieces for
reasons unconnected to whether they carry the same news. @kuntur2026 found the lede the most
original part of an article, with reuse concentrated in the middle and the end, so restricting
to it drops the part most likely to have been copied.

The article model lives here too. φ is a property of an article rather than of the scoring
machinery, and keeping the record beside the maps that read it avoids a circular import between
this module and `counts`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import NamedTuple


class Mention(NamedTuple):
    """One occurrence of a company in one article — a single span in the dump.

    A record rather than a bare tonality because a mention has more to it than the sentiment
    read at it, and the tone rule may come to depend on the rest. The dump stores character
    offsets, a sentence number, and flags for whether the reference was a coreference ("it")
    rather than the company's name and whether it was implied. **Only the tonality is kept so
    far**: the offsets index into a body the dump does not store, and §3's tone rule weighs
    every mention equally whatever kind of reference it was.

    That last part is a modelling decision rather than a given, and this is where revisiting it
    starts. Measured over 4,000 articles: 5.3 mentions per article, 97.7% direct, 2.3%
    coreference, 0% implied. Discounting a pronoun relative to the name would mean carrying
    ``is_coref`` here and reading it in :func:`counts.tone` — a field and a branch, with no
    change to anything that holds a :class:`Mention`.
    """

    sentiment: int | None


#: `C(d)` together with the evidence behind each membership — the shape of
#: :attr:`Article.mentions`.
#:
#: **The invariant: one entry per company, and that entry holds every occurrence of it in the
#: article.** A company is therefore in `C(d)` exactly when it is a key here, and all the
#: evidence for a `(article, company)` pair is reachable in one lookup. Three things follow,
#: and everything in `core/` and `engine/` is entitled to assume them:
#:
#: 1. **A key appears once.** §2.1 treats `C(d)` as a set, so an article carrying two entries
#:    for one company is not a state the paper can express. A sequence of pairs admits it and
#:    resolves it silently — the first wins, the second is invisible — whereas a mapping cannot
#:    represent it at all. The dump supplies one entry per company today, but that is a property
#:    of arriving in per-company files, not of the structure; a merged corpus would break it
#:    without anything noticing. This is the invariant made structural rather than assumed.
#: 2. **An absent key and an empty sequence are different answers.** Absent is "the article does
#:    not mention the company"; empty is "it names the company and says nothing about it". The
#:    second is 8.6% of the feed overall and 54% for the busiest names, so collapsing them is a
#:    large and quiet distortion. :func:`counts.tone` is what reads the distinction, returning
#:    None rather than 0 for the empty case.
#: 3. **The value is a collection because the data is.** A company is referred to more than once
#:    in **48.7%** of pairs — median 1, mean 4.97, max 147 — and the tonalities within a pair
#:    disagree often: 12.1% span {0, +1} and 9.9% span {-1, 0, +1}. So the nesting is the shape
#:    of the data, not incidental structure.
MentionsByCompany = Mapping[str, Sequence[Mention]]


class Article(NamedTuple):
    """An article — §2.1's `d`.

    ``mentions`` is `C(d)` together with the evidence behind each membership, so the companies
    and their evidence are one structure rather than two that have to agree. Its invariant is
    stated once, on :data:`MentionsByCompany`; read it before relying on the shape.

    `o(d)`, the outlet, is absent until §3.3's originator route needs it.

    Note:
        The mapping makes this NamedTuple unhashable, unlike a tuple of fields. Nothing puts an
        article in a set or uses one as a dict key — articles are addressed by ``article_id`` —
        so this costs nothing today, but it is the one thing the change gives up.
    """

    article_id: str
    published: dt.datetime
    title: str
    mentions: MentionsByCompany

    @property
    def companies(self) -> frozenset[str]:
        """`C(d)` — the set of companies this article mentions."""
        return frozenset(self.mentions)

    def mentions_of(self, company: str) -> Sequence[Mention] | None:
        """Every mention of *company* in this article, or None if it does not mention it.

        None and an empty sequence are different answers: None is "not mentioned", empty is
        "mentioned, nothing said about it". §3's tone rule turns on that distinction.

        Args:
            company: The company wanted.

        Returns:
            The mentions, or None where the article does not mention *company*.
        """
        return self.mentions.get(company)


def phi_head(article: Article) -> str:
    """φ_head — the headline.

    §2.2 defines φ_head as "the headline and lede, the first two or three sentences". **This is
    the headline only**, because the corpus cannot supply the lede: `es_metadata` carries no
    body text and its ``teaser`` field is 0% populated. So the implemented map is a strict
    subset of the one the paper specifies, and every published novelty figure is headline-only.
    That is worth stating in §6 rather than leaving to be inferred from `params.json`.

    Args:
        article: The article to represent.

    Returns:
        The text to embed.
    """
    return article.title


def phi_comp(article: Article, company: str) -> str:
    """φ_comp — the sentences mentioning *company*, each with one sentence of context either side.

    Needs the span offsets and the body they index into; the dump stores the offsets but not the
    body, so this cannot be built from `es_metadata` alone.

    Args:
        article: The article to represent.
        company: The company whose sentences are wanted.

    Raises:
        NotImplementedError: Always. See §2.2.
    """
    msg = "phi_comp needs span offsets and the article body; es_metadata stores neither together"
    raise NotImplementedError(msg)


def phi_full(article: Article) -> str:
    """φ_full — the body, with navigation, advertising and standing legal text removed.

    The body dump supplies body text, but 85.8% of its rows are 2016-2019 and only 1.8% fall in
    the window the arms trade, so this is not buildable for the current sample either. §10.7
    places the cost-signal frontier's descent from φ_full deliberately late for that reason.

    Args:
        article: The article to represent.

    Raises:
        NotImplementedError: Always. See §2.2.
    """
    msg = "phi_full needs body text; the body dump reaches only 1.8% of the trading window"
    raise NotImplementedError(msg)
