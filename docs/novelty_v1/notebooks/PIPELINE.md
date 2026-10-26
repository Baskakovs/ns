# The complete logic: sentiment strategy and novelty scoring

One file for the whole chain, article to position, with every formula, default and file
reference. It exists because the novelty and event-exclusion experiments both keep running into
the same wall — dropping large numbers of articles barely moves the book — and answering *why*
needs the entire pipeline visible at once rather than spread over four modules.

Companion documents: [`NOVELTY.md`](NOVELTY.md) is the novelty experiment's findings,
[`SENTIMENT.md`](SENTIMENT.md) the sentiment book's, and
[`../../../src/honey/novelty_score_exp/CLAUDE.md`](../../../src/honey/novelty_score_exp/CLAUDE.md)
the implementation notes. This file is the specification the three share.

## Notation and grain

| symbol | grain | meaning |
|---|---|---|
| `a` | article-entity match | one sentiment-bearing mention of company `c` in one article |
| `c` | entity | a company, keyed by `yukka_id` upstream and RIC in the panel |
| `t` | date | calendar day of publication (UTC), **not** trading day |
| `pos, neu, neg` | `(t, c)` | article-entity counts by sentiment sign |
| `r` | `(t, c)` | the sentiment ratio — the only thing the book sees |
| `z` | `(t, c)` | the final cross-sectionally standardised signal |

Two grain facts that cause most confusion. Counts are in **match units, not article units**:
~9% of article-entity pairs repeat, because several sentences in one article can each carry
sentiment, and all of them are kept so the counts stay in production's units. And `t` is a
**calendar** day: weekends are deliberately retained through the ratio and the smoothing, then
dropped at the price join, so the EW recursions run on an unbroken daily grid.

## Part 1 — The sentiment strategy

### Stage 0: counts to ratio

`src/honey/data/counts.py:8`

```text
r[t,c] = (pos[t,c] + 1) / (pos[t,c] + neg[t,c] + 2)
```

A Laplace-smoothed positive ratio, bounded in `(0, 1)`. Two consequences worth naming, because
everything in the invariance section below turns on them:

- **Neutral articles are discarded entirely.** `neu` appears nowhere in the formula. An article
  the model read as neutral has exactly the same effect as an article that was never published.
- **The `+1 / +2` prior is the only place article *volume* survives.** `r` is otherwise a pure
  proportion, so scaling `pos` and `neg` by any common factor leaves it almost unchanged. The
  prior pulls low-volume cells toward `0.5` and leaves high-volume cells alone.

`sentiment_ratio()` then pivots long to a wide `date × entity` panel.

### Stages 1-6: the robust signal

`src/honey/strategy/sentiment.py:59`, `_robust_signal`. Defaults `span=21`, `clip_c=4.2`;
production overrides `span=19` (`novelty_knn_model.py:422`).

1. **Location.** `mu = EWMedian(r, span)` — an exponentially-weighted *median*, per name,
   `min_periods = span`.
2. **Scale.** `sigma = 1.4826 x EWMedian(|r - mu|, span)`. A MAD, with the constant putting it on
   a sigma footing under normality.
3. **Lag.** `mu` and `sigma` are both shifted forward one day, so a name is judged against a
   location and scale built only from data through `t-1`.
4. **Winsorise.** `z_raw = (r - mu) / sigma`, clipped to `+/-clip_c`, then mapped *back* into
   ratio space as `r_w = mu + sigma * clip(z_raw)`. Outliers are pulled in, never dropped.
5. **Smooth.** `s = EWMean(r_w, span)`, `adjust=False`, `min_samples=span`.
6. **Cross-sectional standardisation.** Per date, across names:
   `z = (s - nanmean(s)) / nanstd(s)`.

So: median/MAD for the per-name normalisation, mean for the final smooth, then one
cross-sectional z-score. The EW median truncates its window at `5 x span` (`:18`) — 95 days at
`span=19` — so weight beyond that is discarded outright, not merely small. `span` therefore sets
both a decay rate and a hard window.

### Stage 7: weights

Two mutually exclusive modes.

**Threshold** (`:150`), which is what production trades: hold every name with `z > threshold`,
equal-weighted `1/n`, each capped at `max_weight`, the unplaced residual left in cash. An optional
`sector_cap` scales one sector back to a share of the day's invested weight and redistributes pro
rata (`:113`); production leaves it `None`.

**Top-decile** (`:211`), the signature default: rank descending, cut into `n_buckets=10`, keep
`top_buckets=3`. Convenient for research, and **not the production book** — per the record it
overstates the production bar by roughly 0.21 Sharpe, so a change should be judged under
`threshold`.

Both restrict to names holding a non-null price that day, and the signal is inner-joined to the
price spine, which is where weekends leave.

### Stage 8: portfolio

`build_portfolio`, `:260`. Weights are aligned to the trading-day spine and forward-filled when
`rebalance_freq > 1`. Optional vol targeting: EWM variance of portfolio log returns (`vol_com=20`),
annualised by 252, lagged one day, `scale = min(vol_target / vol, 1)` — it de-levers only, never
gears up. Positions are `aum x weight x scale`.

**Timing.** jquantstats computes P&L as `prices.pct_change(t) * cashposition.shift(1)`, so the
position set on date `t` earns the `t -> t+1` return. With `mu`/`sigma` lagged at stage 3 the chain
is free of lookahead. Note `r` itself is *not* lagged: stage 4 scores today's ratio against
yesterday's location and scale, which is correct given that position timing.

### Production parameters

`novelty_knn_model.py:422`, mirroring `sentiment_momentum_top200.py`:

```text
span 19 | clip_c 4.2 | threshold 1.15 | max_weight 0.10
rebalance_freq 10 | vol_com 20 | vol_target 0.20
```

## Part 2 — Novelty scoring

Novelty asks: how much does this article's *headline* resemble the headlines that came before it,
for this company, inside a lookback window?

### The embedding store

`novelty_knn_model.py:186`. Three memmapped `.npy` files: an `(N, 256)` float32 matrix of
unit-norm rows, a sorted `uint64` array of title hashes, and a row index. Lookup is one vectorised
`searchsorted`, not a dict — a dict over 11.7M Python strings cost ~3 GB against the store's
~190 MB. The hash uses a **fixed seed** (`_HASH_SEED = 0`), because Python's own `hash` is salted
per interpreter and the build and lookup sides would disagree across processes. Titles are
embedded **unique only** — a 2-3x saving, since titles repeat heavily.

### Article preparation

`load_company`, `:355`. Per company file: `on_topic` means the article has at least one match span
for that company, and `sentiment` is the **sign of the sum** of its span sentiments. Off-topic
articles are dropped. Rows are sorted by timestamp, which every later step assumes.

### The kNN score

`knn_novelty`, `:299`. For each article `i`, with window `W` and neighbour count `n`:

```text
C(i)       = { j : t_i - W <= t_j < t_i }          candidates, strictly earlier, same company
novelty(i) = 1 - mean( top-n cosine similarity over j in C(i) )
```

Implementation facts that are load-bearing:

- **Candidates are same-company only.** Scoring is per company file; there is no cross-company
  neighbour search. An identical story about a different name is not a duplicate here.
- **`max_cand` caps the candidate set** to the most recent `2000` earlier articles (default), so
  for a busy name the cap binds before the window does and the effective lookback is shorter than
  the nominal one.
- **An un-embedded candidate is masked to `-inf`**, i.e. treated as unknown, not as dissimilar.
- **An un-embedded query stays `NaN`, never `1.0`** (`:311`). A zero vector scores cosine 0 against
  everything, which would read as "nothing similar came before" and hand maximum novelty to
  scraper residue. 39,077 articles are in this state.
- **Nothing earlier in the window gives `1.0`** — genuinely novel.
- All `n` in the grid come out of **one** `np.partition` at `max(n)` plus a cumsum, so a wide `n`
  grid is nearly free. Selection dominates the cost, not the matmul; see the subpackage CLAUDE.md
  before restructuring.

### Clipping

`CLIP_NOVELTY`, `:521`. The score is `1 - cosine` in float32, so a near-identical reprint lands a
hair above 1.0 and the raw score goes **negative** — 28.6% of articles, down to `-0.0016`. Left
alone that is wrong twice: as a weight, a perfect duplicate would *subtract* from its sentiment
bucket instead of contributing nothing, and any fractional power of it is NaN, which silently
poisons a whole day's counts. Clipped to `[0, 1]` in the loaders, so every consumer inherits it.

### Two paths from score to signal

Both produce the same `(date, entity, positive, neutral, negative)` shape the production ratio
consumes, and that is the entire interface — the novelty arms change *only* the counts.

**Threshold** (`threshold_long`, `:458`): keep articles with `novelty > theta`, then count as
production counts. `theta` is a free parameter that doubles as a sample-size knob — raising it makes
the signal more selective *and* estimates it from less data.

**Weighted** (`weighted_long`, `:478`): every article contributes its own novelty to its sentiment
bucket, so `pos = sum of novelty over positive articles`. No `theta`. Null novelty is dropped
rather than weighted.

### Measurement

Every arm is built by handing `build_arm` (`:433`) a different `ratio_wide` and changing nothing
else, so a Sharpe difference can only come from the signal. `window_sharpe` scores a fold;
`sharpe_se` (`:505`) is the standard error, `~sqrt(252/n)`, and it is the number that governs the
whole exercise: on ~80-day validation blocks it is about **1.8**, larger than any edge being looked
for, so a fold Sharpe of 3 and one of 1 are not distinguishable.

## Part 3 — Why dropping articles barely moves the book

The hypothesis was that discarding most of the article base should move the score a lot, and that
if it does not, the count swap cannot be the only change required. **That is correct, and this is
the mechanism.**

### Four layers of invariance

The production chain is, by construction, insensitive to exactly the kind of perturbation article
exclusion produces. Suppose exclusion keeps a fraction `f` of a cell's articles without regard to
sentiment, so `pos -> f*pos` and `neg -> f*neg`:

1. **The ratio is a proportion.** `f*pos / (f*pos + f*neg) = pos / (pos + neg)`. The factor
   cancels exactly. Only the Laplace prior survives: `(f*pos + 1) / (f*pos + f*neg + 2)` shrinks
   toward `0.5`, and *how much* depends on volume — so exclusion perturbs thin names far more than
   busy ones.
2. **Stage 4 subtracts a 21-day EW median of the same transformed series.** A shrinkage that is
   roughly stable over the span cancels again in `r - mu`.
3. **It then divides by the MAD of that same series**, so a multiplicative compression of
   dispersion cancels a third time.
4. **Stage 6 removes whatever common component is left**, cross-sectionally, every day.

Each layer independently strips out level and scale effects. What reaches the weights is a
*cross-sectional ordering*, and then only whether it clears `1.15`.

### Measured, on the real 7-day scores

38,908,382 article-entity rows, 752 dates x 229 entities, pushed through the production chain
(`span=19`, `clip_c=4.2`) and compared with the all-articles baseline:

| arm | article mass kept | pooled `r` | CS rank rho | basket overlap |
|---|---:|---:|---:|---:|
| `novelty > 0.10` | 59.9% | 0.967 | 0.965 | 71.9% |
| `novelty > 0.30` | 25.2% | 0.876 | 0.872 | 52.3% |
| `novelty > 0.50` | 4.8% | 0.696 | 0.688 | 32.8% |
| novelty-weighted | 18.2% | 0.924 | 0.916 | 58.2% |

Discarding **three quarters** of all articles leaves a signal correlated 0.876 with the original.
Discarding 95% still leaves 0.696.

### The decisive control

Compare the informed drop against a **random** drop at the same rate:

| arm | kept | pooled `r` | basket overlap |
|---|---:|---:|---:|
| `novelty > 0.30` (informed) | 25.2% | **0.876** | 52.3% |
| random drop (seed 0) | 25.2% | 0.939 | 59.9% |
| random drop (seed 1) | 25.2% | 0.939 | 59.5% |
| random drop (seed 2) | 25.2% | 0.938 | 60.2% |

Random thinning alone accounts for most of the displacement. Novelty-specific structure moves the
signal only from 0.939 to 0.876 — so **the larger part of what novelty filtering does to this book
is inject sampling noise, not information.** That is a sufficient explanation for the null result
without novelty being uninformative about articles.

The basket overlap tells the other half of the story: 52% at `novelty > 0.30` means **48% of the
book turns over** while the Sharpe does not move. The threshold at `1.15` sits far enough into the
tail that small signal changes flip membership, so the arms trade differently and perform the
same. Turnover without performance change is the signature of noise, not edge.

### Why the within-cell channel is so thin

`weighted_long` can only differ from the baseline through **covariance between novelty and
sentiment sign inside a single `(date, entity)` cell**. If every article in a cell carried the same
novelty `k`, the weighted counts would be `k*pos` and `k*neg`, and layer 1 above would cancel `k`
entirely — leaving only the Laplace artefact. Measured over the 143,203 cells with at least four
articles:

```text
mean novelty(positive) - mean novelty(negative):  mean -0.0199, sd 0.1348
median |difference| 0.0793   vs  median within-cell novelty sd 0.1662
```

The systematic part is `-0.02` against a within-cell spread of `0.17`. There is a real, slightly
negative tilt — new articles skew marginally negative — but it is an eighth of the noise it sits
in. That single number is the entire alpha budget of the weighting path as currently specified.

### What would actually change the score

Anything that survives all four layers. In rough order of promise:

1. **Break the volume/confidence conflation.** The Laplace prior is the only place article count
   enters, and it does so as an accident of smoothing rather than as a decision. A cell built from
   3 novel articles and one built from 300 reprints currently produce comparable `r` with
   comparable authority. Making effective sample size explicit — shrink toward the
   cross-sectional mean in proportion to novelty *mass* rather than raw count — is a change no
   normalisation layer can absorb, because it alters the cross-section, not a level.
2. **Shorten `span` for the novelty arms.** Layers 2 and 3 absorb anything that persists across 19
   days. A novelty effect is, by hypothesis, a *transient*. At `span=19` the pipeline is built to
   remove precisely the thing being measured.
3. **Use novelty cross-sectionally, not per article.** Rank companies by the novelty of their flow
   on day `t` and use that as a second signal, rather than as a filter on counts feeding the first.
   A filter is invisible to the layers above; a separate cross-sectional score is not.
4. **Attack the neutral discard.** `neu` is thrown away at stage 0, yet the neutral share is
   itself informative about whether a day's flow is real news or bulletin residue.
5. **Score `theta` and sample size separately.** As specified, `theta` moves selectivity and
   sample size together, so its two effects work against each other and cannot be attributed.

Note that (1) through (3) are changes to the **signal construction**, not to the counts. That is
the substance of the hypothesis: the count swap is the one intervention this pipeline is designed
to ignore.

## Caveats

- The correlation table is signal displacement, not P&L. It shows the *mechanism* of the null
  result; the Sharpe figures behind the null itself are in [`NOVELTY.md`](NOVELTY.md).
- All of it is `es_metadata`, 2024-07 onward. The body-text dump does not reach this period (1.8%
  of its rows), so nothing here is affected by it either way.
- `sharpe_se ~ 1.8` on fold-sized blocks bounds every performance claim in the experiment.
- Novelty is title-only for this period, and same-company only by construction.
