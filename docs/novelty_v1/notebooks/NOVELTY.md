# Novelty-Weighted Sentiment

## Scripts

| Script                             | Kind            | Purpose                                                        |
|------------------------------------|-----------------|----------------------------------------------------------------|
| `novelty/novelty_alpha_decomposition.py` | Marimo notebook | Splits each arm's return into market exposure and the rest |
| `novelty/novelty_subset_ceiling.py`      | Marimo notebook | Sweeps the whole (window × n) grid for headroom over production |
| `embed_titles_sync.py`             | CLI (argparse)  | Embeds unique dump titles into the vector store (run once)      |
| `novelty_knn_model.py`             | CLI + library   | Scores novelty; also holds the signal, portfolio and eval helpers |

The two CLIs live in `honey/novelty_score_exp/`; see the `CLAUDE.md` there for the
implementation details that matter when changing them.

---

## Idea

Most of what a news feed carries about a company is not new. A story breaks once and is then
reprinted, syndicated, translated and rewritten, so one event can arrive as dozens of articles.
The production sentiment signal counts all of them, which makes an event's weight in the signal
depend on how widely it was syndicated as much as on what happened.

Each article is therefore scored for **novelty** — how different its headline is from headlines
that already existed — and that score drives the daily sentiment count instead of a flat count of 1
per article.

---

## Data layer

Loaded through `honey.data.preamble`, exactly as the sentiment strategy does:

- **Prices / returns** — STOXX 600, top-200 by rank, point-in-time masked.
- **Counts** — Yukka's t-stat ≥ 2.5 production counts (`load_counts`), the baseline arm.
- **Sector mask** — `load_sector_mask`, the point-in-time Financials panel `build_portfolio` takes.
- **Benchmark** — `load_benchmark("STOXX 600")`.

Experiment-specific inputs live in `data/` (gitignored, ~45 GB): the `es_metadata` article dump and
the title embeddings derived from it. Article data begins **2024-07-01**, which is the binding
constraint on everything below.

---

## Novelty score

For each article, against the **same company's earlier** articles inside a lookback window:

```text
novelty = 1 - mean(cosine similarity to the n nearest earlier titles)
```

A verbatim reprint scores ≈ 0, a genuinely new story ≈ 1. Causal by construction — only earlier
articles are ever candidates. Two parameters define the measurement:

| Parameter  | Meaning                                          | Values swept                    |
|------------|--------------------------------------------------|---------------------------------|
| **window** | how far back to look for earlier coverage        | 6h, 24h, 3d, 7d, 14d            |
| **n**      | how many nearest earlier articles to average over| 1, 3, 5, 7, 10, 20, 50, 70, 100 |

**window** is the memory. At 6h an article only counts as a reprint if the original appeared within
the last six hours; at 14d the original is still remembered two weeks later. Longer windows catch
more reprints but risk reading a genuine follow-up as an echo.

**n** is how strict the comparison is. At n=1 the score depends only on the single closest earlier
article — *is there one near-identical piece already out there?* At n=100 it is the average distance
to the hundred nearest, which measures how crowded the recent coverage is rather than whether this
story is a duplicate. Small n asks "is this a copy?"; large n asks "is this company being talked
about a lot?"

5 windows × 9 values of n gives the **45 cells** compared throughout.

---

## Signal construction

The score reaches a daily signal one of two ways, both in `novelty_knn_model.py`:

| Mode          | Function                        | Free parameters |
|---------------|---------------------------------|-----------------|
| **Threshold** | `threshold_long(scores, theta)` | theta           |
| **Weight**    | `weighted_long(scores)`         | none            |

Threshold keeps `novelty > theta` and counts the survivors. Weighting lets each article contribute
its own novelty, so a reprint adds almost nothing and a first telling almost one.

Weighting exists because theta discards the score twice over: an article that just clears the bar
counts the same as a wholly new story, and theta doubles as a sample-size knob — raising it makes the
signal more selective *and* estimates it from less data. Both modes emit
`(date, entity, positive, neutral, negative)`, the shape `sentiment_ratio` consumes, so novelty arms
and the production arm share one downstream pipeline.

---

## Portfolio construction

`build_arm` pins `honey.strategy.sentiment.build_portfolio` to `PROD_PARAMS` — the production
sentiment leg parameter for parameter:

```python
span = 19, clip_c = 4.2, threshold = 1.15, max_weight = 0.1,
rebalance_freq = 10, vol_com = 20, vol_target = 0.2
```

Note this is a **threshold** book, not top-decile: hold every name whose robust cross-sectional z
exceeds 1.15, equal-weight capped at 10% per name with the residual in cash, then scale to a 20% vol
target. Every arm including production is built through `build_arm`, so a difference in outcome is
attributable to the signal and never to the book.

This matters: measured on identical counts and panel, production scores **1.22 in a top-decile book
but 1.01 in its own**. Comparing a candidate built one way against a baseline built the other way
measures the book, not the signal.

---

## Findings

At span 19 through the production book, novelty entering as a weight, 566 trading days
(2024-07-01 → 2026-08-12) and 518 of those for anything benchmark-relative.

### Production has no alpha

Regressed on the STOXX 600: **beta 0.894, R² 77.3%, alpha −0.26% p.a. with t = −0.05**. Its Sharpe is
market exposure. Sharpe is therefore the wrong scoreboard for a *signal* — the honest metric is
beta-hedged alpha.

### Every novelty cell carries lower beta

**45 of 45**, betas 0.793–0.834 against production's 0.894. Robust to the window used and to which
market series is regressed against.

### Novelty decays monotonically in n

Mean alpha by n falls from +0.30% (n=1) to −0.84% (n=100); **Spearman(n, alpha) = −0.80, p = 0.010**.
The same ordering holds on Sharpe at every span tested (ρ ≈ −0.9). This is the only inferential result
with real power, because it is a rank test across the grid rather than a level test on one cell.

### Nothing is individually significant

**0 of 45** cells reach |t| > 2, and the paired block bootstrap on the alpha difference spans zero.
The direction is not even stable: whether novelty adds alpha flips sign depending on whether the final
18 trading days are included, over which production returned +3.6% against +0.5% for the grid. Two
years cannot resolve an effect this size.

### The grid is one nested family, not 45 ideas

`novelty_n1 ≤ n3 ≤ n5 ≤ n100` holds for **99.87%** of 38.9M articles, and the traded signals correlate
0.915–0.998 with each other (against 0.855 with production). Averaging cells therefore cannot
diversify — an ensemble lands *between* its members, which is what happened when tried. Two
corollaries: further tuning inside this grid is exhausted, and the multiple-comparisons exposure from
"45 configurations" is far smaller than it looks.

### n=1 is partly a recency effect

**67.5%** of articles find their nearest neighbour within six hours of publication, so leaning on the
top-1 similarity is implicitly leaning on recency. Decaying candidate similarity by age is the one
genuinely *new* axis available, since window and n are both nested — and it would run faster, because
a decay-weighted mean over all candidates removes the selection step that dominates the cost.

---

## Caveats to carry with any number here

- **Two years, one regime.** The window is entirely inside a market that rose 14.5% a year.
  Production itself earns 1.19 over the article window against 0.66 over its full 2015–2026 history,
  so this stretch is known to be unrepresentative.
- **Whole-window Sharpe standard error is ±0.67**; per-fold, on 80-day blocks, ±1.8. The effects
  being chased are 0.05–0.25.
- **Benchmark coverage is short.** `cache/benchmark_stoxx600.parquet` ends 18 trading days before the
  price panel, and beta/alpha need a market return, so benchmark-relative figures stop there. Those 18
  days are worth +0.10 Sharpe to production and +0.00 to the novelty arms, which reverses the Sharpe
  comparison: 20/45 cells above production on the truncated window, 4/45 on the full panel.
- **Titles only.** The dump's `teaser` field is 0% populated, so reworded reprints are undetectable
  from an 86-byte headline. This caps reprint detection regardless of sample size.

---

## Next steps

Cheap, and needing no new data:

1. **Filter or tier `source_name`** — it is populated and unused, and ~12.8% of the flow is
   machine-generated price bulletins that carry sentiment tags. Needs one dump re-pass, no
   re-embedding.
2. **Build the recency-weighted scorer** — the only non-nested axis left, and faster than the current
   one.

Everything requiring a significance claim needs the ten-year article history.
