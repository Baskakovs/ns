# The novelty query layer

Parquet is the storage; these files are a **DuckDB view layer** over it. Nothing is copied
into a `.duckdb` file, so the data has one representation and SQL is the readable interface
onto it.

```python
from honey.novelty_score_exp.engine import db

con, applied = db.connect(f="top_k_mean", window="6", k="5")
print(applied)                         # views created, and any skipped
con.sql("SELECT count(*) FROM mentions").fetchone()
```

`connect` takes the grid cell — `f` (the candidate function), `window` (hours, as the scores
filenames carry it) and `k` — and substitutes it into the definitions.

## The views hold no rules

Every view projects columns of a parquet the engine wrote, and `mentions_scored` joins two of
them on their key. Nothing else. The tone of a pair, the article and pair dedups, the clip,
the calibration and the panels are all computed in `engine/`, where each is pinned to its
definition in `core/` by an equivalence test. SQL used to hold its own copies of several of
these, and two had already drifted (raw novelty fed to the panels; a production panel over a
different corpus from the novelty arms).

`tests/test_novelty_db.py::TestTheViewsHoldNoLogic` enforces this: a definition containing an
aggregate, a filter, `DISTINCT`, `CASE` or similar fails the suite.

## Nothing stale is exposed

Each parquet a view reads was written by the engine and stamped with the fingerprint of the
code that wrote it. `connect` checks every stamp before creating a view and raises
`StaleArtifactError` on a mismatch — whatever `strict` says, because a file written under a
definition `core/` no longer holds is wrong, not missing. A file that does not exist yet is
skipped and reported in `applied`.

## Files

Applied in filename order, which is dependency order.

| file | view | backing parquet | written by |
|---|---|---|---|
| `01_titles.sql` | `titles` | `es_title_embeddings/titles_emb_*.parquet` | the embedding run |
| `02_article_titles.sql` | `article_titles` | `article_titles.parquet` | the headline-dump pass |
| `03_sources.sql` | `sources` | `derived/sources.parquet` | `engine.derive` |
| `04_constituents.sql` | `constituents` | `stoxx600_ever_members.parquet` | the secmaster export |
| `05_articles.sql` | `articles` | `derived/articles/*.parquet` | `engine.derive` |
| `06_mentions.sql` | `mentions` | `derived/mentions/*.parquet` | `engine.derive` |
| `07_scores.sql` | `scores`, `mentions_scored` | `es_title_embeddings/novelty_scores_{f}_w{h}h.parquet` | `engine.run` |

There is **no column called `novelty`**. `scores` exposes `raw_novelty` and
`calibrated_novelty`; they share a dtype and a range, so the name is the only thing that says
which scale a number is on.

## Building what the views read

```bash
uv run python -m honey.novelty_score_exp.engine.derive              # mentions, articles, sources (~2 min)
uv run python -m honey.novelty_score_exp.engine.run --window-hours 6  # one scores file per (f, window)
```

The derived stamps cover the code, not the input: rebuild after the dump changes.

## What moved out

- **Panels** (`08_panel.sql`) are `engine.aggregation.panel`, pinned to `core.aggregation`.
- **Checks** (`99_checks.sql`) are `engine.checks`, run on demand with `checks.run()`.

Measured row counts on the current store: `titles` 38,548,717 · `article_titles` 54,236,238 ·
`sources` 359,755 · `constituents` 962 · `articles` 133,666,501 · `mentions` **199,725,471**
(from 217,028,161 feed rows, 1.087x) · `scores` 38,908,382.

## Where each column comes from

The **API `sentiment_dump`** sweep (`version=9-3-5` pinned) is the backbone for article
identity, time, language, source and mention sentiment — 217.0M mentions over 133.7M articles
and 945 companies, 2013-01-01 to 2026-08-30. The **headline dumps** supply titles and cover
only part of it. `es_metadata` is the only asset carrying identity, headline, company and
sentiment on one row, and it reaches back only to 2024-07-01.

| column | source | coverage |
|---|---|---|
| `articles.article_id`, `.publish_time`, `.source_id`, `.lang` | feed `news_id`, `publish_time`, `source_uri`, `news_lang` | full window |
| `articles.title` | headline dumps, joined on `news_id` | **8.1%** — 10.86M of 133.7M. NULL means "article known, headline not" |
| `mentions.company`, `.sentiment` | feed `yukka_id`, `sentiment` | full window |
| `scores.raw_novelty`, `.calibrated_novelty` | computed by `engine.run` | only where a title and an embedding exist |
| `titles.embedding_vector` | `titles_emb_*.parquet` | 38.5M vectors |
| `constituents.*` | `stoxx600_ever_members.parquet` | 951 ever-members, 962 spells |

## What this layer cannot do

**`title_hash` is not reproducible in SQL.** It is polars' `hash(seed=0)`, and DuckDB's `hash()`
returns a different value for the same string (verified: 7672462473463961424 vs
4972010805505530912). The hash stays internal to `engine/store.py`, where the numpy memmap needs
a uint64 key, and these views join on the **title string** instead.

**Views carry no constraints.** The previous schema was `CREATE TABLE` with PK, FK and CHECK
enforced at insert. The invariants are `engine.checks` functions returning the count of
violating rows, which can be run at any time rather than only at load.

## Load-time hazards

Measured, not assumed. Each one silently distorts the signal rather than failing loudly.

1. **Pin `version=9-3-5` on every API call.** The default 9-4-0 is a different universe rather
   than a newer model: ~12% of the volume, 249 sources instead of 3,397, en/de only, and
   sentiment disagreeing on 2.5% of shared `news_id`s.
2. **`articles` picks a winner.** The feed returns one row per (article, company), so an article
   repeats once per mentioning company — fan-out 1.579. `engine.derive.build_articles` recovers
   the article grain, but resolves any disagreement arbitrarily; `checks.article_consistency`
   says whether there was one to resolve.
3. **`mentions` aggregates to one row per pair, and that is load-bearing.** The feed returns
   1.087 rows per pair — 10.2% of pairs repeat, almost all byte-identical re-ingests; only 536
   of 1,167,510 multi-row pairs disagree on sentiment. Counting rows would put the panels 1.087x
   above §3's `P`, a count of **articles**. The pair's sentiment is `core.counts.tone` over its
   feed rows (`engine.derive.feed_tone`, pinned to it), with a NULL row read as "no polarity
   found" — no mention — rather than as a neutral one. `feed_rows` carries how many rows were
   collapsed; `checks.feed_repeats` reports the disagreeing ones.
4. **NULL sentiment is not neutral, and NULL novelty is not novel.** NULL sentiment is 8.6% of
   the feed overall and 54% for the busiest names; counting it as neutral would be a large,
   quiet distortion.
5. **Novelty is (article, company)-scoped**, not article-scoped: the candidate pool is that
   company's own earlier headlines, so one article scores differently for each company it
   mentions. That is why both scales sit on `mentions_scored` while `embedding_vector` sits on
   `titles`.
6. **Never join `scores` to `mentions` on `(company, publish_date)`.** `scores` holds ~257 rows
   per company-day, so that predicate is a cartesian product per company-day. Join on
   `(article_id, company)` — which is what `mentions_scored` does.
7. **Never deduplicate stories before scoring.** The novelty score *is* the instrument for
   repetition; removing reprints first deletes the quantity being measured. `duplicate_group_id`
   is for evaluation only.
8. **2022–2023 has sentiment but no titles**, and the feed's pre-2024 span has headlines for
   only ~8% of articles. The sentiment panel is complete across the whole window; novelty is a
   2024-07-onward story.
