# event_exclusion_exp

Research code for one question: **which of Yukka's detected events make the sentiment signal
worse?** Build the daily sentiment statistic 149 ways — once unmodified, once per event with that
event's articles subtracted — and rank events by the paired change in cross-sectional Spearman IC.

Driven from `book/marimo/notebooks/event_exclusion.py`. Outputs land under `/data/`, which is
gitignored; the package is excluded from the coverage gate, but the count algebra it rests on is
unit-tested in `tests/test_event_exclusion.py`.

## Pipeline

The events sweep is resumable and month-sharded; a kill costs only the shards in flight. The
sentiment dump is the novelty experiment's existing pull and needs no re-fetch — only that its
`--batch`/universe match, which `panel.py` verifies from each dump's `_plan.json`.

```bash
# 1. article-level events, 2024-2026, 1,536 shards, ~50 min at 14 workers
uv run src/honey/event_exclusion_exp/events_sweep.py --out data/events_dump \
    --from 2024-01-01 --to 2026-09-01 --workers 14

# 1b. sentiment: reuse data/sentiment_dump (71.6M rows, 2024-01..2026-08, version 9-3-5).
#     Only if shards are missing, its own sweep resumes them:
uv run src/honey/novelty_score_exp/sentiment_sweep.py --out data/sentiment_dump \
    --from 2024-01-01 --to 2026-09-01

# 2. both price panels + the yukka_id -> ric bridge
uv run src/honey/event_exclusion_exp/prices.py --out data/prices --from 2023-11-01 --to 2026-09-01

# 3. daily count panels (the ~71M-row join, done per entity group)
uv run src/honey/event_exclusion_exp/panel.py --out data/panels
uv run src/honey/event_exclusion_exp/panel.py --out data/panels --facts-only   # robustness

# 4. the sweep
uv run src/honey/event_exclusion_exp/run_ic.py --tag eur
uv run src/honey/event_exclusion_exp/run_ic.py --tag eur --suffix _facts
uv run src/honey/event_exclusion_exp/run_ic.py --tag eur_accum --accumulate
uv run src/honey/event_exclusion_exp/run_ic.py --tag local --key yukka_id \
    --prices data/prices/local_wide.parquet

# 5. integrity checks (count algebra, join, vs Data API export, shard completeness)
uv run src/honey/event_exclusion_exp/verify.py --panels data/panels
```

## API routes, and why these ones

The obvious route does not exist. **No sentiment endpoint accepts an event filter**, and none
exposes an article id: `sentiment_filter` takes only entities/sentiment/implication, and
`collapse_on` *collapses* dimensions rather than exposing them. Dead ends worth not re-exploring:

| Endpoint | Why it cannot serve the formula |
|---|---|
| `/api/sentiment_matches`, `/v5/api/events` | Aggregate counts only, no shared article key, no sentiment x event cross-tab |
| `sentiment_counts_ts_itemized` | "Itemized" means per *entity*, not per sentiment. No event filter |
| `news_count_itemized` | Has `event_ids` *and* `sentiment`, but one scalar per entity for the whole range — daily would need ~288k calls |
| `*_snippets` | Article-level but cursor-paginated at 20 rows a page |
| `/company_data/v2/events` | Carries `sentiment` on the event row, but keys on `news_id_hash` (numeric), which does **not** join to the sentiment dump's `news_id` (sha1) — 0% overlap. Its sentiment is also sentence-level, varying within one (news, entity) pair |
| `/company_data/eod_data` | Ready-made daily panel with ~140 event-count columns, but unsigned — no per-event pos/neg split. Good independent cross-check on totals |

**What works:** `POST /company_data/events_dump` joined to `POST /company_data/sentiment_dump` on
`(news_id, yukka_id)`, both pinned to `version=9-3-5`. Measured miss rate: 0.0% — the events dump is
an exact subset, so the subtraction can never go negative.

## Running the notebook

Interactively, the usual way:

```bash
uv run marimo edit --sandbox book/marimo/notebooks/event_exclusion.py
```

Headless (executes every cell — the check that it actually still works):

```bash
uv run --with marimo marimo export html --no-include-code \
    -o /tmp/event_exclusion.html book/marimo/notebooks/event_exclusion.py
```

Use the **project** environment plus marimo, as above, not `marimo export --sandbox`: the sandbox
rebuilds its own requirements and drops the `[tool.uv.sources]` editable path to `honey`, so the
notebook's imports fail there for reasons that have nothing to do with the notebook.

`EVENT_EXCLUSION_DATA` overrides the data directory, for reading a variant run without editing the
notebook.

## Null sentiments are common, and they are not missing data

**A retraction worth reading before you re-derive it.** Early in this experiment `data/sentiment_dump`
was diagnosed as truncating high-volume entities, and a full month-sharded re-pull was started to
replace it. That diagnosis was **wrong**. The comparison behind it was invalid: a row count *after*
`drop_nulls("sentiment")` was compared against the API's *raw* entry count.

| `company:ubs`, January 2024 | rows |
|---|---|
| year-sharded `data/sentiment_dump`, raw | 69,951 |
| month-sharded re-pull, raw | 69,951 |
| either, after dropping null sentiments | 32,139 |

The two pulls are identical. Aggregate totals agree too: 71.6M rows year-sharded against a 72M
projection monthly. **`data/sentiment_dump` is sound; no re-pull is needed.**

What is real is the nulls themselves:

* **8.6% of the dump** has `sentiment = null` — but very unevenly, reaching **54% for `company:ubs`**.
* They are almost entirely `news_type = non_professional`.
* A null-sentiment `(news_id, yukka_id)` **never** also appears with a real sentiment (measured 0.0%),
  so it means "entity mentioned, no sentiment assigned", not a duplicate row.

Such an article carries no sentiment to remove, so it cannot move any variant. Both the baseline and
the event counts drop it, which is exactly what keeps `pos_a <= pos` true, and `Tot = pos + neu + neg`
stays an identity. Dropping is correct — but **never compare a filtered count to a raw one**, which
is why `verify.spot_check_completeness` compares raw row counts and `panel.build`'s integrity check
uses `sentiment_pairs` (pre-filter) rather than `sentiment_matches` (post-filter).

## `events_dump` 500s on individual poisoned records (2025-12 onwards)

The single most time-consuming thing in this experiment, and the least guessable. Specific
`(entity, day)` combinations return **HTTP 500 reproducibly**, even for a single-entity single-day
request — and they are per-*record*, not per-date:

| batch | 2025-12-17 | 2025-12-18 | 2025-12-19 |
|---|---|---|---|
| b001 | 200 | **500** | 200 |
| b002 | 200 | 200 | **500** |
| b029 | 200 | 200 | 200 |

Confirmed culprits include `company:aena_sme_sa` on 2025-12-18 and, more broadly, ~100 entity-days
in 2025-12 alone across names like `company:hsbc`, `company:anheuser_busch`, `company:a_springer`.
Only recent periods are affected; Jan 2024 – Nov 2025 pulls cleanly. **Worth reporting upstream.**

Two dead ends this cost, recorded so nobody repeats them:

* **It is not request size.** A whole month of 2025-12 does 500, and splitting to weeks and then
  days does help for *some* shards — but no date split can ever fix a poisoned record, so bisection
  alone stalls at single days and fails.
* **It is not concurrency, and I twice concluded that it was.** Serial probes appeared to prove it
  (the same windows that failed at 12 workers succeeded serially) — but those probes happened to
  miss the poisoned combinations. Once gap-handling was added, **8 and 16 workers both ran with zero
  errors.** Any "load" theory here should be tested against a *known-poisoned* record before being
  believed.

`_sweep.fetch_window` therefore narrows on two axes: dates first (cheap, fixes oversized responses),
then the **entity list** once a window is down to a single day. A single entity-day that still fails
is appended to `gaps`, logged, and skipped. `verify.gap_report` reads those back out of the manifest.

Cost note: isolating one poisoned record takes ~10 narrowing steps, so `SPLIT_BACKOFF` is
deliberately **empty** — a retry-with-sleep on the way down gets paid ten times over for two
failure modes that waiting cannot cure. Setting it to `(2,)` took the sweep from ~3.5 shards/min to
~1.4.

**What the gaps mean for the result.** A gapped cell keeps its sentiment counts but loses its event
counts, so every variant equals the baseline there. The effect is to shrink measured deltas towards
zero — it can never manufacture one. At the observed rate (~0.2% of company-days) it is immaterial,
but `verify.py` prints the exact figure rather than assuming so.

## Gotchas

* **`entities x days <= 1000`.** `events_dump` rejects anything larger with a 400 and an explicit
  message; `_sweep.check_complexity` asserts the batch size before any request goes out.
  `sentiment_dump` has no such cap and is pulled by year. `_sweep.fetch_window` additionally
  bisects a window by date when the gateway 502s on an oversized response, which happened on the
  heaviest entity batches — retrying the same request never works, and those batches hold the
  highest-volume names.
* **Unsigned-count underflow.** A boolean `.sum()` in Polars yields `UInt32`, on which `pos - neg`
  wraps to ~4.29e9 for any net-negative company-day, which then ranks as the *most positive* name in
  the cross-section. `panel._count_aggs` casts to `Int64`, and `statistic._ratio` casts again for
  panels built elsewhere. Silent and catastrophic if reintroduced.
* **`constituent_prices` currency is per row, not per query.** `close_price` is EUR-normalised on all
  but ~23 of 738 dates per name (market holidays), where the row's `currency` reports the local
  currency instead. Those rows are nulled. Verified against
  `src/honey/data/cache/prices/top200/2026-08-13.parquet`: 218k overlapping cells, median relative
  difference 0.0, p99 2e-16. The endpoint also passes provider sentinels (`-9000000`) through raw,
  so `honey.data.utils.clean_prices` is applied.
* **`price_data_itemized` returns one arbitrary listing per entity**, in local currency, sometimes a
  US ADR (45 of 951 — dropped). FX cannot be bridged from the two feeds: implied rates disagree by
  ~3.8% across names because the listings and price scales differ. Hence local currency, ranking
  only.
* **The `unique` in `event_tags` is load-bearing.** The events dump is multi-row per article-entity
  (one row per event x role x factuality x temporality). Without the dedupe the join fans out and
  one article contributes its sentiment several times, breaking `pos_a <= pos`.
* **`panel.py` joins by entity group, not shard index.** The two dumps' partitions are
  reconstructed from their `_plan.json` files, so they stay joinable if one ever needs a different
  `--batch` than the other. Readers take an `entities` filter for the same reason: a sentiment shard
  wider than the group would otherwise contribute its extra entities to every group that touches it.

## Method notes

* **Counts are match-level, tags are article-level.** ~9% of `(news_id, yukka_id)` pairs repeat in
  the sentiment dump (several sentiment-bearing sentences per article). Those rows are kept, so
  `pos`/`neg`/`tot` stay in the same units as production `counts()`; each match inherits its
  article's event tags. Sentence-level attribution of a sentiment to a *specific* event is not
  recoverable from any API.
* **The test is paired.** Market-wide IC swings dwarf any single event's contribution, so
  `mean_ic(variant) - mean_ic(baseline)` as two independent samples has almost no power. The
  per-date difference `d_t` is formed first and its mean tested, with Newey-West errors at lag `h`
  because horizon-`h` dates share `h-1` days of forward return.
* **Read `q`, not `p`.** ~148 simultaneous tests means ~7 spurious hits at `p < 0.05`.
* **Read `match_share` next to every delta.** The extremes of a 148-way ranking are dominated by
  rare events whose estimates are noise. Section 4 of the notebook is the funnel check that guards
  against a spurious headline.

## Known limitations

* Production's server-side `t_stat=2.5` source filter is **not reproducible** from the dumps (they
  carry `source_uri` but not the statistic), so baseline IC levels sit above the live book's. The
  result is the *difference* between arms, not the level.
* The EUR panel covers current constituents only; names delisted mid-sample are missing. The
  local-currency panel (905 entities) is the survivorship check.
* ~2.7 years of data. With 148 events tested, few or no survivors after FDR correction is a real and
  reportable outcome, not a failed experiment.
