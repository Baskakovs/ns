# CLAUDE.md — novelty_score_exp

Implementation notes for the novelty experiment. **[`Novelty Paper.md`](Novelty%20Paper.md) is the
specification** — read it first. This file records what the code does, what the data actually
supports, and the places the two disagree with the paper.

The previous implementation is at `archive/novelty_v1/`, outside `src/` and on no import path.
It is kept so results can be traced to the code that produced them. **Do not read it to decide
how anything here should work** — it predates the paper and disagrees with it.

## Where this stands — read this first

Branch `70-restructure-novelty`, 833 tests passing under `-m "not integration"`. The approved
plan is at `~/.claude/plans/warm-percolating-origami.md`; its step numbering is used below.

**Read "The divergence guard" below before changing anything in `core/` or `engine/`.** Importing
`engine` raises until a changed pair has been re-certified, by design.

**The embedding run is finished and needs nothing further.** 28,386,030 titles, $10.21, 16.2h,
no errors. The store was rebuilt from all 1,885 shards and validated: **38,548,717 x 256**, keys
sorted, **zero hash collisions**, worst norm deviation 7.695e-4 against `NORM_TOL` 1e-3. The
`titles` view sees all of it. Note the store is a *superset* of the union — 26.89M new plus the
11.66M es_metadata titles, which are kept because scoring reads es_metadata.

**`core/` is complete, and the scorer is wired end to end.** Steps 2-9 are done, the scorer
writes `article_id` with both scales, and **6h has been re-scored** through it
(`novelty_scores_top_k_mean_w6h.parquet`). What remains is the arms (step 10) and re-scoring the
other windows (step 11).

### Built

| module | step | state |
|---|---|---|
| `core/similarity.py` | 3 | done, **reviewed** |
| `core/counts.py` | 4 | done, **reviewed** |
| `core/representation.py` | 5 | done, **reviewed** |
| `core/reference_set.py` | 5 | done, **reviewed** |
| `core/functions.py` | 6 | done |
| `core/novelty.py` | 6 | done |
| `core/calibration.py` | 8 | done |
| `core/aggregation.py` | 9 | done |
| `engine/params.py`, `io.py` | 2 | done |
| `engine/store.py` | 3 | done |
| `engine/corpus.py` | 4 | done, partially reviewed |
| `engine/scorer.py` | 7 | done, pinned by the equivalence test |
| `engine/calibrate.py` | 8 | done |
| `engine/db.py` + `sql/` (7 files) | — | done, projection-only |
| `engine/derive.py`, `checks.py`, `run.py` | — | done |
| `twins.py` + `certificates.json` | — | done — the divergence guard |

### Not built yet

`arms.py`, `embed/`, `ingest/` (step 10). Then step 11: re-score the remaining windows through
`engine.run` and regenerate the grids:

    uv run python -m honey.novelty_score_exp.engine.run --window-hours 24   # ~12 min each

**Only 6h exists in the current format.** The archived `novelty_scores_w{24,72,168,336}h.parquet`
(and the old-format `w6h`) are still on disk but no longer match the `novelty_scores_{f}_w*h`
glob, carry no code stamp, and are keyed `(entity, date)` — nothing reads them. Delete them once
the windows are re-scored.

**Every figure in `paper_data_pilot/` is on raw `nu`**, and was computed before calibration was
wired in, so none reflects what §sec:calib specifies.

### Two integration tests fail, and it is not this code

`tests/test_repository.py::TestStoxx600ConstituentCap` — both fail on
`404 Not Found` for `STOXX600/ranking.parquet` in R2. They passed earlier in the same session,
so the object went missing from the bucket rather than the code changing. Unrelated to this
package, and it would break `honey execute` the same way. `-m "not integration"` is green at 704.

### How the review is being run

The author reads `core/` closely, line by line against the paper, and trusts `engine/` on the
strength of tests that pin the two together. `tests/test_engine_honours_core.py` is organised by
*decision* rather than by module for that reason — all 40 engine symbols are covered. Do not
change a rule in `core/` without checking what that file pins.

Three real defects have come out of that review, all the same shape — **one rule implemented in
two places, and the copies had already diverged**:

- `similarity` returned NaN for an unembedded query where `similarities` returned 0.0.
- `tone_expr` returned 0 for an article naming a company with no sentiment-bearing span, where
  `core.tone` returns None; `load_company`'s filter masked it.
- `mentions` counted feed rows, so the panels sat 1.087x above the article count that §3 and
  `core.counts` both use.
- `reference_set` and `daily_set` were the same four lines with one word changed, each carrying
  its own copy of condition (2) and the ordering. Now one `_accounts_of`, with only the interval
  at the call site.
- `calibration` had its own `EMPTY_REFERENCE` beside `core.novelty`'s, two constants that must
  be equal; and `calibrate` took a bare `Sequence[float]` that nothing tied to `R_c(d;w)`.
- The calibration equivalence test carried **its own** copy of the window rule, so it would have
  agreed with a bug rather than catching one. It now goes through `core.calibrated_novelty` —
  which is why that fix had to land before the Fenwick rewrite, not after.

The rule "an unembedded member is not in the divisor" now has **three** expressions — a list
filter in `core/novelty.py`, a comprehension in `core/calibration.py`, and absence from the tree
in `engine/calibrate.py`. Nothing in the language keeps them equal; only the equivalence tests
do. Check all three before changing any.

- **The scorer calibrated the unclipped raw column**, where `core` calibrates the clipped one.
  Near-reprints score a hair below zero in float32 and core flattens them to a tie at 0.0
  before ranking. Each step was pinned; the *composition* was not. On the real 6h run, 12.0% of
  raw scores were negative, 7.2% of calibrated values changed when fixed (raw unchanged on all
  38.9M rows), and **384,405 articles crossed `theta = 0.30`**. `scorer.score_arrays` now clips
  first and is compared end to end against core on both scales, including a reprint corpus.
- **The SQL panels read raw `nu`**, while `engine.aggregation` and §3 read `nu_tilde`; and
  `panel_production` counted every mention in 13 years while core's production counts only the
  scoreable ones. Both are gone with the panels themselves — see the query layer below.

That pattern is worth looking for first in anything not yet reviewed.

### The divergence guard

The equivalence tests only protect you when they run. `twins.py` makes a change that nobody
re-tested an **error**, not a silent drift:

- **`twins.TWINS`** pairs every public `core/` definition with its engine implementation and the
  test files that pin them. `twins.CORE_ONLY` lists the core definitions with no engine form and
  why. `tests/test_novelty_twins.py` fails on a core symbol in neither, so a new core function
  forces a decision.
- **A fingerprint** of a module is a hash of its AST with docstrings stripped (comments and
  formatting never reach the AST), plus the fingerprints of every package module it imports. So
  editing `core/reference_set.py`'s window rule invalidates the calibration pair although
  `core/calibration.py` is untouched; editing prose invalidates nothing.
- **`certificates.json`** records each pair's fingerprint at the moment its tests passed.
  **Importing `engine` raises `CoreDivergenceError`**, naming every pair whose code changed since.
- **Re-certifying** runs the pairs' tests and writes the certificate only if they pass:

      uv run python -m honey.novelty_score_exp.twins certify     # or: ... twins status

  There is deliberately no flag that writes it without running them. It lives outside
  `engine/` because importing `engine` is what fails.
- **Data is stamped too.** Every parquet the engine writes carries the fingerprint of the module
  that produced it (`engine.scorer` for scores, `engine.derive` for the derived tables), and
  every read door — `io.read_scores`, `db.connect`, `derive.check` — raises
  `StaleArtifactError` on a mismatch. So a definition change invalidates the numbers computed
  under the old one, not just the code: re-certify, then rebuild.

Verified live: changing `total > 0` to `total >= 0` in `core/counts.tone` made `import engine`
raise naming the `tone` and `aggregation` pairs; `certify` then failed 19 tests and left the
certificate unchanged; a docstring edit to the same function raised nothing.

### Open decisions

1. **`Counts` carries three buckets**, but the spec and §3 both say `(P, N)`. `neutral` exists
   only because production's panel shape has three columns and `S` ignores it. Keep in `core/`,
   or assemble at the edge?
2. **`engine/corpus.py` has three points unreviewed**: the narrowness of `DUMP_SCHEMA` (it drops
   `source_name`, which §3.3 needs, and the span offsets and coref flags); `discover_dump`
   opening 231 files to read `entity` from the data rather than the filename; and the
   `explode().explode().group_by("row")` reconstruction, which handles a nesting level that is
   **always exactly length 1** in this dump.
3. **es_metadata is still 40 GB of NDJSON** and is not in the query layer at all — the views
   read `sentiment_dump`. Converting the parsed form measured 23.6x smaller and 14x faster
   (~1.7 GB, ~0.3 min for the whole dump), but that is the *narrowed* seven columns, not a
   lossless conversion. Deferred by the author for now. If it happens, `source_name` should be
   kept and it should be exposed as `es_mentions` rather than unioned into `mentions`.
4. **`DUMP_SCHEMA` has no `source_name`, so §3.3 cannot be built at all.** The originator
   route's live form weights by the per-outlet origination rate `gamma_{c,o}`, which needs
   `o(d)`. The field is in the dump and not in the schema. This is a prerequisite for step 10,
   not a nicety — `core/aggregation.originator` raises and says so.

**Resolved since the last revision of this file:** `Article.mentions` is now a
`Mapping[str, Sequence[Mention]]` (`MentionsByCompany`), so `C(d)` cannot hold a company twice;
`companies` returns a `frozenset`, matching §2.1.

### Conventions this session established the hard way

- **Ruff rejects Greek `ν`** (RUF002, ambiguous with Latin `v`); `θ`, `φ`, `ψ`, `§` pass. Spell
  it `nu_tilde` in docstrings rather than scattering `noqa`.
- **`uvx ruff format` is not authoritative** — the pre-commit pin formats differently, so a
  commit can abort on a file `uvx` called clean. Run `uvx pre-commit run` before committing.
- **Re-read a file after a scripted string replace.** Ruff reformats between write and edit, so
  a replace can silently miss; that cost three debugging probes once.
- **S108 flags a literal `/tmp` in tests** even in a string that is never opened. Use `tmp_path`.
- **`uv run` can prune dev dependencies** from `.venv` mid-session — `pytest` vanished once.
  `uv sync --all-extras --all-groups` restores it. Read status files with plain `python3` while
  a long run is in flight.
- **Polars `when/then` needs `pl.lit(None)`**, not bare `None`, when the `otherwise` branch is
  itself an aggregation.

## Layout

One module per section of the paper, split into a readable definition and a fast implementation.

`[x]` exists, `[ ]` planned. See the handover above for which step each belongs to.

```
core/        the paper. scalar, pure, no I/O. never imports engine/
  [x] representation.py   the article model, and φ                §2.2
  [x] similarity.py       ψ, s(d,d'), the unit-norm check          §2.3
  [x] reference_set.py    R_c(d;w), A_{c,t}, E_c(d;τ)              §2.4, §3
  [x] counts.py           tone, (P, N)                            §3
  [x] functions.py        the candidate family f                  §2.5
  [x] novelty.py          ν_c(d), and the contract every f runs under   §2.1
  [x] calibration.py      calibrated novelty = Ĝ_c(ν)             §sec:calib
  [x] aggregation.py      filter / weight / originator            §3.1–3.3
twins.py     the divergence guard: core/engine pairs, fingerprints, certificates, stamps
engine/      scale. must agree with core/, pinned by an equivalence test — and refuses to
             import when a pair changed since its tests last passed
  [x] params.py           KNNParams, provenance metadata
  [x] store.py            memmapped vectors, hash seed, has_vec
  [x] corpus.py           dump parsing
  [x] io.py               the only parquet read/write door + manifest
  [x] scorer.py           blocked/threaded scorer
  [x] calibrate.py        vectorised rolling-window form of Ĝ_c
  [x] derive.py           mentions / articles / sources, built from the feed and stamped
  [x] checks.py           the invariants that were 99_checks.sql
  [x] run.py              score the dump for one (f, window); writes the stamped parquet
  [x] db.py               the DuckDB view layer over sql/
[x] sql/         the query layer (below)
[ ] arms.py      PROD_PARAMS, build_arm, window_sharpe, sharpe_se
[ ] embed/       embedding infrastructure, deliberately to the side
[ ] ingest/      sentiment_sweep — the API feed

embed_titles_sync.py, title_dump.py, body_dump.py and sentiment_sweep.py are still at the
package root; moving them into embed/ and ingest/ is step 10.
```

`core/` duplicating `engine/` is the one intentional duplication in the package. The equivalence
test is what licenses it; without that test they are two opinions rather than a spec and an
implementation.

### What this takes from the rest of `honey`

**One function.** `arms.py` imports `honey.strategy.sentiment.build_portfolio`. Nothing in
`core/` or `engine/` imports from `honey` at all.

The Laplace ratio `S = (P+1)/(P+N+2)`, the robust signal (EW-median location and scale, lagged,
clipped, EW-smoothed, cross-sectionally z-scored), the threshold book, the per-name cap and the
vol target are all reached *through* that one call and are never reimplemented here. `core/`
produces `(P, N)`; `S` stays production's. That division is §10.5's requirement to hold
everything but the map from `A_{c,t}` to `S_{c,t}` fixed.

## The data, and what it actually supports

Four assets. Every figure below is measured, not assumed.

| asset | grain | span | size |
|---|---|---|---|
| `data/es_metadata/` | one row per (article, company) | 2024-07-01 → 2026-07-22 | 38 GB, 234 files, 38.9M rows, 229 companies |
| `data/sentiment_dump/` | one row per (article, company) | 2013-01-01 → 2026-08-30 | 670 shards, 217.0M mentions, 133.7M articles, 945 companies |
| `data/es_titles_dump/` | `id`, `title`, `published` | 2016-01 → 2026-05 | 244 shards, 48.5M rows, 45.1M ids, 23.0M unique titles |
| `data/es_body_dump/` | `id`, `text`, `published` | 2016-01 → 2026-04 | 12 shards, 9.24M rows, 9.11M ids, 5.60M unique titles |

**`es_metadata` is the backbone, because it is the only asset that joins to itself.** Each record
carries `id`, `title`, `entity`, `matches_spans`, `publish_time`, `lang`, `source_name` **and
`duplicate_group_id`** in one place. Article identity, headline, company and sentiment need no
join at all.

The two headline dumps and the API feed do *not* join well:

- Of the API feed's 133.7M articles, only **10.86M (8.1%)** have a headline in either dump.
- Conversely, of the 54.2M headline-bearing ids, only 20% appear in the feed — the rest are
  articles about companies outside the 945-name universe.
- Of the 28,386,030 unique titles across both dumps, only **6,008,412 (21.2%)** belong to an
  article the feed knows about, so only those can ever be scored or serve as a candidate.

**Neither Yukka API closes that gap.** `quant.api` exposes article references (`news_id`,
`source_link`, `title`) only inside event-instance endpoints — per entity, per event, per date
window — which is not a bulk headline source. `data.api` (9-3-5) has no article-level endpoint at
all; its sentiment endpoints return aggregates and most response schemas are undefined.

### Consequence: build on `es_metadata`, treat the 13-year feed as sentiment-only

Novelty needs a headline, a company and a sentiment on the same row. That holds completely for
2024-07 onward via `es_metadata`, and for ~8% of the feed before that. So the novelty arms are a
two-year story, and the 13-year `sentiment_dump` is useful for the *sentiment* baseline and for
source-level work, not for novelty. §10.2's backfill is blocked on headline coverage, not on
compute.

## Joining novelty onto mentions

`archive/novelty_v1` scored to `(entity, date, sentiment, novelty_n*)` with **no article id**, so
a score could not be attached to the mention that produced it. That was a choice in
`_DUMP_SCHEMA`, not a limit of the data.

**The fix is to select `id`.** `engine/corpus.py` parses `es_metadata` with `id` in the schema and
`engine/io.py` writes `article_id` on every score row, keyed `(article_id, entity)` — the grain
`sql/` calls `mentions`. Nothing else is needed, and no API call is involved.

Two further columns are worth carrying: `lang`, for per-language rank calibration (§10.4), and
`duplicate_group_id`, a free reprint label — 2.35 articles per group measured on one company
file, consistent with the 70.2% non-originator share in `paper_data_pilot/syndication.json`.

Keep the schema otherwise narrow. Widening it past what is listed makes a 38 GB parse materially
slower; the five span fields other than `sentiment` are most of the dump's bytes.

### Never deduplicate stories before scoring

`duplicate_group_id` is for **evaluation only** — a negative set for the intrinsic evaluation of
§10.6, and a figure to quote. It must not filter the articles that reach a novelty score, and
neither must anything else.

The score *is* the instrument for repetition (§1.1, §2.1): removing reprints first would delete
the quantity being measured, and it would do it with a binary flag, which §1.1 says cannot
represent a recombination — the economically interesting case. `A_{c,t}` in §3 is every article
about `c` on day `t`, and it stays that way.

Three things are easy to mistake for story deduplication and are all fine:

- **`articles` is one row per `article_id`** (built by `engine.derive`). The feed returns one row per (article, company),
  so an article repeats once per mentioning company (fan-out 1.579). Collapsing that is
  recovering the article grain, not dropping coverage. Likewise ~0.24% of `(news_id, yukka_id)`
  pairs are byte-identical re-ingests and need a deterministic tie-break.
- **`titles` holds one row per distinct headline.** Identical strings have identical embeddings,
  so embedding each once is a cost saving with no effect on any score. Every *article* still gets
  scored, and two articles sharing a headline both score against the full reference set.
- **The titles dump's own 7.0% repeated ids** (48.5M rows for 45.1M ids) must be deduped, or an
  article becomes its own nearest neighbour and a fresh story reads as a reprint.

## The query layer (`sql/`)

Storage is **parquet**; `sql/` is a **DuckDB view layer over it**, and **the views hold no
rules**: each projects columns of a parquet the engine wrote, and `mentions_scored` joins two of
them on `(article_id, company)`. The tone, the dedups, the clip, the calibration and the panels
are computed in `engine/`, each pinned to `core/`. `test_novelty_db.py::TestTheViewsHoldNoLogic`
fails on any aggregate, filter, `DISTINCT` or `CASE` in a definition. `sql/README.md` has the
view table, column provenance and the load-time hazards.

**There is no column called `novelty`** anywhere — not in a parquet, a view or a frame.
`raw_novelty` / `calibrated_novelty`, and `raw_novelty_k{k}` / `calibrated_novelty_k{k}` on disk.
The two share a dtype and a range, so the name is the only thing that says which one a number
is, and `theta` means something only on the calibrated one.

`mentions` and `articles` used to re-aggregate all 217M feed rows on every query (21.5s for
`count(*)`; every join paid it again). `engine.derive` builds them once — 20s and 95s — per feed
bucket and per hash partition respectively, since a company never spans two buckets (checked at
build) and an article does.

### Two things the view layer cannot do

**`title_hash` is not reproducible in SQL.** It is polars' `hash(seed=0)`; DuckDB's `hash()` gives
a different value for the same string (verified: 7672462473463961424 vs 4972010805505530912). So
the hash stays internal to `engine/store.py`, where the numpy memmap needs a uint64 key, and the
view layer joins on the **title string**. That is more readable anyway, and parquet
dictionary-encodes the repeats.

**Views carry no constraints.** The archived schema enforced PK, FK and CHECK at insert, so "a
clean load is itself the proof". Views have none of that, so the invariants are functions in
`engine/checks.py`, each returning the count of violating rows. That is strictly more useful,
since they can be run at any time rather than only at load.

## Things that will bite you

**Scoring is selection-bound, not matmul-bound.** Measured per block: GEMM 6.6 ms, masking 43 ms,
`np.partition` **163 ms**. `BLOCK = 256`, the row-major `(articles, candidates)` layout and the
single partition at `max(k)` all follow from that, and together they are a ~13× speedup. Any
change that reintroduces a per-k selection undoes it, so a wide grid of `k` is nearly free.

**The candidate cap is not part of the score.** `KNNParams.max_cand` exists purely to make a run
fit. §2.4 bounds `R_c(d;w)` by time alone, so whenever the cap binds the code scores a shorter
lookback than `w` claims — and it binds unevenly, since a thinly covered name never reaches it
while a heavily covered one hits it in hours. At a 7-day window the candidate set runs 1,801 at
p25, 3,995 at the median and 77,417 at most, so a cap of 2,000 would bind for **72.3%** of
articles. Default is 0 (uncapped), matching the published runs.

**The three edge-case rules live in `core/novelty.py`, not in each candidate function.** A
candidate `f` receives vectors that are all valid and a set that is non-empty, and returns a
number; the wrapper owns the rest. So `top_k_mean` has no edge-case handling at all and adding
N3 or N5–N7 costs only the arithmetic.

**A missing embedding is NaN, never 1.0.** `vectors_for` returns a *zero* row for an unembedded
title, and a zero vector scores cosine 0 against everything — which reads as "nothing similar came
before" and would award maximum novelty to scraper residue. 39,077 rows are in this state.

**Raw novelty can come out negative before the clip.** It is `1 - cosine` in float32 and a
near-exact reprint lands a hair above 1.0. Demonstrated on a real stored vector:
self-similarity 1.000790, so `1 - s` is −0.00079. The clip to [0, 1] is in
`scorer.score_arrays`, **before calibration**, exactly where `core.novelty.raw_novelty` applies
it — so the stored values are the defined quantity and nothing downstream clips again.
Unclipped, a perfect duplicate would *subtract* from its sentiment bucket.

**The store's unit-norm claim is looser than the paper says.** Over all 11,660,617 vectors the
deviation runs median 1.94e-4, p90 4.00e-4, worst **8.17e-4**. `NORM_TOL = 1e-3` clears that;
`core/similarity.check_unit_norm` raises rather than renormalising, since rescaling silently
converts a build error into a permanent invisible bias.

**Calibration slides one window; it does not re-scan.** `engine/calibrate.py` keeps a Fenwick
tree over the distinct novelty values and adds/removes articles as the window moves, so the cost
is `O(n log u)` rather than one comparison per `(article, member)` pair. Measured on
`company_barclays.jsonl`, 2,324,613 articles: re-scanning took 10.8s / 54.8s / 110.5s at
24h / 168h / 336h against 106.6bn pairs at the widest; sliding takes **4.2s at every window**.
Runtime being flat in the window is the signature that it is working. A range join is the wrong
shape and would be far worse — it materialises that pair table.

**Equal novelties share a bucket**, which is how §sec:calib's `<=` falls out of the tree rather
than needing a branch. Bucketing by position instead would silently lose ties.

**The hash seed is fixed at 0.** Python's builtin `hash` is salted per interpreter, so a store
built in one process and read in another would miss every key — and a miss is indistinguishable
from "never embedded", which makes the bug look like a data problem.

**`max_cand <= 0` means uncapped, not unset.** `if p.max_cand:` preserves that by accident and
`is not None` breaks it, so use `KNNParams.capped`.

## The embedding run

Populates `titles.embedding_vector` as `sql/01_titles.sql` defines it: 256 dims, one row per
distinct headline, so a reprinted headline is embedded once.

```bash
# the key comes from .env (loaded by src/honey/__init__.py); -m is required, see the
# module docstring on why `uv run <file>` cannot work with the PEP 723 header
uv run --with openai python -m honey.novelty_score_exp.embed_titles_sync
```

Dashboard at `data/es_title_embeddings/status.html` — meta-refreshes every 2s, showing rate, ETA,
spent-versus-projected cost, shards written, and the union/embedded/outstanding/store breakdown.
`_status.json` carries the same for polling. Ctrl-C **once** flushes in-flight work; rerunning
resumes off the shards.

**Done — nothing outstanding.** 28,386,030 / 28,386,030 embedded, $10.21, 16.2h, no errors,
1,885 shards totalling 24 GB. `MEAN_TOK` is 18.98, measured on a 100,000-title sample with
`cl100k_base`; the run averaged ~950/s once the machine stopped sleeping.

The cache was rebuilt afterwards and must be, because `engine/store.load()` only builds when the
cache is *absent* — new shards are otherwise invisible, and the failure is silent: lookups miss,
vectors come back zero, novelty comes back NaN, and it looks like a data problem.

```bash
uv run python -c "from honey.novelty_score_exp.engine import store; store.build_cache('data/es_title_embeddings')"
```

17m45s, and the result validated: **38,548,717 x 256**, keys sorted, **0 hash collisions**
(`store.py` estimated ~4e-6 expected; now measured at this N), worst norm deviation 7.695e-4
against `NORM_TOL` 1e-3 — only 1.3x headroom, so the provider's normalisation is simply that
loose.

Two things the run did *not* buy. es_metadata coverage was already 99.9% beforehand, since the
old store was built from exactly those titles — so scoring the current corpus is no better off.
And only 21.2% of the union attaches to a scoreable article, so ~22M of the new vectors are
coverage for a wider universe than the current 945 names. That was the known bet at $8.46.

## Where the code and the paper disagree

Each of these is the code being right and the paper wanting an edit, except the last.

1. **`k` vs `n`.** §2.5 defines the family as `ν^(k)`; Appendix 11 says "the neighbour count is n,
   and k is never used as an index" and Appendix 12 derives in `n`. The code uses `k` throughout,
   so the appendices are what need fixing.
2. **The aggregation day.** §3 defines `A_{c,t}` over the interval from the previous trading day's
   close to its own. The code uses the **UTC calendar date** of publication, retains weekends
   through the smoothing and drops them at the price join. That is what produced every published
   number, so §3 should state the actual convention.
3. **The divisor.** §2.4 gives `n_c(d) = |R_c(d;w)|`. The code divides by the number of
   *embedded* members, which is right — a missing vector is unknown, not dissimilar — but is not
   what the paper says. `Ĝ_c`'s `1/|R_c(d)|` has the same ambiguity and is resolved the same way.
4. **The unit-norm bound.** Footnote 1 gives ~2.5e-4; measured, that is the median and the bound
   is 8.17e-4.
5. **`max_cand`** has no counterpart in the paper at all. It was 0 in every published run, so one
   sentence in §6 is enough.
6. **`tone(d)` is written article-level.** §3 says "the tonality assigned to article `d`",
   unsubscripted, but the data assigns tonality per `(article, company)` — one article carries
   separate spans for each company it mentions and can be positive for one and negative for
   another. Appendix 11 says company subscripts are "dropped wherever context fixes the
   company", and §3's sums do run inside `A_{c,t}`, so this is most likely an elision rather
   than a claim. Worth making explicit, since the pair-versus-article distinction is the same
   one §2.1 is careful about for novelty.
7. **Coreference and implied spans count like direct ones.** `matches_spans` flags whether an
   occurrence was a pronoun rather than the company's name; the tone rule ignores the flag.
   Measured over 4,000 articles: 5.3 spans per article, 97.7% direct, 2.3% coreference, 0%
   implied. Low-impact, and arguably correct — a pronoun reference is still a reference — but
   it is a modelling decision and nothing currently says so.
8. **The published figures predate calibration.** The paper thresholds and weights `ν̃`, which
   the engine now computes and the panels read; every filter and weight figure in
   `paper_data_pilot/` is still on the raw scale until the grids are regenerated.
