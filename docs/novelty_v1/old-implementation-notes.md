# CLAUDE.md — novelty_score_exp

Implementation notes for the embedding-novelty experiment's library. **What the experiment is, how the
parameters are defined and what it found are in
[`book/marimo/notebooks/NOVELTY.md`](../../../book/marimo/notebooks/NOVELTY.md)** — read that first if
you want the strategy; read this before changing the code.

## Files

| file | what it is |
|---|---|
| `novelty_knn_model.py` | the only library: scoring, both signal paths, the portfolio, eval helpers |
| `embed_titles_sync.py` | one-off CLI: embeds unique dump titles into the vector store |
| `body_dump.py` | reduces the ES **body** dump to its unique titles (first line of each `text`) |

The two marimo notebooks live in `book/marimo/notebooks/novelty/`, following the convention that
notebooks sit under `book/` (which is also what keeps them exempt from ruff's `ANN` rules and from the
`^src/` interrogate hook — see `ruff.toml`'s `**/notebooks/*.py` ignores).

Not committed, kept locally via `.git/info/exclude`: `status_dashboard.py` (a live HTML progress page;
`novelty_knn_model` falls back to a no-op `write_status` without it) and `test.py` (scratch).

## Module layout

`novelty_knn_model.py` is one file in three labelled parts, and the split is real — the halves have no
shared state and opposite dependencies:

- **Part 1 — scoring.** Needs numpy, the memmapped vector store and the article dump. Knows nothing
  about portfolios.
- **Part 2 — signal.** Needs `honey.data` / `honey.strategy`. Never touches an embedding.

If it needs splitting into separate modules later — most likely when the recency-weighted scorer lands
in Part 1 — the seam is already marked and the cut is mechanical. Paths and `ARTICLE_DATA_START` sit
above both because both need them.

## Things that will bite you

**Scoring is selection-bound, not matmul-bound.** Measured per block: GEMM 6.6 ms, masking 43 ms,
`np.partition` **163 ms**. `BLOCK = 256`, the row-major `(articles, candidates)` layout and the single
partition at max(n) all follow from that, and together they are a ~13× speedup over the naive version.
Any change that reintroduces a per-n selection undoes it. A wide grid of n is therefore nearly free —
one partition plus a cumsum yields every column.

**The vector store is memory-mapped on purpose.** `build_store_cache` writes three `.npy` files: the
`(11.7M × 256)` float32 matrix, a sorted uint64 title-hash array, and a row index. Lookups go through
`searchsorted`, not a dict — a dict over 11.7M Python strings was a 12 GB anonymous allocation. The
hash uses a **fixed seed** (`_HASH_SEED = 0`); Python's own `hash` is salted per interpreter, so the
build side and the lookup side would disagree across processes.

**novelty can come out negative.** It is `1 - cosine similarity` in float32, and a near-identical
reprint lands a hair above 1.0, so the raw score goes to −0.0016 for **28.6%** of articles.
`CLIP_NOVELTY` clips to [0, 1] in the loaders. Unclipped this is wrong twice: as a weight a perfect
duplicate would *subtract* from its sentiment bucket, and any fractional power of it is NaN, which
silently poisons a whole day's counts.

**A missing embedding yields NaN, never 1.0.** A zero vector scores cosine 0 against everything, which
would read as "nothing similar came before" and award maximum novelty to scraper residue (39,077 rows).
Unknown has to stay unknown — the failure is closed, not open.

**`_DUMP_SCHEMA` is deliberately partial.** Narrowing the span struct to `sentiment` alone keeps the
other five span fields — most of the dump's bytes — from ever being materialised. Widening it makes
parsing 38 GB much slower.

## Data notes

- `es_metadata`: ~38 GB of NDJSON, one file per company, 229 companies, 38.9M articles, from 2024-07-01.
- Titles are embedded **unique only** — 11.7M of 38.9M (30%), a 3.3× saving. `embed_titles_sync.py` is
  resumable via atomically-written parquet shards; measured 1,244 titles/s, 2h36m, $5.77 for the full
  set with text-embedding-3-small at 256 dims.
- Each scores parquet carries its `KNNParams` in the file's key-value metadata, so a file records what
  produced it; `nk_manifest()` reads that back and is how the notebooks discover which windows exist.
- `source_name` is populated and currently unused; ~12.8% of the flow is machine-generated price
  bulletins. `teaser` exists in the schema but is 0% populated — this data is title-only.
- Which article fields the Quant API itself can serve is measured live in `test.py`, against the
  spec at <https://quant.api.yukkalab.com/openapi.json> — kept there rather than written down here,
  since it changes without notice.

## Regenerating

```bash
# 1. embed unique titles into the store (once)
OPENAI_API_KEY=sk-... uv run src/honey/novelty_score_exp/embed_titles_sync.py

# 2. score one lookback window (repeat per window)
uv run src/honey/novelty_score_exp/novelty_knn_model.py \
  --window-hours 336 --n 1 3 5 7 10 20 50 70 100 --max-cand 0 \
  --out data/es_title_embeddings/novelty_scores_w336h.parquet
```

`--max-cand 0` means the window, not a candidate cap, is the binding lookback.

Outputs under `cache/`, `figures/`, `reports/` and `results*/` are gitignored — all regenerable, and
plotly inlines its whole JS bundle per chart (~4.6 MB each), so committing them would cost ~150 MB.

## The ES titles dump (`title_dump.py`)

A second, differently-shaped export lives in `data/es_titles_dump/` as NDJSON shards named
`dump-NNNNN` (no extension), hash-partitioned so **every shard spans the whole date range** —
sampling one shard samples the corpus, and a missing shard is not a missing period.

Records carry three fields: `id`, `title`, `published`. **No entity, no sentiment, no
`source_name`, no `duplicate_group_id`, no `lang`.** That is enough to extend the embedding
store, which is keyed by title alone, and *not* enough to score novelty, which needs the company
an article is about and the sentiment attached to it. Do not plan scoring work on this file
without a matching metadata export; the `id` is the join key if one arrives.

It is also not the old dump extended. Measured across the two: only ~5% of its titles appear in
`es_metadata`, and even inside the window both cover (2024-07 onward) roughly four in five of its
titles are absent there. Different selection, not a longer history of the same one. The calendar
has a real hole — **2021-10 through 2024-01 is absent** — and coverage is uneven by a factor of
about fifty, with 2021 alone contributing more rows than 2024-2026 together.

Run `python -m honey.novelty_score_exp.title_dump` to reduce the shards to
`_unique_titles.parquet` (leading underscore keeps it out of the `dump-*` glob). Roughly ten
seconds for 180 shards. `embed_titles_sync.py` reads that parquet rather than rescanning 6 GB of
JSON on every resume, and tracks what is already embedded by **hashing** shard titles rather than
holding them as strings — at these counts a Python set of titles costs several GB and would
dominate the run.

## The ES body dump (`body_dump.py`)

A third export, in `data/es_body_dump/`, is the first one with **full article body text**. Same shard
shape as the titles dump — NDJSON named `dump-NNNNN` — but the records are `id`, `text`, `published`
and there is **no `title` field**: the headline is the *first line* of `text`. Over a 3000-record
sample that line runs 41 chars at p10, 66 at the median and 119 at p90, and reads as a headline
throughout; `body_dump.title_expr()` is that rule.

The shards arrive `.zst`-compressed with a **2^27 (128 MiB) window**, which is exactly zstd's default
decode memory limit, so plain `zstd -d` decompresses them and `--long=27` is redundant — it would only
become necessary at `--long=28` or beyond. 11 of the 12 carry an XXH64 checksum and all pass `zstd -t`.
The compressed originals are kept beside the decompressed shards, so `SHARD_GLOB` is
`dump-[0-9][0-9][0-9][0-9][0-9]` — five digits, no extension — rather than the titles dump's `dump-*`.
`dump-00003.zst` arrived truncated and is parked as `dump-00003.zst.truncated`; re-download it and the
shard count goes 12 → 13.

Measured over the 12 intact shards: 47.0 GB of JSON, 9.24M rows, 9.11M unique ids, **5.60M unique
titles**, reduced in 2m45s. **Its ids are disjoint from the titles dump** — 0 of 3000 sampled ids
appear in any of the 244 `es_titles_dump` shards — and only 4.2% of its titles overlap, so it is
additional coverage rather than a longer history of the same articles. It is also the smaller export:
9.1M articles against the titles dump's 33.2M.

Like the titles dump it carries no entity and no sentiment, so it does not make novelty scoreable on
its own.

### What it covers — read this before planning body-text work

Body text spans 2016-01..2026-04 but is **concentrated in the early years**, which is the opposite of
where the strategy trades:

| era | titles dump | body dump |
|---|---:|---:|
| 2016-2019 | 1,287,510 (2.7%) | **7,931,497 (85.8%)** |
| 2020-2021 | 30,707,624 (63.3%) | 952,699 (10.3%) |
| 2022-2023 | 0 | 187,193 (2.0%) |
| 2024-07 onward | 14,663,734 (30.2%) | **163,095 (1.8%)** |

So body text does *not* lift the title-only ceiling in the `es_metadata` / production window — there are
only 163k body-text rows there. It lifts it for 2016-2019, a period the titles dump barely reaches and
`es_metadata` (2024-07 onward) does not reach at all. Use it to build and validate a duplicate
classifier on real body text, not to score the live book.

**The two dumps are complementary rather than overlapping.** Only **711 shared article ids out of
54,239,042**, and their calendar gaps are disjoint: the titles dump is missing **2021-10..2024-01**, the
body dump is missing **2020-01..2020-10** (2020 has 8 rows total). Between them the only fully absent
stretch is 2022-2023 in the titles dump, which the body dump covers thinly.

It is hash-partitioned like the titles dump — every shard spans the full 2016-01..2026-04 range, 87-112
distinct months each — so the truncated `dump-00003` costs a random ~8%, not a date range.

## Both dumps feed the embedder

`embed_titles_sync.py` reads `SOURCES` — `title_dump` and `body_dump` — and embeds the **deduplicated
union**: 23,022,053 + 5,598,614 with 234,637 shared is 28,386,030 unique titles. The store is keyed by
title alone, so a title in both dumps is paid for once, and a title already embedded from the titles
dump is not re-embedded when the body dump introduces it again. A dump whose parquet has not been built
is skipped rather than fatal — only one may exist on a given machine — but the run prints the
per-source breakdown and names any absent source, so a silently missing dump is visible in the log.

`openai` is a PEP 723 inline dep of that script, not a project dependency, so its import is deferred
into `main()`; the module itself imports under pytest.
