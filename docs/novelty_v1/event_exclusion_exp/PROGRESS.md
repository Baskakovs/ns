# Event-exclusion experiment — progress and handoff

State as of **2026-09-03**, after the full-history run. Design rationale, API routes and gotchas live
in `CLAUDE.md` next to this file; this document is only "where things stand and what to do next".

## The full-history run is done

Every artefact below covers **2015-10-01 → 2026-08-30**, 945 entities, 147 events.

| artefact | path | state |
|---|---|---|
| sentiment dump | `data/sentiment_dump` | 670/672 shards, 217.0M rows, 6.16 GB, 2013–2026 |
| events dump | `data/events_dump` | **6,288/6,288 shards**, 83.8M rows, 1.8 GB, 0 outstanding |
| price panels | `data/prices_hist/` | EUR 2,903×347 (`eur_wide` unmasked; `eur_wide_pit` 10.9% masked; **`eur_wide_prodmask` 35.7%, the one that reproduces production**), local 2,891×900, `id_bridge` 347 RICs |
| count panels | `data/panels_hist/` | **2,751,294 company-days**, 945 entities, 147 events (+ `_facts` variant) |
| results | `data/results_hist/` | 6 original arms + 3 production-statistic arms × 147 events × 3 horizons |
| notebook view | `data/hist/` | symlinks `panels`/`prices`/`results` so `EVENT_EXCLUSION_DATA=data/hist` just works |
| tests | `tests/test_event_exclusion.py` | 50 passing |

Reproduce with:

```bash
uv run src/honey/event_exclusion_exp/prices.py --out data/prices_hist --from 2015-08-01 --to 2026-09-01
uv run src/honey/event_exclusion_exp/panel.py  --out data/panels_hist --from 2015-10-01 --to 2026-09-01
uv run src/honey/event_exclusion_exp/panel.py  --out data/panels_hist --from 2015-10-01 --to 2026-09-01 --facts-only
bash src/honey/event_exclusion_exp/run_arms.sh   # the six IC arms
uv run src/honey/event_exclusion_exp/verify.py --panels data/panels_hist
EVENT_EXCLUSION_DATA=data/hist uv run marimo edit --sandbox book/marimo/notebooks/event_exclusion.py
```

`prices.py` masks by index membership **by default**, so its `eur_wide.parquet` is the PIT arm. The
unmasked headline panel is `to_wide(eur_long)` with no mask — cheaper to re-derive from the saved
`eur_long.parquet` than to re-pull, which is how `eur_wide.parquet` here was made.

## Findings

### 1. RETRACTED: the signal does not decay. That was a universe artefact.

An earlier version of this document reported the baseline IC decaying monotonically from +0.0202
(2015-2019) to -0.0143 (2024-2026) at h=21 and called it "the most important result in the
experiment". **That was wrong, in two compounding ways, and it is withdrawn.**

* **The experiment's price panel is not point-in-time.** `data/prices_hist/eur_wide.parquet` is
  2.3% null; production's `load_prices(200)` is **35.7%** null over the same 344 RICs and 2,902
  dates, because it holds a name only while that name is actually in the traded band. The extra
  third of the cells are names production would never have held on that day, and they carry the
  negative IC. Holding the signal, the RICs and the dates fixed and varying *only* the price panel:

  | price panel | null | 2015-19 h=21 | 2020-23 h=21 | 2024-26 h=21 |
  |---|---|---|---|---|
  | production `load_prices` | 35.7% | +0.0364 | +0.0157 | **+0.0086** |
  | `eur_wide` (experiment) | 2.3% | +0.0168 | −0.0022 | **−0.0190** |
  | `eur_wide_pit` (experiment's PIT arm) | 10.9% | +0.0177 | +0.0004 | −0.0170 |
  | `eur_wide` masked wherever production is null | 35.7% | +0.0283 | +0.0059 | **+0.0013** |

  The experiment's own PIT arm masks 10.9% where production effectively masks 35.7%, so it does not
  fix this. Masking to production's exact footprint recovers almost the whole gap.

* **The three-era table manufactured a trend.** Even on the corrected panel, the year-by-year series
  is noise around a small positive, not a slope: at h=21, `-0.011 +0.035 +0.048 +0.036 +0.016 +0.031
  -0.029 +0.005 -0.002 +0.003 -0.008` for 2016…2026 — eight of eleven positive, with 2022 the worst
  year, not 2026. Three coarse buckets on a series like that will show whatever the boundaries pick.

**The corrected baseline**, on the production-equivalent arm (`eur_prodmask`: Laplace statistic,
production smoothing, production's PIT footprint):

| era | h=1 | h=5 | h=21 |
|---|---|---|---|
| 2015-10 … 2019-12 | +0.0035 | +0.0157 | +0.0274 |
| 2020-01 … 2023-12 | +0.0059 | +0.0095 | +0.0055 |
| 2024-01 … 2026-08 | +0.0096 | +0.0068 | −0.0012 |
| **full sample** | **+0.0059** | **+0.0112** | **+0.0123** |

Positive at every horizon; at h=1 it *rises* across the eras. This lands on top of production's own
reading via `load_counts`/`_robust_signal` (+0.0054 / +0.0119 / +0.0125), which is the independent
check that the corrected arm is measuring the right object. `production_ic_history.py` already said
this — "over the full history the production signal's IC is positive at every horizon" — and it
should have been the first thing reconciled against.

Long-horizon IC being weakest in the last two years is the one durable part of the original claim,
and even that is ~0 rather than negative, on ~700 overlapping observations.

### 2. The signal under test was never production's signal

The experiment's statistic is `(pos - neg) / tot` on the raw dump, unsmoothed. Production trades
`(pos + 1) / (pos + neg + 2)` — neutral articles excluded — through `_robust_signal` (EW median span
19, lagged, clipped at 4.2, cross-sectionally z-scored), on `t_stat=2.5`-vetted counts. `CLAUDE.md`
warned the *levels* were not comparable; what it did not say is that the sign of the recent-era
trend is not comparable either.

`run_ic.py` now takes `--ratio {net,laplace}` and `--smooth`, so any arm can be run under
production's definition. Four arms are on disk: `eur` (original), `eur_laplace`, `eur_prod`
(statistic + smoothing) and `eur_prodmask` (**the production-equivalent arm — prefer this one**).

### 3. Merger survives every one of those corrections

It is the one result that did not move. Excluding Merger articles improves IC in all six original
arms *and* under production's statistic, smoothing and universe:

| arm | Δ h=1 | q | Δ h=5 | q | Δ h=21 | q |
|---|---|---|---|---|---|---|
| `eur` net ratio, raw | +0.00037 | 0.460 | +0.00075 | **0.001** | +0.00093 | **0.013** |
| `eur_laplace` production statistic | +0.00033 | 0.215 | +0.00071 | **0.002** | +0.00088 | **0.008** |
| `eur_prod` + smoothing | +0.00051 | **0.000** | +0.00087 | **0.001** | +0.00130 | **0.003** |
| `eur_prodmask` + PIT universe | +0.00060 | **0.003** | +0.00101 | **0.000** | +0.00102 | **0.040** |

0.56% of matches, 153,551 company-days, and stable across eras on the original arm (+0.00074 /
+0.00075 / +0.00075 at h=5). Against a corrected baseline of +0.0112 at h=5, +0.00101 is a **+9%
relative lift** — smaller than the +16% claimed against the broken baseline, and still the largest
single-event effect in the sweep.

**Product Launch** does not replicate on the original arm but *does* clear q<0.01 on the
production-equivalent one (+0.00038, q=0.006, h=1). Treat it as open, not settled either way.

The `eur_prodmask` survivor list is much longer (31 at q<0.10 versus 5) because smoothing removes
day-to-day noise from the paired difference and tightens every standard error. That makes the
ranking more sensitive, not more trustworthy — read `match_share` alongside it. `Currency Trading:
Down` tops it at share=0.00003, which is exactly the rare-event noise the funnel check exists to
catch.

## Data caveats attached to this run

* **2 sentiment shards are permanently missing** — `b013 2020`, `b004 2021` — both reproducible 502s
  on oversized responses after 4 attempts each, twice. `sentiment_sweep` has no date-bisection
  (unlike `_sweep.fetch_window`), so there is no way to fetch a half-year into a year-keyed shard
  without new code. `panel.py`'s integrity check sees them exactly: b004 loses 20.6% of its event
  articles, b013 8.7%. That is 40 entity-years of ~13,300. It drops those names from both arms
  equally, so it cannot bias a paired delta.
* **30,239 gap entity-days** (poisoned `events_dump` records), 786 entities, 107 periods — **1.099%**
  of the panel's company-days, worst in 2026 (~350/month). One-directional toward zero delta.
* **The EUR panel starts 2015-10-13**, not 2015-08-01 — `constituent_prices` returns nothing before
  that date, so the first fortnight of the events panel has no return warmup.
* `verify.py` check 4 finds 2 of 6 sampled sentiment shards ~1% short (adidas 2026, daimler 2026);
  both are the current year, consistent with articles being backfilled after the pull.
* `verify.py` check 5 (whole-panel baseline vs Data API export) reports 16.9% of company-days
  differing, but the large gaps are the **export** returning 0 for high-volume names — Barclays, UBS
  and Deutsche Bank all return `api_tot = 0` on 2024-02-10 while the dump has 1,273 / 1,055 / 783.
  Same false-alarm class as the sentiment-truncation scare below. Not a dump defect.

## Two fixes made to the tooling during this run

* **`verify.py --sentiment` defaulted to `data/sentiment_dump_monthly`** — the redundant 2024-only
  artefact of the false-alarm re-pull. Against an 11-year events dump that made check 2 report
  **85.1% of event pairs unmatched**, which reads as a catastrophic join failure and is purely the
  stale default's date range. Repointed to `data/sentiment_dump`; the check now passes at **0.084%**.
  `data/sentiment_dump_monthly` (612 files) is still on disk and can be deleted.
* **`verify.py` check 4 aborted the whole report on an API 500.** Those are the documented poisoned
  records, so a sampled shard that 500s is now recorded as unverifiable and the run continues —
  previously it took sections 4b, 5 and 6 down with it.

## Next steps, in the order I would do them

0. **Re-run the remaining five arms on `--ratio laplace --smooth` against a production-masked price
   panel.** Only the EUR headline arm has been corrected. `eur_wide_prodmask.parquet` is built by
   masking `eur_wide` wherever `load_prices(200)` is null; that should move into `prices.py` as a
   real option rather than living as a one-off. The facts-only, accumulate, month-end and
   local-currency arms are all still on the uncorrected panel and statistic.
1. **Threshold-basket forward return.** Still the highest-value change and still unchosen — see the
   open decision below. Now more pressing, not less: with a positive baseline IC over most of the
   sample there is finally a signal whose top slice is worth scoring.
2. **Write up the IC decay** (finding 1). It is a bigger result than anything the event ranking
   produced and it stands independently of the exclusion machinery.
3. **Consider re-pulling `b013 2020` and `b004 2021`** by teaching `sentiment_sweep` the same
   date-bisection `_sweep.fetch_window` already has. ~40 lines, removes the only real hole.
4. **Report 2015–2019 separately.** The gap rate there is ~4× the recent one and news volume per
   company-day is ~4× lower; the era table above is the right granularity, not a pooled average.

## Open decisions

**The metric is still the binding constraint.** Full-cross-section rank IC is not how this signal is
used — `honey.strategy.sentiment.build_portfolio` selects names by `signal > threshold`, a long-only
book on the top of the distribution. A signal can have ~zero rank IC across 300 names and still have
a profitable top slice. Scoring the same 147 arms on **threshold-basket forward return** is the one
change that could alter the conclusion, and it is a contained edit to `run_ic.py`. Raised four times
now, not yet chosen.

**No minimum-`tot` floor is applied**, by explicit decision. Median sentiment matches per company-day
runs 5 in 2016 to 21 in 2024, so early ratios are computed off a handful of articles. The era table
is the check on whether that distorts anything; at h=5 Merger it plainly does not.

**How much PIT masking is right?** Production's footprint is the traded band and is the correct
target for a levels question, but it discards two thirds of the cross-section and cannot answer
"does this event hurt the signal on names we might add". The experiment's own 10.9% mask is too
loose to reproduce production and too tight to be a clean wide-universe arm. Pick one deliberately
per question rather than inheriting whichever panel a script defaulted to.

**Price levels disagree in the tail.** On the 642k cells where both panels are non-null, median
relative difference is 0.0 but p99 is **0.377**. `CLAUDE.md` records a p99 of 2e-16, measured over
2023-11 onward only. Over the full history something (split or dividend adjustment, most likely)
diverges on a minority of names. Not diagnosed, and it does not drive the finding above — masking
does — but it should not be left unexplained.

**~45 entities are still excluded** because `price_data_itemized` returns only a US ADR for them (3i
Group, Akzo Nobel, Assa Abloy, Atlas Copco, Aviva…). Recoverable by dropping the `ADR_EXCHANGES`
filter in `prices.py`, but those ADRs close 4.5 h after the European names, so their `t+1` window is
not synchronous with the rest of the cross-section.

## Three things settled, so they are not re-litigated

**FX conversion is not needed, but returns are not FX-independent.** Same names, same signal, currency
the only difference: baseline IC at h=1 is −0.0018 in EUR and +0.0008 in local on the 32-month panel —
a swing exceeding every event effect measured. FX cancels *within* a currency bloc, not across eight
of them. But Δ IC is a paired difference on a common return panel, so the shift hits both arms and
mostly cancels: rank correlation of event deltas between EUR and local is +0.94 at h=1. **Rankings can
come from the wide local panel; absolute IC levels must not.**

**`data/sentiment_dump` is sound.** It was diagnosed mid-experiment as truncating high-volume entities
and a full re-pull was started. That was **wrong** — the comparison put a `drop_nulls("sentiment")`
count against the API's raw row count. Both pulls return exactly 69,951 rows for `company:ubs` in
January 2024. The real fact is that **8.6% of sentiment rows have a null sentiment** (54% for UBS),
almost all `non_professional` sources, and a null-sentiment pair never also appears with a real
sentiment. Dropping them is correct.

**Never compare a filtered count to a raw one.** Both false alarms in this experiment — the
truncation scare above and the 85% join failure this run — were denominator mistakes, not data
defects. Check what the other side of the comparison actually covers before believing a bad number.
