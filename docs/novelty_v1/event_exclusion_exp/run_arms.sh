#!/usr/bin/env bash
# The six IC arms of the full-history run. Paths are the historical ones; edit P/R for a variant.
set -x
P=data/panels_hist
R=data/results_hist
B=data/prices_hist/id_bridge.parquet
run() { uv run src/honey/event_exclusion_exp/run_ic.py --panels $P --out $R --bridge $B "$@" || echo "ARM FAILED: $*"; }

run --tag eur          --prices data/prices_hist/eur_wide.parquet
run --tag eur_facts    --prices data/prices_hist/eur_wide.parquet --suffix _facts
run --tag eur_pit      --prices data/prices_hist/eur_wide_pit.parquet
run --tag eur_accum    --prices data/prices_hist/eur_wide.parquet --accumulate
run --tag eur_monthend --prices data/prices_hist/eur_wide.parquet --eval month-end
run --tag local        --prices data/prices_hist/local_wide.parquet --key yukka_id
echo ALL_ARMS_DONE
