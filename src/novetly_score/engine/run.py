"""Score the whole dump for one `(f, window)` and write the stamped scores parquet.

    uv run python -m honey.novelty_score_exp.engine.run --window-hours 6

Every `k` in the grid comes out of the one pass (see :mod:`honey.novelty_score_exp.engine.scorer`),
so a run is per candidate and window, not per cell. The file lands at
:func:`honey.novelty_score_exp.engine.params.scores_path` and carries the scorer's fingerprint,
so it stops being readable the moment the definition behind it changes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from honey.novelty_score_exp.core.functions import DEFAULT_F, FAMILY
from honey.novelty_score_exp.engine import io, store
from honey.novelty_score_exp.engine.corpus import discover_dump
from honey.novelty_score_exp.engine.params import DUMP, EMB, KNNParams, scores_path
from honey.novelty_score_exp.engine.scorer import score_company

#: The grid every published run emitted.
K_GRID = (1, 3, 5, 7, 10, 20, 50, 70, 100)


def score(p: KNNParams, dump_dir: str | Path = DUMP, emb_dir: str | Path = EMB) -> Path:
    """Score every company in the dump under *p* and write one parquet.

    Args:
        p: The run's parameters.
        dump_dir: The per-company ``es_metadata`` files.
        emb_dir: The embedding store, where the scores parquet is also written.

    Returns:
        The parquet written.
    """
    st = store.load(emb_dir)
    companies = discover_dump(dump_dir)
    frames = []
    t = time.perf_counter()
    for i, (entity, path) in enumerate(sorted(companies.items()), 1):
        frame = score_company(entity, path, st, p)
        if frame is not None:
            frames.append(frame)
        print(f"[{i}/{len(companies)}] {entity} {time.perf_counter() - t:.0f}s", flush=True)
    return io.write_scores(pl.concat(frames), scores_path(emb_dir, p.window_hours, p.f), p)


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window-hours", type=float, required=True)
    ap.add_argument("--f", default=DEFAULT_F, choices=sorted(FAMILY))
    ap.add_argument("--k", type=int, nargs="+", default=list(K_GRID))
    ap.add_argument("--max-cand", type=int, default=0)
    args = ap.parse_args()
    p = KNNParams(window_s=args.window_hours * 3600, k_values=tuple(args.k), max_cand=args.max_cand, f=args.f)
    print(f"scoring {p.describe()}", flush=True)
    print(score(p))


if __name__ == "__main__":
    main()
