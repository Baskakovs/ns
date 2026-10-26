"""Embedding novelty scoring, and the signal/portfolio plumbing built on it.

    uv run src/honey/novelty_score_exp/novelty_knn_model.py --help

Each article's headline is compared against the same company's *earlier* headlines in a lookback window:

    novelty = 1 - mean(cosine similarity to the n nearest earlier titles)

A verbatim reprint scores ~0, a new story ~1. Causal — only earlier articles are candidates.

The score reaches a signal two ways: `threshold_long` (keep `novelty > theta`, count survivors) or
`weighted_long` (each article contributes its own novelty, no theta). Both feed `build_arm`, which pins
honey's `build_portfolio` to `PROD_PARAMS` — production's own settings — so every arm shares one book
and a difference in outcome can only come from the signal.

CLAUDE.md beside this file covers the implementation traps; book/marimo/notebooks/NOVELTY.md
covers what the experiment is and what it found.
"""

import argparse
import glob
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
from jquantstats import Portfolio

from honey.strategy.sentiment import build_portfolio as build_sentiment

# ====================================================================================================================
# WHERE THE DATA LIVES
#
# Paths and the one date constant. Both halves below need these, so they sit above both.
# ====================================================================================================================

_DATA = Path(__file__).resolve().parents[3] / "data"  # repo-root data/ (gitignored)
#: Hard lower bound on usable article data — the dump starts here, so every panel is cut to it.
ARTICLE_DATA_START = date(2024, 7, 1)
ARTICLE_DATA_END = date(2030, 1, 1)

EMB = str(_DATA / "es_title_embeddings")
DUMP = str(_DATA / "es_metadata")


def discover_dump(dump_dir: str) -> dict[str, str]:
    """Map yukka_id -> file path for a per-company dump directory (reads each file's first record)."""
    import json
    from pathlib import Path

    out: dict[str, str] = {}
    for p in sorted(Path(dump_dir).glob("company_*.jsonl")):
        if p.stat().st_size == 0:
            continue
        with p.open() as f:
            first = f.readline()
        if not first.strip():
            continue
        entity = json.loads(first).get("entity")
        if entity:
            out[entity] = str(p)
    return out


# ====================================================================================================================
# PART 1 — SCORING:  dump + embeddings -> a novelty score per article
#
# Self-contained: needs numpy, the memmapped vector store and the article dump, and knows
# nothing about portfolios. Read in order: run parameters, the store, the block scorer,
# then the dump parsing that drives it.
# ====================================================================================================================

#: The live HTML dashboard lives in `status_dashboard.py`, which is deliberately optional and not
#: committed — a scoring run should not depend on its own progress display. Without it, `write_status`
#: becomes a no-op and the run is otherwise identical.
if TYPE_CHECKING:

    def write_status(*args: Any, **kwargs: Any) -> None:
        """Declared for the type checkers, which never see either branch below."""

else:
    try:
        from honey.novelty_score_exp.status_dashboard import write_status
    except ImportError:  # pragma: no cover - the dashboard is an optional local convenience

        def write_status(*_args, **_kwargs) -> None:
            """No-op stand-in when `status_dashboard.py` is absent."""


@dataclass
class Progress:
    """One snapshot of a scoring run, for the optional status dashboard."""

    phase: str
    subtitle: str = ""
    done: int = 0
    total: int = 0
    articles: int = 0
    novel_pct: float = 0.0
    elapsed: float = 0.0
    eta: float | None = None


#: Parquet key under which a run's `KNNParams` are stored, so a scores file says what produced it.


PARAMS_KEY = "novelty_params"


@dataclass(frozen=True)
class KNNParams:
    """The knobs: lookback window, the n's to average over, novelty threshold, candidate cap.

    `n_values` holds every neighbour count to emit — one `novelty_n{n}` column each. They all come out
    of a single pass: the cost is the top-`max(n)` selection, which `np.partition` does in time
    independent of the cut point, so a wide n grid is free (measured flat from n=5 to n=256).
    """

    window_s: float = 24 * 7 * 3600  # 7-day causal window
    n_values: tuple[int, ...] = (5,)  # nearest earlier titles to average — one novelty column per value
    theta: float = 0.30  # only for the calibration summary; is_novel is derived downstream
    max_cand: int = 2000  # most-recent earlier titles to compare against; <= 0 = no cap, window only

    @property
    def n_max(self) -> int:
        """Largest neighbour count — the depth the top-k selection has to reach."""
        return max(self.n_values)

    @property
    def window_hours(self) -> float:
        """Lookback window in hours."""
        return self.window_s / 3600

    @property
    def window_days(self) -> float:
        """Lookback window in days — the nominal one; `max_cand` may bind first for busy companies."""
        return self.window_s / 86400

    @property
    def window_label(self) -> str:
        """Compact window tag for filenames and chart labels (`6h`, `3d`, …)."""
        return f"{self.window_hours:g}h" if self.window_hours < 48 else f"{self.window_days:g}d"

    def describe(self) -> str:
        """One-line human label for charts and dashboards."""
        cap = "uncapped" if self.max_cand <= 0 else f"max_cand={self.max_cand:,}"
        return f"window {self.window_label}, n={list(self.n_values)}, {cap}"


def novelty_col(n: int) -> str:
    """Column name holding novelty at neighbour count `n`."""
    return f"novelty_n{n}"


def params_metadata(p: KNNParams) -> dict[str, str]:
    """`KNNParams` as parquet key-value metadata (JSON under `PARAMS_KEY`)."""
    extra = {"window_days": p.window_days, "window_hours": p.window_hours, "window_label": p.window_label}
    return {PARAMS_KEY: json.dumps(asdict(p) | extra)}


def read_params(path: str | Path) -> KNNParams | None:
    """Recover the `KNNParams` a scores parquet was written with (None for files predating the metadata)."""
    raw = pl.read_parquet_metadata(str(path)).get(PARAMS_KEY)
    if not raw:
        return None
    d = json.loads(raw)
    d = {k: v for k, v in d.items() if k in KNNParams.__dataclass_fields__}
    if "n_values" in d:
        d["n_values"] = tuple(d["n_values"])
    return KNNParams(**d)


#: Fixed seed for the title hash, so the store side and the lookup side agree across processes
#: (unlike Python's own `hash`, which is salted per interpreter).
_HASH_SEED = 0


@dataclass(frozen=True)
class Store:
    """Title embeddings: one (N, dim) f32 matrix plus a sorted-hash index into its rows.

    Hashing the titles rather than keeping a `{title: row}` dict costs ~190 MB instead of ~3 GB at
    N=11.7M, and turns per-title lookup into one vectorised `searchsorted`. A 64-bit collision would
    silently hand back the wrong vector, but at this N the expected count is ~4e-6.
    """

    mat: np.ndarray  # (N, dim) float32, unit-norm rows
    keys: np.ndarray  # (N,) uint64 title hashes, ascending
    rows: np.ndarray  # (N,) row of `mat` for the correspondingly-positioned key


def _hash_titles(titles: pl.Series) -> np.ndarray:
    """Vectorised uint64 hash of a title column — the shared key for both sides of the store."""
    return titles.hash(seed=_HASH_SEED).to_numpy()


def _cache_paths(emb_dir: str) -> tuple[Path, Path, Path]:
    """The three files the memmapped store lives in (matrix, sorted hashes, row index)."""
    d = Path(emb_dir)
    return d / "_store_mat.npy", d / "_store_keys.npy", d / "_store_rows.npy"


def build_store_cache(emb_dir: str) -> None:
    """Decode the parquet shards once into a memmappable f32 matrix + sorted title-hash index.

    The shards hold float64 lists, so re-reading them costs ~45 s and ~12 GB of anonymous memory that
    the OS then has to swap. Writing the f32 matrix straight to a `.npy` lets later runs memmap it:
    the pages become file-backed and evictable, and startup drops to nothing.
    """
    mat_p, keys_p, rows_p = _cache_paths(emb_dir)
    shards = sorted(glob.glob(f"{emb_dir}/titles_emb_*.parquet"))
    n = sum(pl.scan_parquet(s).select(pl.len()).collect().item() for s in shards)
    dim = len(pl.read_parquet(shards[0], columns=["embedding"])["embedding"][0])
    mat = np.lib.format.open_memmap(mat_p, mode="w+", dtype=np.float32, shape=(n, dim))
    keys = np.empty(n, dtype=np.uint64)
    pos = 0
    for s in shards:
        d = pl.read_parquet(s)
        vecs = d["embedding"].list.to_array(dim).to_numpy()  # (rows, dim) contiguous — no boxed Python floats
        mat[pos : pos + len(vecs)] = vecs  # single vectorised cast, written straight through to disk
        keys[pos : pos + len(vecs)] = _hash_titles(d["title"])
        pos += len(vecs)
    mat.flush()
    order = np.argsort(keys, kind="stable")  # repeated titles embed identically, so any of their rows will do
    np.save(keys_p, keys[order])
    np.save(rows_p, order)


def load_store(emb_dir: str, rebuild: bool = False) -> Store:
    """Memmap the title-embedding store, building the cache from the parquet shards on first use."""
    mat_p, keys_p, rows_p = _cache_paths(emb_dir)
    if rebuild or not (mat_p.exists() and keys_p.exists() and rows_p.exists()):
        build_store_cache(emb_dir)
    return Store(np.load(mat_p, mmap_mode="r"), np.load(keys_p), np.load(rows_p))


def vectors_for(titles: list[str], store: Store) -> np.ndarray:
    """Look up each title's vector (zero-vector where a title has no embedding — the ~0.1% junk titles)."""
    h = _hash_titles(pl.Series(titles, dtype=pl.Utf8))
    pos = np.searchsorted(store.keys, h)
    hit = pos < len(store.keys)
    hit[hit] = store.keys[pos[hit]] == h[hit]  # searchsorted only bounds the slot — confirm the key matches
    v = np.zeros((len(titles), store.mat.shape[1]), dtype=np.float32)
    rows = store.rows[pos[hit]]
    order = np.argsort(rows)  # walk `mat` forwards: on a 12 GB memmap this is ~4x a scattered gather
    v[np.flatnonzero(hit)[order]] = store.mat[rows[order]]
    return v


#: Articles per similarity block. Each block is one GEMM over `block + candidates` rows, so a small
#: block wastes fewer pairs on the masked-out upper triangle while staying big enough for BLAS.
BLOCK = 256

#: Below this many articles the thread pool costs more than the work it saves — score inline instead.
_THREAD_MIN = 8192


def _score_block(
    lohi: tuple[int, int],
    start: np.ndarray,
    ix: np.ndarray,
    vecs: np.ndarray,
    n_values: tuple[int, ...],
    out: np.ndarray,
    has_vec: np.ndarray,
) -> None:
    """Score one block of articles against its candidate range, writing into `out` in place.

    Row-major `(articles, candidates)` so each row's partition walks contiguous memory; non-causal and
    un-embedded pairs are masked to -inf. One `np.partition` at max(n) plus a cumsum yields every n.
    Selection dominates the cost here, not the matmul — see CLAUDE.md before restructuring.
    """
    i0, i1 = lohi
    c0 = int(start[i0])  # `start` is non-decreasing, so i0's candidate window opens first
    sims = vecs[i0:i1] @ vecs[c0:i1].T  # unit vectors -> dot = cosine; (block, candidates)
    cand = ix[c0:i1][None, :]
    sims[(cand < start[i0:i1, None]) | (cand >= ix[i0:i1, None])] = -np.inf  # out-of-window / not earlier
    sims[:, ~has_vec[c0:i1]] = -np.inf  # an un-embedded candidate is unknown, not dissimilar
    k = min(n_values[-1], sims.shape[1])  # n_values is sorted ascending; reach the deepest n once
    top = -np.sort(-np.partition(sims, -k, axis=1)[:, -k:], axis=1)  # k best per article, descending
    valid = np.isfinite(top)  # counts real neighbours: honours the window *and* the embedding mask
    avail = valid.sum(1)
    np.copyto(top, 0.0, where=~valid)  # -inf padding contributes nothing to the running sum
    csum = np.cumsum(top, axis=1)  # prefix sums -> every n's mean in one shot
    for j, n in enumerate(n_values):
        take = np.minimum(n, avail)  # can't average over more neighbours than exist
        m = (take > 0) & has_vec[i0:i1]  # un-embedded queries stay NaN
        rows = np.flatnonzero(m)
        out[i0:i1, j][m] = 1.0 - (csum[rows, take[m] - 1] / take[m]).astype(np.float32)


def knn_novelty(
    times: np.ndarray,
    vecs: np.ndarray,
    p: KNNParams,
    workers: int | None = None,
    has_vec: np.ndarray | None = None,
) -> np.ndarray:
    """Per article: 1 - mean cosine-sim to its n nearest EARLIER neighbours in the window (1 if none).

    Returns `(len(times), len(p.n_values))`; column j is novelty at `sorted(p.n_values)[j]`. All n come
    from one selection, so a wide n grid is nearly free. Blocks of `BLOCK` articles are farmed out to
    threads (numpy drops the GIL in matmul); `workers` defaults to min(12, cpu count).

    Titles with no embedding return **NaN**, never 1.0 — a zero vector scores cosine 0 against
    everything, which would read as "nothing similar came before" and award maximum novelty to scraper
    residue. `has_vec` defaults to "the row is not all-zero".
    """
    n_art = len(times)
    n_vals = tuple(sorted(p.n_values))
    if has_vec is None:
        has_vec = np.einsum("ij,ij->i", vecs, vecs) > 0  # squared norm; cheaper than np.linalg.norm
    nov = np.full((n_art, len(n_vals)), np.nan, dtype=np.float32)
    nov[has_vec] = 1.0  # embedded but nothing earlier in the window -> genuinely novel
    if n_art < 2:
        return nov
    ix = np.arange(n_art)
    # first index still inside the lookback window, then the max_cand cap — both non-decreasing in i.
    # max_cand <= 0 means no cap: the window alone bounds the candidate set (see KNNParams.max_cand).
    start = np.searchsorted(times, times - p.window_s, side="left")
    if p.max_cand > 0:
        start = np.maximum(start, ix - p.max_cand)
    bounds = [(i0, min(i0 + BLOCK, n_art)) for i0 in range(0, n_art, BLOCK)]

    def do(b: tuple[int, int]) -> None:
        """Score one `(lo, hi)` block of articles in place."""
        _score_block(b, start, ix, vecs, n_vals, nov, has_vec)

    if n_art < _THREAD_MIN:  # small company: pool setup would dominate
        for b in bounds:
            do(b)
        return nov
    with ThreadPoolExecutor(workers or min(12, os.cpu_count() or 4)) as ex:
        list(ex.map(do, bounds, chunksize=16))
    return nov


#: Only the fields the kNN scorer needs. A partial schema makes polars skip the rest of each record, and
#: narrowing the span struct to `sentiment` keeps the other five span fields — most of the dump's bytes —
#: from ever being materialised.
_DUMP_SCHEMA: dict[str, Any] = {
    "publish_time": pl.Utf8,
    "title": pl.Utf8,
    "matches_spans": pl.List(pl.List(pl.Struct({"sentiment": pl.Int64}))),
}


def load_company(path: str) -> pl.DataFrame:
    """Parse one company file into time-sorted on-topic `(ts, title, sentiment)` rows.

    The polars NDJSON reader in place of `model.load_flat_arts`' per-line `json.loads`: ~10x faster and
    it yields real UTC timestamps. (`load_flat_arts` returns naive datetimes whose `.timestamp()` is
    read as *local* time, which shifts by an hour across DST and breaks the sorted-times assumption.)
    Mirrors `model.parse_flat_sentiment`: on_topic = at least one span, sentiment = sign of the span sum.
    """
    return (
        pl.scan_ndjson(path, schema=_DUMP_SCHEMA)
        .with_row_index("row")
        .explode("matches_spans")  # list[list[struct]] -> list[struct]
        .explode("matches_spans")  # -> struct, null for an article with no spans
        .group_by("row")
        .agg(
            pl.first("publish_time"),
            pl.first("title"),
            pl.col("matches_spans").is_not_null().sum().alias("n_spans"),
            pl.col("matches_spans").struct.field("sentiment").fill_null(0).sum().alias("tone"),
        )
        .filter(pl.col("n_spans") > 0)  # on-topic only
        .select(
            pl.col("publish_time").str.to_datetime("%Y-%m-%dT%H:%M:%SZ").alias("ts"),
            pl.col("title"),
            pl.col("tone").sign().cast(pl.Int64).alias("sentiment"),
        )
        .sort("ts")
        .collect()
    )


def score_company(entity: str, path: str, store: Store, p: KNNParams) -> pl.DataFrame | None:
    """One company's articles -> `(entity, date, sentiment, novelty_n*)`, or None if it has none.

    Novelty is float32 and un-embedded titles stay null, so a missing vector cannot pass for high
    novelty downstream.
    """
    arts = load_company(path)
    if arts.is_empty():
        return None
    times = arts["ts"].dt.epoch("s").to_numpy().astype(float)
    vecs = vectors_for(arts["title"].to_list(), store)
    nov = knn_novelty(times, vecs, p)
    return arts.select(
        pl.lit(entity).alias("entity"),
        pl.col("ts").dt.date().alias("date"),
        "sentiment",
        *[pl.Series(novelty_col(n), nov[:, j]).fill_nan(None) for j, n in enumerate(sorted(p.n_values))],
    )


# ====================================================================================================================
# PART 2 — SIGNAL:  novelty scores -> daily counts -> a portfolio -> a comparable number
#
# Needs honey.data / honey.strategy and never touches an embedding. `threshold_long` and
# `weighted_long` are the two ways a score can reach a signal; everything else is shared
# plumbing so that every arm is built and measured identically.
# ====================================================================================================================

#: The **production sentiment leg**, parameter for parameter, as
#: `book/marimo/notebooks/sentiment_momentum_top200.py` builds it (its `sent_params`).
#:
#: This is deliberately not a generic backtest config. Every arm in this experiment is built by
#: handing `build_sentiment` a different `counts`/`ratio_wide` and changing nothing else, so a
#: difference in Sharpe can only come from the signal. The book itself is *not* top-decile: it is a
#: threshold book — hold every name whose robust cross-sectional z exceeds `threshold`, equal-weight
#: but capped at `max_weight` each, residual in cash — then scaled to `vol_target`.
PROD_PARAMS: dict[str, Any] = {
    "span": 19,
    "clip_c": 4.2,
    "threshold": 1.15,
    "max_weight": 0.1,
    "rebalance_freq": 10,
    "vol_com": 20,
    "vol_target": 0.2,
}


def build_arm(
    prices: pl.DataFrame,
    ratio_wide: pl.DataFrame,
    sector_mask: pl.DataFrame | None = None,
    sector_cap: float | None = None,
    **overrides: Any,
) -> Portfolio:
    """Build one arm through the production sentiment book; `overrides` patch `PROD_PARAMS`.

    The single place any portfolio here is built, so production and every novelty arm share a book by
    construction. `sector_cap=None` (production's own setting) leaves `sector_mask` inert. A threshold
    book can hold cash on a day when no name clears the bar, which makes `_robust_signal` nanmean an
    all-nan row; that warning is suppressed here rather than at every call site.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return build_sentiment(
            prices=prices,
            ratio_wide=ratio_wide,
            sector_mask=sector_mask,
            sector_cap=sector_cap,
            **(PROD_PARAMS | overrides),
        )


def threshold_long(scores: pl.DataFrame, theta: float) -> pl.DataFrame:
    """**Threshold path.** Keep `novelty > theta`, count the survivors -> daily `(date, entity, pos/neu/neg)`.

    The straightforward reading of a novelty score: decide what counts as new, then count articles as
    production does. `theta` is a free parameter, and it doubles as a sample-size knob — raising it makes
    the signal more selective *and* estimates it from less data, which is why `weighted_long` below
    exists as the alternative.
    """
    return (
        scores.filter(pl.col("novelty") > theta)
        .group_by("date", "entity")
        .agg(
            (pl.col("sentiment") > 0).sum().alias("positive"),
            (pl.col("sentiment") == 0).sum().alias("neutral"),
            (pl.col("sentiment") < 0).sum().alias("negative"),
        )
        .sort("date", "entity")
    )


def weighted_long(scores: pl.DataFrame) -> pl.DataFrame:
    """**Weighting path.** Each article contributes its own novelty to its sentiment bucket — no theta.

    `threshold_long` discards the score twice: an article that just clears theta counts the same as a
    wholly new story, and theta doubles as a sample-size knob. Weighting keeps the full continuum and
    removes the parameter. Null novelty (no embedding) is dropped rather than weighted.
    """
    return (
        scores.drop_nulls("novelty")
        .group_by("date", "entity")
        .agg(
            pl.col("novelty").filter(pl.col("sentiment") > 0).sum().alias("positive"),
            pl.col("novelty").filter(pl.col("sentiment") == 0).sum().alias("neutral"),
            pl.col("novelty").filter(pl.col("sentiment") < 0).sum().alias("negative"),
        )
        .sort("date", "entity")
    )


def window_sharpe(pf: Portfolio, lo: date, hi: date) -> float:
    """Sharpe of a portfolio truncated to [lo, hi] (nan-safe) — the scoring primitive for a fold."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = pf.truncate(lo, hi).stats.sharpe()["returns"]
    return float("nan") if s is None or s != s else float(s)


def sharpe_se(n_days: int, periods_per_year: float = 252.0) -> float:
    """Standard error of an annualised Sharpe estimated from `n_days` observations (~sqrt(ppy/n)).

    Worth printing next to every fold number. A 2-year article history split into four folds leaves
    ~80-day validation blocks, where this is ~1.8 — larger than any edge being looked for, so a fold
    Sharpe of 3 and a fold Sharpe of 1 are not distinguishable and the search is mostly ranking noise.
    """
    return float(np.sqrt(periods_per_year / max(n_days, 1))) if n_days > 0 else float("nan")


#: novelty = 1 - cosine similarity, computed in float32. For a near-identical reprint the similarity
#: lands a hair above 1.0 and the score comes out *negative* — 28.6% of articles, down to -0.0016.
#: Left alone that is wrong in two ways: as a weight, a perfect duplicate would *subtract* from its
#: sentiment bucket rather than contribute nothing; and any fractional power of it is NaN, which
#: silently poisons a whole day's counts. Clipping to [0, 1] is the definition the score was always
#: meant to have. Applied in the loaders so every consumer inherits it.
CLIP_NOVELTY = pl.col("novelty").clip(0.0, 1.0)


def scores_for(window: str, n: int, id_map: dict[str, str], universe: list[str]) -> pl.DataFrame:
    """One grid cell's per-article scores, RIC-mapped, universe-filtered, novelty clipped to [0, 1]."""
    row = next(r for r in nk_manifest() if r[0] == window)
    col = novelty_col(n)
    return (
        pl.read_parquet(row[2], columns=["entity", "date", "sentiment", col])
        .rename({col: "novelty"})
        .with_columns(pl.col("entity").replace(id_map), CLIP_NOVELTY)
        .filter(pl.col("entity").is_in(universe))
    )


def nk_manifest(emb_dir: str | Path = EMB) -> list[tuple[str, float, Path, KNNParams]]:
    """Built scores parquets as `(window_label, window_hours, path, params)`, shortest window first."""
    out = []
    for path in Path(emb_dir).glob("novelty_scores_w*h.parquet"):
        p = read_params(path)
        if p is not None:
            out.append((p.window_label, p.window_hours, path, p))
    return sorted(out, key=lambda r: r[1])


# ====================================================================================================================
# COMMAND LINE
#
# One scoring run: load the store once, score every company, write one parquet stamped
# with the KNNParams that produced it.
# ====================================================================================================================


def main() -> None:
    """Load embeddings once, score every company, write one per-article novelty parquet."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb", default=EMB)
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--out", default=f"{EMB}/novelty_scores.parquet")
    ap.add_argument("--theta", type=float, default=KNNParams.theta, help="only used for the calibration summary")
    ap.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=list(KNNParams.n_values),
        help="neighbour counts to emit, one novelty_n{n} column each; a wide grid is free (single selection)",
    )
    ap.add_argument("--window-hours", type=float, default=KNNParams.window_s / 3600)
    ap.add_argument(
        "--max-cand",
        type=int,
        default=KNNParams.max_cand,
        help="cap on candidates per article; 0 = no cap, so the window alone bounds the lookback "
        "(the busiest company holds ~77k articles in a 7d window, ~152k in 30d)",
    )
    ap.add_argument("--limit", type=int, default=None, help="score only the first N companies (benchmarking)")
    ap.add_argument("--rebuild-store", action="store_true", help="re-decode the parquet shards into the memmap cache")
    args = ap.parse_args()
    p = KNNParams(
        window_s=args.window_hours * 3600,
        n_values=tuple(sorted(set(args.n))),
        theta=args.theta,
        max_cand=args.max_cand,
    )
    out_dir = Path(args.out).parent
    t0 = time.time()

    write_status(out_dir, Progress("loading embeddings", subtitle="reading title vectors…"))
    store = load_store(args.emb, rebuild=args.rebuild_store)
    n_sorted = sorted(p.n_values)
    track = novelty_col(n_sorted[len(n_sorted) // 2])  # the dashboard's novel-rate needs one representative n
    sub = (
        f"{store.mat.shape[0]:,} vectors x {store.mat.shape[1]}d · {p.describe()} · "
        f"novel% at {track}>θ={p.theta} · → {Path(args.out).name}"
    )

    files = discover_dump(args.dump)
    if args.limit:
        files = dict(list(files.items())[: args.limit])
    total, frames, n_art, n_nov = len(files), [], 0, 0
    for i, (entity, path) in enumerate(files.items(), 1):
        df = score_company(entity, path, store, p)
        if df is not None:
            frames.append(df)
            n_art += df.height
            n_nov += int((df[track] > p.theta).sum())
        el = time.time() - t0
        write_status(
            out_dir,
            Progress(
                "scoring",
                subtitle=sub,
                done=i,
                total=total,
                articles=n_art,
                novel_pct=(n_nov / n_art * 100 if n_art else 0.0),
                elapsed=el,
                eta=el / i * (total - i),
            ),
        )
    all_df = pl.concat(frames)
    all_df.write_parquet(args.out, metadata=params_metadata(p))  # the file records what produced it
    write_status(
        out_dir,
        Progress(
            "complete",
            subtitle=f"wrote {Path(args.out).name} · {p.describe()}",  # keep the params on the finished page
            done=total,
            total=total,
            articles=n_art,
            novel_pct=(n_nov / n_art * 100 if n_art else 0.0),
            elapsed=time.time() - t0,
        ),
    )

    # calibration summary per n: theta is not comparable across n, so each column needs its own reading
    thetas = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
    print(f"wrote {args.out}  ({all_df.height:,} rows · {p.describe()})", flush=True)
    print(f"{'column':>12} {'mean':>7} {'median':>7} |  " + "  ".join(f"θ={t:.2f}" for t in thetas), flush=True)
    for n in n_sorted:
        nv = all_df[novelty_col(n)]
        rates = "  ".join(f"{cast(float, (nv > t).mean()):6.3f}" for t in thetas)
        print(
            f"{novelty_col(n):>12} {cast(float, nv.mean()):7.3f} {cast(float, nv.median()):7.3f} |  {rates}",
            flush=True,
        )
    print("=> pick theta per n so the novel rate lands near ~0.40 (the dup_group originator base rate).", flush=True)


if __name__ == "__main__":
    main()
