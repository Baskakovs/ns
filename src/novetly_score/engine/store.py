"""The title-embedding store: a memmapped matrix plus a sorted-hash index into it.

Three decisions here are load-bearing and a rewrite should keep all of them.

**The index is a sorted hash array, not a dict.** A ``{title: row}`` dict over 11.7M Python
strings was a ~3 GB allocation; the hash array is ~190 MB and turns lookup into one
vectorised ``searchsorted``. A 64-bit collision would silently return the wrong vector, but
at this N the expected number is ~4e-6.

**The hash seed is fixed.** Python's builtin ``hash`` is salted per interpreter, so a store
built in one process and read in another would miss on every key — and a miss is
indistinguishable from "this title was never embedded", which means the failure would look
like a data problem rather than a bug.

**The matrix is memmapped on purpose.** The parquet shards hold float64 lists; decoding them
costs ~45 s and ~12 GB of anonymous memory the OS then has to swap. Writing f32 straight to
a ``.npy`` makes later runs' pages file-backed and evictable, and startup free.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import polars as pl

from honey.novelty_score_exp.core.similarity import check_unit_norm, is_embedded

#: Fixed seed for the title hash, so the build side and the lookup side agree across
#: processes. See the module docstring on why a salted hash is worse than useless here.
HASH_SEED: Final = 0

#: Basenames of the three files the store lives in.
_CACHE_FILES: Final = ("_store_mat.npy", "_store_keys.npy", "_store_rows.npy")


@dataclass(frozen=True)
class Store:
    """Title embeddings plus the index that finds a title's row."""

    mat: np.ndarray  # (N, dim) float32, unit-norm rows
    keys: np.ndarray  # (N,) uint64 title hashes, ascending
    rows: np.ndarray  # (N,) row of `mat` for the correspondingly-positioned key

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return int(self.mat.shape[1])


class Lookup(NamedTuple):
    """The result of resolving titles to vectors.

    ``embedded`` travels with ``vectors`` because the two are only meaningful together: an
    unembedded row is all-zero, and a zero vector scores cosine 0 against everything, which
    reads as "nothing similar came before" and would award *maximum* novelty to scraper
    residue. This is the one place that mask is produced; no caller re-derives it.
    """

    vectors: np.ndarray  # (n, dim) float32, zero rows where the title had no embedding
    embedded: np.ndarray  # (n,) bool


def hash_titles(titles: pl.Series) -> np.ndarray:
    """Vectorised uint64 hash of a title column — the shared key for both sides of the store.

    Args:
        titles: Titles to hash.

    Returns:
        `(n,)` uint64 hashes.
    """
    return titles.hash(seed=HASH_SEED).to_numpy()


def cache_paths(emb_dir: str | Path) -> tuple[Path, Path, Path]:
    """The three files the memmapped store lives in (matrix, sorted hashes, row index).

    Args:
        emb_dir: Directory holding the store.

    Returns:
        Paths to the matrix, the sorted key array and the row index.
    """
    d = Path(emb_dir)
    return tuple(d / name for name in _CACHE_FILES)  # type: ignore[return-value]


def build_cache(emb_dir: str | Path) -> None:
    """Decode the parquet shards once into a memmappable f32 matrix and sorted hash index.

    Args:
        emb_dir: Directory holding the ``titles_emb_*.parquet`` shards; the cache lands beside them.

    Raises:
        FileNotFoundError: If the directory holds no embedding shards.
    """
    mat_p, keys_p, rows_p = cache_paths(emb_dir)
    shards = sorted(glob.glob(f"{emb_dir}/titles_emb_*.parquet"))
    if not shards:
        msg = f"no titles_emb_*.parquet shards in {emb_dir}"
        raise FileNotFoundError(msg)
    n = sum(pl.scan_parquet(s).select(pl.len()).collect().item() for s in shards)
    dim = len(pl.read_parquet(shards[0], columns=["embedding"])["embedding"][0])
    mat = np.lib.format.open_memmap(mat_p, mode="w+", dtype=np.float32, shape=(n, dim))
    keys = np.empty(n, dtype=np.uint64)
    pos = 0
    for s in shards:
        d = pl.read_parquet(s)
        vecs = d["embedding"].list.to_array(dim).to_numpy()  # contiguous — no boxed Python floats
        mat[pos : pos + len(vecs)] = vecs  # one vectorised cast, written straight to disk
        keys[pos : pos + len(vecs)] = hash_titles(d["title"])
        pos += len(vecs)
    mat.flush()
    order = np.argsort(keys, kind="stable")  # repeated titles embed identically, so any row will do
    np.save(keys_p, keys[order])
    np.save(rows_p, order)


def load(emb_dir: str | Path, rebuild: bool = False) -> Store:
    """Memmap the store, building the cache from the parquet shards on first use.

    Args:
        emb_dir: Directory holding the store.
        rebuild: Re-decode the shards even if a cache exists.

    Returns:
        The memmapped store.
    """
    mat_p, keys_p, rows_p = cache_paths(emb_dir)
    if rebuild or not (mat_p.exists() and keys_p.exists() and rows_p.exists()):
        build_cache(emb_dir)
    return Store(np.load(mat_p, mmap_mode="r"), np.load(keys_p), np.load(rows_p))


def validate(store: Store, *, sample: int | None = 200_000, seed: int = 0) -> None:
    """Check the store's vectors are unit-norm, on a sample by default.

    The full store is ~12 GB, so validating every row means reading all of it. A random
    sample catches a store built by the wrong pipeline, which is the failure this guards
    against; pass ``sample=None`` to check every row.

    Args:
        store: The store to check.
        sample: Rows to draw, or None for all of them.
        seed: RNG seed, so a reported failure is reproducible.

    Raises:
        ValueError: If any sampled vector is off unit norm. See
            :func:`honey.novelty_score_exp.core.similarity.check_unit_norm`.
    """
    n = store.mat.shape[0]
    if sample is None or sample >= n:
        rows = np.arange(n)
    else:
        rows = np.sort(np.random.default_rng(seed).choice(n, size=sample, replace=False))
    check_unit_norm(np.asarray(store.mat[rows], dtype=np.float64))


def vectors_for(titles: list[str], store: Store) -> Lookup:
    """Resolve titles to vectors, reporting which ones were found.

    A title with no embedding gets a zero row and a False in ``embedded``. The two must
    travel together — see :class:`Lookup`.

    Args:
        titles: Titles to look up.
        store: The store to look them up in.

    Returns:
        The vectors and the embedded mask.
    """
    h = hash_titles(pl.Series(titles, dtype=pl.Utf8))
    pos = np.searchsorted(store.keys, h)
    hit = pos < len(store.keys)
    hit[hit] = store.keys[pos[hit]] == h[hit]  # searchsorted bounds the slot — confirm the key
    v = np.zeros((len(titles), store.dim), dtype=np.float32)
    rows = store.rows[pos[hit]]
    order = np.argsort(rows)  # walk `mat` forwards: on a 12 GB memmap ~4x a scattered gather
    v[np.flatnonzero(hit)[order]] = store.mat[rows[order]]
    return Lookup(v, is_embedded(v))
