"""The only door to a scores parquet: writes them stamped, lists them, reads them checked.

Two rules this module exists to enforce.

**A file is read only if the code that wrote it is the code running now.** Every scores parquet
carries the fingerprint of :mod:`honey.novelty_score_exp.engine.scorer` and everything it
imports, `core/` included (:func:`honey.novelty_score_exp.twins.stamp`). Reading checks it, so
scores computed under a definition `core/` no longer holds raise
:class:`honey.novelty_score_exp.twins.StaleArtifactError` instead of flowing into a panel.

**Both scales, each under its own name.** A read returns ``raw_novelty`` and
``calibrated_novelty`` side by side. They share a dtype and a range, so a single column called
"novelty" would say nothing about which one it held — and `theta` means something only on the
calibrated scale.

What this module no longer does is clip. The stored values *are* the defined quantity —
:func:`honey.novelty_score_exp.engine.scorer.score_arrays` clips exactly where
:func:`honey.novelty_score_exp.core.novelty.raw_novelty` does — and the stamp is what
guarantees a file came from there. A second clip here would be a second copy of the rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, NamedTuple

import polars as pl

from honey.novelty_score_exp.core.functions import DEFAULT_F
from honey.novelty_score_exp.engine.params import (
    EMB,
    KNNParams,
    calibrated_novelty_col,
    params_metadata,
    raw_novelty_col,
    read_params,
)
from honey.novelty_score_exp.twins import StaleArtifactError, check_stamp, stamp

#: The module whose code every scores parquet is stamped with.
PRODUCER: Final = "engine.scorer"

#: Glob matching a built scores parquet. Filenames carry the candidate and the window in hours.
SCORES_GLOB: Final = "novelty_scores_*_w*h.parquet"

#: Columns a scores parquet carries besides its per-k novelty columns, in read order.
ID_COLUMNS: Final = ("article_id", "entity", "date", "sentiment")

#: The two scales, as a read names them.
RAW: Final = "raw_novelty"
CALIBRATED: Final = "calibrated_novelty"


class Built(NamedTuple):
    """One scores parquet on disk, and whether the running code would have written it."""

    label: str  # window label, e.g. ``6h``
    hours: float
    f: str
    path: Path
    params: KNNParams
    current: bool  # stamped by the scorer as it stands now


def write_scores(scores: pl.DataFrame, path: str | Path, p: KNNParams) -> Path:
    """Write one scoring run, stamped with its parameters and with the code that produced it.

    Written to a temporary file and renamed, so an interrupted run leaves no half-written
    parquet that :func:`manifest` would list as a complete set of scores.

    Args:
        scores: Per-article rows: the ID columns plus both novelty columns per ``k``.
        path: Destination parquet path.
        p: The run's parameters, recorded in the file's metadata.

    Returns:
        The path written.
    """
    dest = Path(path)
    tmp = dest.with_suffix(".tmp")
    scores.write_parquet(tmp, metadata=params_metadata(p) | stamp(PRODUCER))
    tmp.rename(dest)
    return dest


def _is_current(path: Path) -> bool:
    """Whether *path* carries the running scorer's stamp."""
    try:
        check_stamp(path, PRODUCER)
    except StaleArtifactError:
        return False
    return True


def manifest(emb_dir: str | Path = EMB) -> list[Built]:
    """Built scores parquets, shortest window first, each marked current or stale.

    How a caller discovers which windows exist. Files without recoverable parameters are
    skipped rather than guessed at; stale ones are listed, so a caller can say what needs
    re-scoring, but :func:`read_scores` will refuse them.

    Args:
        emb_dir: Directory holding the scores parquets.

    Returns:
        One :class:`Built` per readable file, sorted by window length then candidate.
    """
    out = []
    for path in Path(emb_dir).glob(SCORES_GLOB):
        p = read_params(path)
        if p is not None:
            out.append(Built(p.window_label, p.window_hours, p.f, path, p, _is_current(path)))
    return sorted(out, key=lambda b: (b.hours, b.f))


def columns_for(path: Path, k: int) -> tuple[str, str]:
    """The raw and calibrated column names for *k*, checked against the file.

    Args:
        path: Parquet path.
        k: Neighbour count.

    Returns:
        ``(raw column, calibrated column)``.

    Raises:
        KeyError: If the file lacks either, naming the neighbour counts it does hold.
    """
    have = set(pl.read_parquet_schema(str(path)))
    wanted = (raw_novelty_col(k), calibrated_novelty_col(k))
    if all(c in have for c in wanted):
        return wanted
    held = sorted(int(c.rsplit("_k", 1)[1]) for c in have if c.startswith(RAW + "_k"))
    msg = f"{path.name} has no novelty columns for k={k}; it holds k in {held}"
    raise KeyError(msg)


def read_scores(window: str, k: int, emb_dir: str | Path = EMB, f: str = DEFAULT_F) -> pl.DataFrame:
    """One grid cell's per-article scores: the id columns, ``raw_novelty`` and ``calibrated_novelty``.

    The single read door. Everything downstream of scoring comes through here, which is what
    makes the stamp check unconditional rather than a convention each caller has to remember.

    Args:
        window: Window label, e.g. ``7d`` or ``6h``.
        k: Neighbour count to read.
        emb_dir: Directory holding the scores parquets.
        f: The candidate function's name.

    Returns:
        :data:`ID_COLUMNS` plus both scales, nulls preserved where an article was never
        embedded.

    Raises:
        FileNotFoundError: If no file carries that window for *f*.
        StaleArtifactError: If the file was written by code other than the running scorer.
        KeyError: If the file has no columns for *k*.
    """
    built = manifest(emb_dir)
    match = next((b for b in built if b.label == window and b.f == f), None)
    if match is None:
        have = ", ".join(f"{b.f}@{b.label}" for b in built) or "none"
        msg = f"no scores parquet for {f} at window {window!r}; built: {have}"
        raise FileNotFoundError(msg)
    check_stamp(match.path, PRODUCER)
    raw, calibrated = columns_for(match.path, k)
    return pl.read_parquet(match.path, columns=[*ID_COLUMNS, raw, calibrated]).rename(
        {raw: RAW, calibrated: CALIBRATED}
    )


def scores_for(
    window: str,
    k: int,
    id_map: dict[str, str],
    universe: list[str],
    emb_dir: str | Path = EMB,
    f: str = DEFAULT_F,
) -> pl.DataFrame:
    """:func:`read_scores`, then mapped onto the price panel's identifiers and its universe.

    Args:
        window: Window label.
        k: Neighbour count.
        id_map: ``yukka_id -> RIC``, as the price panel keys names.
        universe: RICs to keep.
        emb_dir: Directory holding the scores parquets.
        f: The candidate function's name.

    Returns:
        The same frame, entity-mapped and universe-filtered.
    """
    return (
        read_scores(window, k, emb_dir, f)
        .with_columns(pl.col("entity").replace(id_map))
        .filter(pl.col("entity").is_in(universe))
    )
