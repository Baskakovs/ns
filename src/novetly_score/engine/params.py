"""The knobs a scoring run is fixed by, and the provenance it records.

Every scores parquet carries its :class:`KNNParams` in the file's key-value metadata, so a
file says what produced it and :func:`honey.novelty_score_exp.engine.io.manifest` can
recover it without a side-car index.

Naming follows the paper: the neighbour count is ``k`` (§2.5), the lookback duration ``w``
(§2.4, carried here as ``window_s``), the novelty cutoff ``theta`` (§3.1), and the candidate
function ``f`` (§2.5), by its name in :data:`honey.novelty_score_exp.core.functions.FAMILY`.

**There is no column called "novelty".** A file carries ``raw_novelty_k{k}`` and
``calibrated_novelty_k{k}``: the two scales share a dtype and a range, so the name is the only
thing that says which one a number is on, and `theta` means something only on the calibrated
one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Final

import polars as pl

from honey.novelty_score_exp.core.functions import DEFAULT_F, FAMILY

#: Repo-root ``data/`` (gitignored). Four parents up: engine -> novelty_score_exp -> honey -> src.
_DATA: Final = Path(__file__).resolve().parents[4] / "data"

#: Hard lower bound on usable article data — the dump starts here, so every panel is cut to it.
ARTICLE_DATA_START: Final = date(2024, 7, 1)

#: Upper bound, well past the end of the data; panels are cut to it so a stray future
#: timestamp cannot stretch an axis.
ARTICLE_DATA_END: Final = date(2030, 1, 1)

#: The embedding store and the per-company article dump.
EMB: Final = str(_DATA / "es_title_embeddings")
DUMP: Final = str(_DATA / "es_metadata")

#: Parquets the engine derives from the feed (`mentions`, `articles`, `sources`); see
#: :mod:`honey.novelty_score_exp.engine.derive`.
DERIVED: Final = str(_DATA / "derived")

#: Parquet key under which a run's :class:`KNNParams` are stored as JSON.
PARAMS_KEY: Final = "novelty_params"


@dataclass(frozen=True)
class KNNParams:
    """The knobs: candidate function, lookback window, neighbour counts, cutoff, candidate cap.

    ``k_values`` holds every neighbour count to emit — one ``raw_novelty_k{k}`` and one
    ``calibrated_novelty_k{k}`` column each. They
    all come out of a single pass: the cost is the top-``max(k)`` selection, which
    ``np.partition`` does in time independent of the cut point, so a wide grid of ``k`` is
    free (measured flat from k=5 to k=256).
    """

    window_s: float = 24 * 7 * 3600  # 7-day causal window; the paper's w (§2.4)
    k_values: tuple[int, ...] = (5,)  # neighbour counts to emit, one column pair per value
    theta: float = 0.30  # novelty cutoff for the filter route (§3.1)

    #: Cap on the candidate set: at most this many most-recent earlier articles, whatever the
    #: window holds. **Purely to make the computation feasible — it is not part of the score.**
    #: The paper's R_c(d;w) (§2.4) is bounded by time alone, so whenever this binds the code is
    #: scoring against a shorter lookback than `window_s` claims, and it binds *unevenly*: a
    #: thinly covered name never reaches the cap while a heavily covered one hits it in hours,
    #: so `w` stops denoting one duration across the cross-section.
    #:
    #: Measured on the corpus at a 7-day window, the candidate set an article faces runs 1,801
    #: at p25, 3,995 at the median and 77,417 at most — so a cap of 2,000 would bind for 72.3%
    #: of articles. Default 0 (uncapped) accordingly, matching the published runs; reach for a
    #: cap only when a run will not otherwise fit, and report it if you do.
    max_cand: int = 0

    #: The candidate function, by its name in :data:`core.functions.FAMILY` — which is also the
    #: key into the engine's registry, so a run and its tests name the same `f`.
    f: str = DEFAULT_F

    def __post_init__(self) -> None:
        """Reject a candidate `core/` does not define, naming the ones it does.

        Raises:
            ValueError: If ``f`` is not in :data:`core.functions.FAMILY`.
        """
        if self.f not in FAMILY:
            msg = f"unknown candidate f={self.f!r}; core.functions.FAMILY defines {sorted(FAMILY)}"
            raise ValueError(msg)

    @property
    def k_max(self) -> int:
        """Largest neighbour count — the depth the top-k selection has to reach."""
        return max(self.k_values)

    @property
    def capped(self) -> bool:
        """Whether a candidate cap binds at all.

        ``max_cand <= 0`` means *uncapped*, not unset: the lookback window alone bounds the
        candidate set, which is what the published runs used. A truthiness test on
        ``max_cand`` preserves that by accident and an ``is not None`` test breaks it, so the
        predicate is spelled out here and callers use it rather than re-deriving the rule.
        """
        return self.max_cand > 0

    @property
    def window_hours(self) -> float:
        """Lookback window in hours."""
        return self.window_s / 3600

    @property
    def window_days(self) -> float:
        """Lookback window in days — the nominal one; ``max_cand`` may bind first for busy companies."""
        return self.window_s / 86400

    @property
    def window_label(self) -> str:
        """Compact window tag for chart labels (``6h``, ``7d``, …).

        Not the filename: those carry hours whatever the label reads, so that a directory
        sorts by window length. See :func:`scores_path`.
        """
        return f"{self.window_hours:g}h" if self.window_hours < 48 else f"{self.window_days:g}d"

    def describe(self) -> str:
        """One-line human label for charts and dashboards."""
        cap = f"max_cand={self.max_cand:,}" if self.capped else "uncapped"
        return f"{self.f}, window {self.window_label}, k={list(self.k_values)}, {cap}"


def raw_novelty_col(k: int) -> str:
    """Column holding raw novelty, `nu`, at neighbour count *k*.

    Args:
        k: Neighbour count.

    Returns:
        The column name.
    """
    return f"raw_novelty_k{k}"


def calibrated_novelty_col(k: int) -> str:
    """Column holding calibrated novelty, `nu_tilde`, at neighbour count *k*.

    A sibling of :func:`raw_novelty_col` rather than a replacement: §sec:calib's `nu_tilde` is a
    rank over the raw column, so both live in the same file and a route picks the scale it
    wants. Carrying raw alongside calibrated is also what lets the invariance property be
    checked on real data rather than only on fixtures.

    Args:
        k: Neighbour count.

    Returns:
        The column name.
    """
    return f"calibrated_novelty_k{k}"


def scores_path(emb_dir: str | Path, window_hours: float, f: str = DEFAULT_F) -> Path:
    """Where a scoring run for this candidate and window lands.

    `f` is in the name, so two candidates scored at one window are two files rather than one
    overwriting the other.

    Args:
        emb_dir: Directory holding the scores parquets.
        window_hours: Lookback window in hours.
        f: The candidate function's name.

    Returns:
        The parquet path.
    """
    return Path(emb_dir) / f"novelty_scores_{f}_w{window_hours:g}h.parquet"


def params_metadata(p: KNNParams) -> dict[str, str]:
    """`KNNParams` as parquet key-value metadata (JSON under :data:`PARAMS_KEY`).

    The derived window fields are written alongside the stored ones so a reader can label a
    file without reconstructing the dataclass.

    Args:
        p: The run's parameters.

    Returns:
        Single-entry metadata mapping.
    """
    extra = {"window_days": p.window_days, "window_hours": p.window_hours, "window_label": p.window_label}
    return {PARAMS_KEY: json.dumps(asdict(p) | extra)}


def read_params(path: str | Path) -> KNNParams | None:
    """Recover the `KNNParams` a scores parquet was written with.

    Unknown keys are dropped rather than raising, so a file written by a later version that
    added a field is still readable.

    Args:
        path: Parquet path.

    Returns:
        The parameters, or None for a file predating the metadata.
    """
    raw = pl.read_parquet_metadata(str(path)).get(PARAMS_KEY)
    if not raw:
        return None
    d = json.loads(raw)
    d = {key: val for key, val in d.items() if key in KNNParams.__dataclass_fields__}
    if "k_values" in d:
        d["k_values"] = tuple(d["k_values"])
    return KNNParams(**d)
