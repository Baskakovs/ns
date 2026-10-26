"""The DuckDB query layer over the parquet store — projections and key joins, nothing else.

Storage is parquet and nothing is copied into a ``.duckdb`` file. This module applies the
``sql/`` view definitions to an in-memory connection, so SQL is the readable interface onto it.

    >>> from honey.novelty_score_exp.engine import db
    >>> con, applied = db.connect()                           # doctest: +SKIP
    >>> con.sql("SELECT count(*) FROM mentions").fetchone()   # doctest: +SKIP

**The views hold no rules.** Every one is a column projection over a parquet the engine wrote,
plus `mentions_scored`'s key join. The tone, the dedups, the clip, the calibration and the
panels are computed in `engine/` — each pinned to `core/` — so the definition has one
implementation per layer and SQL is not a third.

**And it exposes nothing the running code did not write.** Before a view is created over a
derived table or a scores file, the file's stamp is checked against the producing module's
current fingerprint; a mismatch raises :class:`honey.novelty_score_exp.twins.StaleArtifactError`
whatever ``strict`` says, because a stale file is not a missing one — it is a wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple

if TYPE_CHECKING:
    import duckdb

from honey.novelty_score_exp.core.functions import DEFAULT_F
from honey.novelty_score_exp.engine import derive, io, params
from honey.novelty_score_exp.engine.params import _DATA
from honey.novelty_score_exp.twins import check_stamp

#: Where the view definitions live, applied in filename order — the numbering is dependency order.
SQL_DIR: Final = Path(__file__).resolve().parent.parent / "sql"

#: Default grid cell. ``window`` is in hours, as the scores filenames carry it.
DEFAULTS: Final[dict[str, str]] = {"f": DEFAULT_F, "window": "6", "k": "5"}


@dataclass(frozen=True)
class Applied:
    """What :func:`connect` managed to set up."""

    views: tuple[str, ...]  # views created
    skipped: tuple[tuple[str, str], ...]  # (file, reason) for definitions whose parquet is absent

    def __str__(self) -> str:  # noqa: D105
        out = f"{len(self.views)} views: {', '.join(self.views)}"
        if self.skipped:
            out += "\nskipped:\n" + "\n".join(f"  {f}: {why}" for f, why in self.skipped)
        return out


class Connected(NamedTuple):
    """A connection and the summary of what was applied to it.

    Returned as a pair rather than stashed on the connection: ``DuckDBPyConnection`` is a C
    extension type with no ``__dict__``, so an attribute cannot be attached to it.
    """

    con: duckdb.DuckDBPyConnection
    applied: Applied


def _substitutions(data: Path, cell: dict[str, str]) -> dict[str, str]:
    """``$``-placeholders for one grid cell, after checking every file they point at.

    Args:
        data: Parquet root.
        cell: ``f``, ``window`` (hours) and ``k``.

    Returns:
        The placeholder values.

    Raises:
        StaleArtifactError: If a derived table or the scores file was written by other code.
        KeyError: If the scores file has no columns for ``k``.
    """
    derived = data / "derived"
    derive.check(derived)
    scores = params.scores_path(data / "es_title_embeddings", float(cell["window"]), cell["f"])
    raw, calibrated = params.raw_novelty_col(int(cell["k"])), params.calibrated_novelty_col(int(cell["k"]))
    if scores.exists():
        check_stamp(scores, io.PRODUCER)
        raw, calibrated = io.columns_for(scores, int(cell["k"]))
    return {
        "data": str(data),
        "derived": str(derived),
        "scores": str(scores),
        "raw_col": raw,
        "calibrated_col": calibrated,
    }


def _render(sql: str, subs: dict[str, str]) -> str:
    """Substitute the ``$``-placeholders into one definition.

    Done by literal replacement rather than DuckDB's prepared parameters, because a parquet
    path inside ``read_parquet`` is part of the query plan and cannot be bound. Longest key
    first, so ``$data`` cannot eat the front of a longer placeholder.

    Args:
        sql: The file's contents.
        subs: Placeholder values, from :func:`_substitutions`.

    Returns:
        The rendered SQL.
    """
    for key in sorted(subs, key=len, reverse=True):
        sql = sql.replace(f"${key}", subs[key])
    return sql


def _statements(sql: str) -> list[str]:
    """Split a rendered definition into individual statements.

    Needed because a file holds several views and DuckDB aborts the whole batch at the first
    failure, so one missing parquet would otherwise cost every view in its file.

    Comments are stripped *before* splitting, which is the whole subtlety: these files carry
    prose containing semicolons (``-- FLOAT[256]; see params.DIMS``), and splitting first
    orphans the tail of such a comment into the next chunk, where it parses as SQL.

    Line comments only, and no string-literal awareness — the definitions' only literals are
    parquet paths, which contain no ``--``.

    Args:
        sql: Rendered SQL, possibly several statements.

    Returns:
        The statements, comments and blank padding stripped, in order.
    """
    stripped = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
    return [chunk.strip() for chunk in stripped.split(";") if chunk.strip()]


def definitions() -> list[Path]:
    """The view-definition files, in dependency order.

    Returns:
        Sorted list of ``.sql`` paths.
    """
    return sorted(SQL_DIR.glob("*.sql"))


def connect(data: str | Path = _DATA, *, strict: bool = False, **cell: str) -> Connected:
    """Open a connection with the ``sql/`` views applied for one grid cell.

    A definition whose backing parquet does not exist is skipped rather than fatal, so the
    layer is usable while the store is still being populated — and the skips are reported on
    the returned ``applied`` rather than swallowed. A parquet that exists but was written by
    other code is **not** skipped: it raises.

    Args:
        data: Parquet root, normally the repo's ``data/``.
        strict: Raise on the first definition that fails instead of skipping it.
        **cell: Over :data:`DEFAULTS` — ``f``, ``window`` (hours) and ``k``.

    Returns:
        The connection paired with the :class:`Applied` summary.

    Raises:
        FileNotFoundError: If the SQL directory holds no definitions.
        StaleArtifactError: If a derived table or the scores file is stale.
    """
    import duckdb

    files = definitions()
    if not files:
        msg = f"no .sql definitions in {SQL_DIR}"
        raise FileNotFoundError(msg)

    subs = _substitutions(Path(data), DEFAULTS | cell)
    con = duckdb.connect(":memory:")
    views: list[str] = []
    skipped: list[tuple[str, str]] = []
    for path in files:
        for statement in _statements(_render(path.read_text(), subs)):
            try:
                con.execute(statement)
            except duckdb.Error as exc:  # a missing parquet, almost always
                if strict:
                    raise
                skipped.append((path.name, str(exc).strip().splitlines()[0]))
                continue
            views.extend(_view_names(statement))
    return Connected(con, Applied(tuple(views), tuple(skipped)))


def _view_names(sql: str) -> list[str]:
    """View names a rendered definition creates, in order of appearance.

    Args:
        sql: Rendered SQL.

    Returns:
        The view names.
    """
    out = []
    for line in sql.splitlines():
        upper = line.upper().strip()
        if upper.startswith("CREATE OR REPLACE VIEW "):
            out.append(line.strip().split()[4])
    return out


def describe(con: duckdb.DuckDBPyConnection) -> list[tuple[str, Any]]:
    """Row counts for the model views, for a quick sanity read of a fresh connection.

    Args:
        con: A connection from :func:`connect`.

    Returns:
        ``[(view, rows)]``, with the count replaced by the error text where a view fails.
    """
    wanted = ("titles", "article_titles", "sources", "constituents", "mentions", "scores")
    out: list[tuple[str, Any]] = []
    have = {
        row[0]
        for row in con.sql(
            "SELECT view_name FROM duckdb_views() WHERE schema_name = 'main' AND NOT internal"
        ).fetchall()
    }
    for name in wanted:
        if name not in have:
            continue
        try:
            out.append((name, int(con.table(name).aggregate("count(*)").fetchone()[0])))
        except Exception as exc:  # noqa: BLE001 — a view over a missing file reports, not raises
            out.append((name, str(exc).strip().splitlines()[0]))
    return out
