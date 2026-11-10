"""Read the snippets of a set of assets, by the assets of a YUKKA universe.

The store holds one parquet file per company and month under ``snippets/<entity>/<YYYY-MM>``,
and a YUKKA universe names its companies by ``yukka_id``. The two line up exactly — the
directory is the id without its ``company:`` prefix — so a universe is a file list::

    from yukka.data import Index
    from novetly_score.data import snippets

    frame = snippets("company:bmw")                       # bmw's whole history
    frame = snippets(Index.STOXX600.assets, first="2025-01", last="2025-03")

The file list comes from ``snippets_manifest/file_stats.parquet`` rather than a walk of 18,000
keys, which also means the files holding no rows are never opened.

**A universe is wider than the store.** Of the STOXX 600's 964 companies, 627 have snippets;
the rest mostly left the index before the snippet history starts. Those are warned about rather
than dropped in silence, because a frame quietly short of a third of its universe looks fine.

**The whole corpus does not fit in memory.** All 627 companies across all 29 months is 181
million rows, and the ``snippet`` column is the text itself. One company is comfortable;
a universe needs ``first``/``last`` to bound the months, or ``columns`` to drop the text, or
both. Both are pushed into the parquet read rather than applied after it.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, Final

from novetly_score.data.duck import BUCKET, OpenDuck

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import polars as pl

#: Prefix holding one directory per company.
SNIPPETS_PREFIX: Final = "snippets"

#: Manifest of every snippet file, with its row counts.
MANIFEST: Final = f"s3://{BUCKET}/snippets_manifest/file_stats.parquet"

#: Prefix a YUKKA company id carries and a snippet directory does not.
COMPANY_PREFIX: Final = "company:"

#: The shape of a month, as the file names spell it.
MONTH: Final = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

#: Columns of a snippet file, in file order. ``snippet_id`` is the row's own key, a native UUID
#: unique across all 181,276,415 rows of the corpus; ``entity`` holds the full ``yukka_id``, so
#: a frame joins straight back onto a YUKKA universe; ``snippet`` is the text and dominates the
#: bytes.
COLUMNS: Final = (
    "snippet_id",
    "entity",
    "sentiment",
    "snippet",
    "article_id",
    "publish_time",
    "news_type",
    "provider",
    "source",
    "read_more_link",
    "implied",
    "lang",
)

_SQL_FILES: Final = """
    SELECT entity, month FROM read_parquet($manifest)
    WHERE rows > 0 AND list_contains($entities, entity)
      AND ($first IS NULL OR month >= $first) AND ($last IS NULL OR month <= $last)
    ORDER BY entity, month
"""

_SQL_MONTHS: Final = "SELECT min(month), max(month) FROM read_parquet($manifest)"


class NoSnippetsError(LookupError):
    """The selection left no file to read, so a frame would be empty for a reason worth naming."""


def entity(asset: Any) -> str:
    """Name the snippet directory of one asset.

    Accepts whatever a YUKKA universe hands out — :class:`yukka.data.IndexConstituent` and
    :class:`yukka.Asset` both carry a ``yukka_id`` — as well as a bare id or directory name, so
    a caller who has only the string needs no import to use this module.

    Args:
        asset: An object with a ``yukka_id``, an id such as ``"company:bmw"``, or the directory
            name ``"bmw"``.

    Returns:
        The directory name, e.g. ``"bmw"``.

    Raises:
        TypeError: If *asset* is neither a string nor carries a ``yukka_id``.
        ValueError: If the id is blank, or names a kind the snippets have no directory for.
    """
    identifier = asset if isinstance(asset, str) else getattr(asset, "yukka_id", None)
    if not isinstance(identifier, str):
        msg = f"not an asset: {asset!r} is neither a YUKKA id nor an object with a yukka_id"
        raise TypeError(msg)
    identifier = identifier.strip()
    if not identifier:
        msg = "not an asset: a blank YUKKA id"
        raise ValueError(msg)
    kind, sep, name = identifier.partition(":")
    if sep and kind != COMPANY_PREFIX.rstrip(":"):
        msg = f"{identifier!r} is a {kind}, and the snippets hold one directory per company"
        raise ValueError(msg)
    return name if sep else kind


def entities(assets: Any) -> tuple[str, ...]:
    """Name the snippet directory of every asset, once each and in the order given.

    Args:
        assets: One asset, an iterable of them, or a frame with a ``yukka_id`` column — which is
            what :attr:`yukka.data.Index.frame` is, so a universe can be passed as it comes.

    Returns:
        Directory names, deduplicated. A universe repeats a company once per membership
        interval, and repeating a file in a read would double its rows.

    Raises:
        ValueError: If no asset was given.
    """
    if hasattr(assets, "columns") and hasattr(assets, "get_column"):  # a polars frame
        assets = assets.get_column("yukka_id").drop_nulls().to_list()
    items: Iterable[Any] = [assets] if isinstance(assets, str) or not hasattr(assets, "__iter__") else assets
    names = tuple(dict.fromkeys(entity(item) for item in items))
    if not names:
        msg = "no assets given: pass a YUKKA id, an Asset, an IndexConstituent, or a list of them"
        raise ValueError(msg)
    return names


def paths(
    assets: Any,
    first: str | None = None,
    last: str | None = None,
) -> tuple[str, ...]:
    """Name the snippet files a selection covers, from the manifest rather than a listing.

    Costs one 190 KB read. Worth calling before a wide one: the file count is the only cheap
    signal of how large the frame will be.

    Args:
        assets: One asset, an iterable of them, or a frame with a ``yukka_id`` column.
        first: Earliest month to read, ``"YYYY-MM"``, or ``None`` for the store's earliest.
        last: Latest month to read, ``"YYYY-MM"``, or ``None`` for the store's latest.

    Returns:
        ``s3://`` paths, in entity then month order. Files holding no rows are left out.

    Raises:
        ValueError: If a month is not ``"YYYY-MM"``, or the span runs backwards.
        NoSnippetsError: If the selection leaves no file to read.
    """
    names = entities(assets)
    first, last = _month(first, "first"), _month(last, "last")
    if first and last and last < first:
        msg = f"the span {first}:{last} runs backwards"
        raise ValueError(msg)
    with OpenDuck() as con:
        rows = con.sql(
            _SQL_FILES,
            params={"manifest": MANIFEST, "entities": list(names), "first": first, "last": last},
        ).fetchall()
        if not rows:
            raise NoSnippetsError(_why_empty(con, names, first, last))
    found = {name for name, _ in rows}
    if missing := [name for name in names if name not in found]:
        warnings.warn(
            f"no snippets for {len(missing)} of {len(names)} companies asked for: "
            + ", ".join(COMPANY_PREFIX + name for name in missing[:10])
            + (f", +{len(missing) - 10} more" if len(missing) > 10 else ""),
            stacklevel=3,
        )
    return tuple(f"s3://{BUCKET}/{SNIPPETS_PREFIX}/{name}/{month}.parquet" for name, month in rows)


def snippets(
    assets: Any,
    first: str | None = None,
    last: str | None = None,
    *,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Read the snippets of a set of assets, over the whole history unless bounded.

        >>> snippets("company:bmw")                                       # doctest: +SKIP
        >>> snippets(Index.STOXX600.assets, "2025-01", "2025-03")         # doctest: +SKIP

    The rows cross from DuckDB to polars through Arrow without a copy, and only the files and
    columns the selection names are fetched.

    Args:
        assets: One asset, an iterable of them, or a frame with a ``yukka_id`` column.
            :attr:`yukka.data.Index.assets` and :attr:`yukka.data.Index.frame` both work as
            they come, as do YUKKA ids and plain directory names.
        first: Earliest month to read, ``"YYYY-MM"``, or ``None`` for the store's earliest.
        last: Latest month to read, ``"YYYY-MM"``, or ``None`` for the store's latest.
        columns: Columns to fetch, a subset of :data:`COLUMNS`, or ``None`` for all of them.
            Dropping ``snippet`` leaves a read of counts and timestamps that moves a fraction
            of the bytes.

    Returns:
        A polars DataFrame, ordered by file and then as the files are written.

    Raises:
        ValueError: If a month or a column name is not one this store has.
        NoSnippetsError: If the selection leaves no file to read.
    """
    import polars as pl

    selected = paths(assets, first, last)
    wanted = _checked(columns) if columns is not None else COLUMNS
    with OpenDuck() as con:
        files = con.sql("SELECT * FROM read_parquet($paths)", params={"paths": list(selected)})
        return pl.DataFrame(files if wanted == COLUMNS else files.project(", ".join(wanted)))


def _checked(columns: Sequence[str]) -> tuple[str, ...]:
    """Check column names before one of them reaches a query.

    Args:
        columns: The names to check.

    Returns:
        The names, as a tuple.

    Raises:
        ValueError: If none was given, or any is not one of :data:`COLUMNS`.
    """
    if not columns:
        msg = f"columns= needs at least one of: {', '.join(COLUMNS)}"
        raise ValueError(msg)
    if unknown := [column for column in columns if column not in COLUMNS]:
        msg = f"no such snippet column: {', '.join(unknown)}. Known: {', '.join(COLUMNS)}"
        raise ValueError(msg)
    return tuple(columns)


def _month(value: str | None, which: str) -> str | None:
    """Check a month argument before it reaches a path.

    Args:
        value: The month, ``"YYYY-MM"``, or ``None`` for an open end.
        which: Which end it is, for the error message.

    Returns:
        The month, stripped, or ``None``.

    Raises:
        ValueError: If it is not ``"YYYY-MM"``.
    """
    if value is None:
        return None
    month = value.strip()
    if not MONTH.match(month):
        msg = f"{which} month {value!r} is not YYYY-MM, e.g. '2025-01'"
        raise ValueError(msg)
    return month


def _why_empty(con: Any, names: tuple[str, ...], first: str | None, last: str | None) -> str:
    """Explain an empty file list, distinguishing a typo from a span outside the store.

    Args:
        con: An open connection.
        names: The directory names asked for.
        first: Earliest month asked for, or ``None``.
        last: Latest month asked for, or ``None``.

    Returns:
        A message naming the likely cause and the span the store does hold.
    """
    held = con.sql(_SQL_MONTHS, params={"manifest": MANIFEST}).fetchall()[0]
    span = f"{first or '…'}:{last or '…'}"
    return (
        f"no snippets for {', '.join(names[:5])}{'…' if len(names) > 5 else ''} over {span}; "
        f"the store holds {held[0]} to {held[1]}"
    )
