"""The snippets of a universe of assets, read from the store by the assets themselves.

The store holds one parquet file per company and month under ``snippets/<entity>/<YYYY-MM>``,
and a YUKKA universe names its companies by ``yukka_id``. The two line up exactly — the
directory is the id without its ``company:`` prefix — so a universe is a file list, and this
module is that translation plus the read::

    from yukka.data import Index
    from novetly_score.data import data

    snippets = data(Index.STOXX600.assets).snippets
    frame = snippets.months("2025-01", "2025-03").frame

Nothing above the ``.frame`` reaches the network. ``data(...)`` resolves ids, ``.snippets``
describes a query, and each narrowing returns another description of one; the read happens when
a frame is asked for, and only for the files the narrowing left.

**Narrowing is not optional in practice.** All 627 companies across all 29 months is 181 million
rows, and the ``snippet`` column is the text itself. ``months()`` drops whole files before any
of them is opened, and ``select()`` drops columns before they are fetched; both are pushed down
into the parquet reads rather than applied afterwards.

**Which files exist is read, not guessed.** ``snippets_manifest/file_stats.parquet`` lists every
(entity, month) pair with its row count, so the file list comes from a 190 KB read instead of a
walk of 18,000 keys, and the 309 pairs that exist but hold no rows are skipped rather than
opened. It also means :attr:`Snippets.rows` can say how much a read will return before it runs.

**A universe is wider than the store.** Of the STOXX 600's 964 companies, 627 have snippets;
the rest are mostly constituents that left the index before the snippet history starts. They
are reported by :attr:`Snippets.missing` rather than dropped in silence, because a frame that is
quietly short of a third of its universe is a result that looks fine and is not.
"""

from __future__ import annotations

import re
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final

from novetly_score.data.duck import BUCKET, OpenDuck

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl
    from duckdb import DuckDBPyConnection, DuckDBPyRelation
    from yukka import Asset  # type-check-only: the layer reads a yukka_id, it does not import yukka
    from yukka.data import IndexConstituent

    #: Anything that names one company: what a YUKKA universe hands out, or the id itself.
    AssetLike = Asset | IndexConstituent | str

    #: Anything that names a set of companies. A frame is :attr:`yukka.data.Index.frame`, taken
    #: for its ``yukka_id`` column, so a universe can be passed exactly as it comes.
    Assets = AssetLike | Iterable[AssetLike] | pl.DataFrame

#: Prefix holding one directory per company.
SNIPPETS_PREFIX: Final = "snippets"

#: Manifest of every snippet file, with its row and article counts.
MANIFEST: Final = f"s3://{BUCKET}/snippets_manifest/file_stats.parquet"

#: Prefix a YUKKA company id carries and a snippet directory does not.
COMPANY_PREFIX: Final = "company:"

#: The shape of a month, as the file names spell it.
MONTH: Final = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

#: Columns of a snippet file, in file order. ``entity`` holds the full ``yukka_id``, so a frame
#: joins straight back onto a YUKKA universe; ``snippet`` is the text and dominates the bytes.
COLUMNS: Final = (
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


class NoSnippetsError(LookupError):
    """The narrowing left no file to read, so a frame would be empty for a reason worth naming."""


def entity(asset: AssetLike) -> str:
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


def entities(assets: Assets) -> tuple[str, ...]:
    """Name the snippet directory of every asset, once each and in the order given.

    Args:
        assets: One asset, an iterable of them, or a frame with a ``yukka_id`` column — which is
            what :attr:`yukka.data.Index.frame` is, so a universe can be passed as it comes.

    Returns:
        Directory names, deduplicated. A universe repeats a company once per membership
        interval, and repeating a file in a read would double its rows.
    """
    if hasattr(assets, "columns") and hasattr(assets, "get_column"):  # a polars frame
        assets = assets.get_column("yukka_id").drop_nulls().to_list()
    items: Iterable[Any] = assets if _is_iterable(assets) else [assets]
    return tuple(dict.fromkeys(entity(item) for item in items))


def _is_iterable(assets: Assets) -> bool:
    """Say whether *assets* is a collection of assets rather than a single one.

    Args:
        assets: The argument to classify.

    Returns:
        ``True`` for an iterable that is not itself a single string asset.
    """
    return not isinstance(assets, str) and hasattr(assets, "__iter__")


class Snippets:
    """A snippet read waiting to be run: which companies, which months, which columns.

        >>> data("company:bmw").snippets.months("2025-01").frame        # doctest: +SKIP

    Every method returns a new instance, so a narrowing is a value and not a mutation, and the
    same :class:`Snippets` can be narrowed two ways without either affecting the other.
    """

    def __init__(
        self,
        names: tuple[str, ...],
        *,
        first: str | None = None,
        last: str | None = None,
        columns: tuple[str, ...] = COLUMNS,
        connection: DuckDBPyConnection | None = None,
    ) -> None:
        """Describe a read. Reaches no network.

        Args:
            names: Snippet directory names, as :func:`entities` returns them.
            first: Earliest month to read, ``"YYYY-MM"``, or ``None`` for the store's earliest.
            last: Latest month to read, ``"YYYY-MM"``, or ``None`` for the store's latest.
            columns: Columns to fetch, a subset of :data:`COLUMNS`.
            connection: An open connection to read on, or ``None`` to open one per read.

        Raises:
            ValueError: If a column is not one of :data:`COLUMNS`. Checked here and not only in
                :meth:`select`, so a column name can only ever reach a query as a known literal.
        """
        self._names = names
        self._first = first
        self._last = last
        self._columns = _checked(columns)
        self._connection = connection
        self._files: pl.DataFrame | None = None

    def months(self, first: str | None = None, last: str | None = None) -> Snippets:
        """Narrow to a span of months, both ends included.

        Whole files are dropped before any is opened, which is the one narrowing that decides
        whether a read is seconds or minutes.

        Args:
            first: Earliest month to keep, ``"YYYY-MM"``. ``None`` leaves the start open.
            last: Latest month to keep, ``"YYYY-MM"``. Left ``None`` it repeats *first*, so
                ``months("2025-01")`` is that one month.

        Returns:
            A new :class:`Snippets` over the narrowed span.

        Raises:
            ValueError: If a month is not ``"YYYY-MM"``, or the span runs backwards.
        """
        first = _month(first, "first")
        last = _month(last, "last") if last is not None else first
        if first is not None and last is not None and last < first:
            msg = f"months({first!r}, {last!r}) runs backwards"
            raise ValueError(msg)
        return self._with(first=first, last=last)

    def select(self, *columns: str) -> Snippets:
        """Narrow to the columns named, in the order named.

        Args:
            *columns: Column names, a subset of :data:`COLUMNS`. Dropping ``snippet`` leaves a
                read of counts and timestamps that moves a fraction of the bytes.

        Returns:
            A new :class:`Snippets` fetching only those columns.

        Raises:
            ValueError: If no column is named, or a name is not one of :data:`COLUMNS`.
        """
        if not columns:
            msg = f"select() needs at least one of: {', '.join(COLUMNS)}"
            raise ValueError(msg)
        return self._with(columns=_checked(columns))

    @property
    def files(self) -> pl.DataFrame:
        """The files this read would open, from the manifest rather than a listing.

        Returns:
            One row per file: ``entity``, ``month``, ``rows``, ``articles``, ``bytes``, ``path``.
            Pairs that exist but hold no rows are left out; so are companies the store has no
            directory for, which :attr:`missing` names instead.
        """
        if self._files is None:
            self._files = self._read_manifest()
        return self._files

    @property
    def coverage(self) -> pl.DataFrame:
        """What each company brings to this read, deepest first.

        Cheap enough to look at before every read: it comes from the same manifest, so it costs
        one small object and no snippet file is opened.

        Returns:
            One row per company: ``entity`` (the ``yukka_id``), ``months``, ``rows``,
            ``articles``, ``bytes``, ``first_month``, ``last_month``.
        """
        import polars as pl

        return (
            self.files.group_by("entity")
            .agg(
                pl.len().alias("months"),
                pl.col("rows").sum(),
                pl.col("articles").sum(),
                pl.col("bytes").sum(),
                pl.col("month").min().alias("first_month"),
                pl.col("month").max().alias("last_month"),
            )
            .with_columns(entity=COMPANY_PREFIX + pl.col("entity"))
            .sort("rows", descending=True)
        )

    @property
    def rows(self) -> int:
        """How many rows the read would return, known before it runs.

        Returns:
            The manifest's row count over the selected files.
        """
        return int(self.files.get_column("rows").sum())

    @property
    def missing(self) -> tuple[str, ...]:
        """The companies asked for that this read has no file for.

        A universe is wider than the store: a constituent that left the index before the
        snippet history starts has no directory at all. Reported, not dropped in silence.

        Returns:
            Their ``yukka_id``s, in the order they were asked for.
        """
        found = set(self.files.get_column("entity").to_list())
        return tuple(COMPANY_PREFIX + name for name in self._names if name not in found)

    @property
    def paths(self) -> tuple[str, ...]:
        """The ``s3://`` paths this read would open, in entity then month order.

        Returns:
            One path per file.
        """
        return tuple(self.files.get_column("path").to_list())

    @property
    def frame(self) -> pl.DataFrame:
        """Run the read and return its rows.

        The rows cross from DuckDB to polars through Arrow without a copy, so the frame costs
        the download and little else. It is held in memory: check :attr:`rows` first on a span
        wide enough to matter.

        Returns:
            A polars DataFrame of the selected columns, ordered by file and then by
            ``publish_time`` within each, which is how the files are written.

        Raises:
            NoSnippetsError: If the narrowing left no file to read.
        """
        import polars as pl

        with self._open() as connection:
            return pl.DataFrame(self.relation(connection))

    def relation(self, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation:
        """The read as a DuckDB relation, for a query that goes further than a frame.

        The escape hatch: a relation composes into SQL, so an aggregation, a window or a join
        against the score files runs inside DuckDB and only its result is materialised.

            >>> with data("company:bmw").snippets.months("2025-01") as held:   # doctest: +SKIP
            ...     rel = held.relation()
            ...     rel.aggregate("entity, lang, count(*)", "entity, lang").pl()

        A relation lives only as long as the connection under it, which is why there is no
        connection to default to outside a ``with`` block: one opened here would close before
        the relation could be read.

        Materialise it with ``pl.DataFrame(relation)``, as :attr:`frame` does. ``fetchall()``
        builds Python objects instead, and for the ``publish_time`` column that means DuckDB
        needs ``pytz`` installed; the Arrow path does not.

        Args:
            connection: An open connection, from :class:`~novetly_score.data.duck.OpenDuck`.
                Left ``None`` it is the connection this :class:`Snippets` holds.

        Returns:
            A relation over the selected files and columns. Nothing is read until it is.

        Raises:
            NoSnippetsError: If the narrowing left no file to read.
            RuntimeError: If no connection was passed and none is held.
        """
        if connection is None:
            connection = self._connection
        if connection is None:
            msg = "relation() needs a connection: pass one from OpenDuck, or ask inside a `with ...snippets as` block"
            raise RuntimeError(msg)
        paths = self.paths
        if not paths:
            raise NoSnippetsError(self._why_empty())
        files = connection.sql("SELECT * FROM read_parquet($paths)", params={"paths": list(paths)})
        if self._columns == COLUMNS:
            return files
        return files.project(", ".join(self._columns))  # projected on the plan, so pushed into the read

    def __enter__(self) -> Snippets:
        """Hold one connection open for every read in the block, instead of one per read.

        Returns:
            A :class:`Snippets` bound to that connection.
        """
        self._held = OpenDuck()
        return self._with(connection=self._held.__enter__())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the held connection, however the block ended.

        Args:
            exc_type: Exception class, if the block raised.
            exc: The exception, if the block raised.
            tb: Its traceback, if the block raised.
        """
        self._held.__exit__(exc_type, exc, tb)

    def __repr__(self) -> str:
        """Describe the read without running it.

        Returns:
            A one-line summary of companies, span and columns.
        """
        span = f"{self._first or '…'}:{self._last or '…'}"
        return f"Snippets({len(self._names)} companies, months {span}, {len(self._columns)} columns)"

    def _with(self, **changed: Any) -> Snippets:
        """Copy this read with some of its narrowings replaced.

        Args:
            **changed: Constructor keywords to override.

        Returns:
            A new :class:`Snippets`; the manifest already read is not carried over, because a
            different narrowing selects different files.
        """
        settings: dict[str, Any] = {
            "first": self._first,
            "last": self._last,
            "columns": self._columns,
            "connection": self._connection,
        }
        return Snippets(self._names, **(settings | changed))

    def _open(self) -> Any:
        """Give a ``with`` block the connection to read on, opening one only if none is held.

        Returns:
            A context manager yielding a connection: the held one, left open on exit, or a
            fresh one closed on exit.
        """
        if self._connection is not None:
            return _Borrowed(self._connection)
        return OpenDuck()

    def _read_manifest(self) -> pl.DataFrame:
        """Read the manifest and cut it down to the files this read wants.

        Returns:
            The file list, ordered by entity then month, with its ``s3://`` path.
        """
        import polars as pl

        predicate = pl.col("entity").is_in(list(self._names)) & (pl.col("rows") > 0)
        if self._first is not None:
            predicate &= pl.col("month") >= self._first
        if self._last is not None:
            predicate &= pl.col("month") <= self._last
        with self._open() as connection:
            manifest = pl.DataFrame(connection.sql("SELECT * FROM read_parquet($path)", params={"path": MANIFEST}))
        return (
            manifest.filter(predicate)
            .sort("entity", "month")
            .select(
                "entity",
                "month",
                "rows",
                "articles",
                "bytes",
                path=pl.format(f"s3://{BUCKET}/{SNIPPETS_PREFIX}/{{}}/{{}}.parquet", "entity", "month"),
            )
        )

    def _why_empty(self) -> str:
        """Explain an empty file list, distinguishing a typo from a company with no snippets.

        Returns:
            A message naming the likely cause: no company recognised, or a span outside the
            store's months.
        """
        import polars as pl

        with self._open() as connection:
            months = pl.DataFrame(
                connection.sql(
                    "SELECT min(month) AS first, max(month) AS last FROM read_parquet($path)", params={"path": MANIFEST}
                )
            )
        first, last = months.row(0)
        span = f"months({self._first or '…'}, {self._last or '…'})"
        held = f"the store holds {first} to {last}"
        unbounded = Snippets(self._names, columns=self._columns, connection=self._connection)
        if len(unbounded.missing) == len(self._names):
            return f"the store has no snippets at all for the {len(self._names)} companies asked for; {held}"
        return f"{span} selects no file for any of the {len(self._names)} companies asked for; {held}"


class _Borrowed:
    """A context manager over a connection it did not open, and so must not close."""

    def __init__(self, connection: DuckDBPyConnection) -> None:
        """Hold the borrowed connection.

        Args:
            connection: The connection to hand out.
        """
        self._connection = connection

    def __enter__(self) -> DuckDBPyConnection:
        """Hand out the borrowed connection.

        Returns:
            The connection, which stays open after the block.
        """
        return self._connection

    def __exit__(self, *_exc: object) -> None:
        """Leave the connection open; its owner closes it.

        Args:
            *_exc: Exception details, ignored.
        """


class Universe:
    """A set of assets, and the data the store holds for them.

        >>> from yukka.data import Index                     # doctest: +SKIP
        >>> data(Index.STOXX600.assets).snippets             # doctest: +SKIP
        Snippets(964 companies, months …:…, 11 columns)

    One attribute per dataset, so a second one — the embeddings, the novelty scores — is added
    here rather than by a new entry point.
    """

    def __init__(self, assets: Assets, *, connection: DuckDBPyConnection | None = None) -> None:
        """Resolve assets to the names the store files them under. Reaches no network.

        Args:
            assets: One asset, an iterable of them, or a frame with a ``yukka_id`` column.
            connection: An open connection for the reads to use, or ``None`` to open one per read.

        Raises:
            ValueError: If no asset was given.
        """
        self._names = entities(assets)
        self._connection = connection
        if not self._names:
            msg = "no assets given: pass a YUKKA id, an Asset, an IndexConstituent, or a list of them"
            raise ValueError(msg)

    @property
    def snippets(self) -> Snippets:
        """The sentiment snippets for these assets, as a read not yet run.

        Returns:
            A :class:`Snippets` over every month the store has, narrowable by
            :meth:`Snippets.months` and :meth:`Snippets.select`.
        """
        return Snippets(self._names, connection=self._connection)

    @property
    def names(self) -> tuple[str, ...]:
        """The snippet directory names these assets resolved to.

        Returns:
            One name per asset, deduplicated, in the order given.
        """
        return self._names

    def __len__(self) -> int:
        """Count the distinct assets.

        Returns:
            How many companies this universe names.
        """
        return len(self._names)

    def __repr__(self) -> str:
        """Describe the universe.

        Returns:
            A one-line summary.
        """
        return f"Universe({len(self._names)} companies)"


def data(assets: Assets, *, connection: DuckDBPyConnection | None = None) -> Universe:
    """Open a view of the store's data for a set of assets. Reaches no network.

        >>> from yukka.data import Index                           # doctest: +SKIP
        >>> frame = data(Index.STOXX600.assets).snippets.months("2025-01").frame   # doctest: +SKIP

    Args:
        assets: One asset, an iterable of them, or a frame with a ``yukka_id`` column.
            :attr:`yukka.data.Index.assets` and :attr:`yukka.data.Index.frame` both work as they
            come, as do YUKKA ids and plain directory names.
        connection: An open connection for the reads to use, or ``None`` to open one per read.
            Passing one from :class:`~novetly_score.data.duck.OpenDuck` is worth it in a loop.

    Returns:
        A :class:`Universe` whose attributes are the datasets held for those assets.

    Raises:
        ValueError: If no asset was given.
    """
    return Universe(assets, connection=connection)


def _checked(columns: tuple[str, ...]) -> tuple[str, ...]:
    """Check column names before one of them reaches a query.

    Args:
        columns: The names to check.

    Returns:
        The names, as a tuple.

    Raises:
        ValueError: If any is not one of :data:`COLUMNS`.
    """
    unknown = [column for column in columns if column not in COLUMNS]
    if unknown:
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
