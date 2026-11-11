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

**Counts are derived, not stored.** The paper's `P` and `N` — a company's positive and negative
*articles* on a day — are what the snippets reduce to, and the store holds no panel of them.
:meth:`Universe.counts` is that reduction, run inside DuckDB, so the text is never fetched and
only the panel crosses into Python::

    panel = data(Index.STOXX600.frame).counts("2025-01", "2025-06", key="ric")

That one runs, because its span leaves nothing to narrow. :attr:`Snippets.counts` is the same
panel as a value — narrowable, composable into SQL, and able to say what it would cost first.

**A universe is wider than the store.** Of the STOXX 600's 964 companies, 627 have snippets;
the rest are mostly constituents that left the index before the snippet history starts. They
are reported by :attr:`Snippets.missing` rather than dropped in silence, because a frame that is
quietly short of a third of its universe is a result that looks fine and is not.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
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

#: What a snippet's sentiment label contributes to its article's tone. A label not named here —
#: ``NULL`` among them — contributes nothing rather than 0: it leaves the article
#: *uncharacterised*, which is no article at all rather than a neutral one.
TONE: Final = {"pos": 1, "neu": 0, "neg": -1}

#: Columns a counts read needs. ``snippet`` is not among them, which is what makes a panel over
#: the whole universe a different size of read from a frame of it.
COUNT_COLUMNS: Final = ("entity", "article_id", "publish_time", "sentiment")

#: One snippet's contribution, as SQL. Built from :data:`TONE` so the mapping has a single
#: definition, and so no label reaches a query except from this module.
_TONE_CASE: Final = (
    "CASE sentiment " + " ".join(f"WHEN '{label}' THEN {score}" for label, score in TONE.items()) + " END"
)

#: The counts, as one aggregation over a relation named ``snippets``. There are two groupings and
#: the first is the one that matters: a row per ``(entity, article_id)`` is the paper's `tone(d)`,
#: so what the second counts is **articles**. Counting snippet rows instead would scale both
#: counts by however many times each article names the company.
_COUNTS_SQL: Final = f"""
    WITH pairs AS (
        SELECT entity,
               article_id,
               -- the UTC day, fixed here rather than left to whatever the connection's
               -- TimeZone setting happens to be
               (publish_time AT TIME ZONE 'UTC')::DATE AS day,
               sum({_TONE_CASE}) AS tone,
               count({_TONE_CASE}) AS scored
        FROM snippets
        GROUP BY 1, 2, 3
    )
    SELECT entity,
           day,
           count_if(scored > 0 AND tone > 0)::BIGINT AS positive,
           count_if(scored > 0 AND tone = 0)::BIGINT AS neutral,
           count_if(scored > 0 AND tone < 0)::BIGINT AS negative,
           count_if(scored = 0)::BIGINT AS uncharacterised
    FROM pairs
    GROUP BY 1, 2
    ORDER BY entity, day
"""  # noqa: S608  # nosec B608 — the one interpolation is _TONE_CASE, from TONE's own literals


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


def identifier_history(universe: pl.DataFrame, column: str) -> pl.DataFrame:
    """What each company has been filed under as *column*, and from when.

    The other half of :func:`entities`: that one resolves a universe to the names the store files
    companies under, this one to the names some *other* dataset files them under — a RIC for a
    price feed, an ISIN for a holdings file. Those names are **point in time**. A company changes
    line (TUI, ``TUIT.L`` in London until 2024, then ``TUI1n.DE`` in Frankfurt) or changes share
    class (Industrivärden, the A line until 2022-09 and the C line from 2022-12), and the
    universe frame carries one spell per line with its own ``internal_key``. Collapsing that to
    one value per company would label a day with an identifier the company did not have on it.

    **Concurrent spells are resolved, not duplicated.** A few companies have two live lines at
    once — Unilever carried ``ULVR.L`` and ``UNA.AS`` from the same day until the 2020
    unification — and two rows starting on one day would key one day's counts twice. The spell
    running longest wins, which is the line that survived.

    Args:
        universe: A frame with ``yukka_id`` and *column*, and ``start_date``/``end_date`` where it
            has spells — :attr:`yukka.data.Index.frame` as it comes. Without ``start_date`` there
            is no history to read, so *column* must then be unique per company.
        column: The identifier column to take, e.g. ``"ric"``.

    Returns:
        ``yukka_id``, *column*, ``start_date``, ``end_date`` — one row per spell, at most one per
        ``(company, start_date)``, earliest first. A company no spell gives a *column* for is
        absent: there is nothing to file its rows under. ``start_date`` and ``end_date`` are null
        when the frame had no spells at all.

    Raises:
        ValueError: If a column is missing from *universe*, or the frame has no ``start_date`` and
            a company carries two values of *column*, leaving nothing to pick by.
    """
    import polars as pl

    absent = [name for name in ("yukka_id", column) if name not in universe.columns]
    if absent:
        msg = f"universe frame has no {' or '.join(absent)} column; it has {', '.join(universe.columns)}"
        raise ValueError(msg)
    if "start_date" not in universe.columns:
        return _without_spells(universe, column)
    ends = pl.col("end_date") if "end_date" in universe.columns else pl.lit(None, dtype=pl.Date).alias("end_date")
    return (
        universe.select("yukka_id", column, "start_date", end_date=ends)
        .drop_nulls(column)
        .drop_nulls("start_date")
        .sort("end_date", nulls_last=True)
        .group_by("yukka_id", "start_date")
        .agg(pl.col(column).last(), pl.col("end_date").last())  # the spell running longest wins
        .select("yukka_id", column, "start_date", "end_date")
        .sort("yukka_id", "start_date")
    )


def _without_spells(universe: pl.DataFrame, column: str) -> pl.DataFrame:
    """The identifier map of a frame that carries no spell dates.

    Args:
        universe: A frame with ``yukka_id`` and *column* but no ``start_date``.
        column: The identifier column to take.

    Returns:
        ``yukka_id``, *column*, and null ``start_date``/``end_date``, one row per company.

    Raises:
        ValueError: If a company carries two values of *column*: without dates there is no most
            recent one, so resolving it would be a guess dressed as a fact.
    """
    import polars as pl

    collected = (
        universe.select("yukka_id", column)
        .drop_nulls(column)
        .unique()
        .group_by("yukka_id")
        .agg(pl.col(column).unique())
    )
    ambiguous = collected.filter(pl.col(column).list.len() > 1)
    if ambiguous.height:
        named = ", ".join(ambiguous.get_column("yukka_id").to_list()[:3])
        msg = f"{ambiguous.height} companies carry two {column}s ({named}) and the frame has no start_date to pick by"
        raise ValueError(msg)
    return collected.with_columns(
        pl.col(column).list.first(),
        start_date=pl.lit(None, dtype=pl.Date),
        end_date=pl.lit(None, dtype=pl.Date),
    ).sort("yukka_id")


def as_of(panel: pl.DataFrame, universe: pl.DataFrame, column: str, *, on: str = "day") -> pl.DataFrame:
    """Relabel a dated panel with the identifier each company held on each of its days.

    An as-of join, not a lookup: a row takes the spell in force on its own day, so a company that
    changed line mid-sample carries the old identifier before the change and the new one after.

    **A day no spell covers keeps the last identifier in force.** Those days exist — a company
    leaves the index and comes back, and the store has its snippets throughout — and dropping
    them would be masking the universe, which is a decision for whoever computes a cross-section
    and not for a relabelling. Every row that comes in goes out.

    Args:
        panel: A frame with ``entity`` (a ``yukka_id``) and a date column.
        universe: The universe frame to read the history off.
        column: The identifier column to relabel with, e.g. ``"ric"``.
        on: Name of *panel*'s date column.

    Returns:
        *panel* with *column* in place of ``entity``, in the same column order. A company the
        universe gives no *column* for at all is absent, there being nothing to key its rows by.

    Raises:
        ValueError: If a column is missing, or an undated frame is ambiguous — see
            :func:`identifier_history`.
    """
    import polars as pl

    history = identifier_history(universe, column)
    if history.get_column("start_date").null_count() == history.height:  # no spells to be as of
        keyed = panel.join(history.select("yukka_id", column), left_on="entity", right_on="yukka_id", how="inner")
    else:
        keyed = (
            panel.sort(on)
            .join_asof(
                history.select("yukka_id", column, "start_date").sort("start_date"),
                left_on=on,
                right_on="start_date",
                by_left="entity",
                by_right="yukka_id",
                strategy="backward",
                # both sides are sorted by the key above; polars cannot verify that per by-group
                # and warns on every call unless told not to look
                check_sortedness=False,
            )
            # a day before a company's first spell takes its first identifier: still PIT-honest,
            # since no earlier one exists, and still no row dropped
            .join(
                history.group_by("yukka_id").agg(pl.col(column).first().alias("_first")),
                left_on="entity",
                right_on="yukka_id",
                how="left",
            )
            .with_columns(pl.col(column).fill_null(pl.col("_first")))
            .drop("_first")
            .drop_nulls(column)
        )
    return keyed.select(column, *[name for name in panel.columns if name != "entity"]).sort(column, on)


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

    @property
    def counts(self) -> Counts:
        """The paper's daily article counts over this read, as an aggregation not yet run.

            >>> data("company:bmw").snippets.months("2025-01").counts.frame   # doctest: +SKIP

        Narrow first, then ask: the counts are aggregated from exactly the files this read
        selects. Any :meth:`select` is overridden — the aggregation reads
        :data:`COUNT_COLUMNS` and never the text.

        Returns:
            A :class:`Counts` over the same files.
        """
        return Counts(self)

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
        return f"Snippets({len(self._names)} companies, months {self._span}, {len(self._columns)} columns)"

    @property
    def _span(self) -> str:
        """The month span, as the reprs print it.

        Returns:
            ``first:last``, with an ellipsis for an end left open.
        """
        return f"{self._first or '…'}:{self._last or '…'}"

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


class Counts:
    """`P` and `N` per company and day, aggregated from the snippets of a read.

        >>> data("company:bmw").snippets.months("2025-01").counts.frame    # doctest: +SKIP

    The store holds no counts, so these are derived rather than imported: the aggregation runs
    inside DuckDB over the parquet, and only the daily panel crosses into Python. That is what
    makes a panel over the whole universe a plausible read where a frame of it is not — the
    ``snippet`` column, which is most of the bytes, is never fetched.

    **The grain is the article, not the snippet.** One row per ``(entity, article_id)`` is the
    paper's `tone(d)`: the sign of the summed sentiment of every snippet that article carries for
    that company. An article calling a company positive three times and negative once is
    positive — once. Counting snippet rows instead would scale both counts by the fan-out.

    **Uncharacterised is not neutral.** An article whose snippets carry no sentiment this layer
    scores — see :data:`TONE` — has no tone at all, which is not the same as a tone of zero. Those
    articles are counted in a column of their own rather than folded into ``neutral`` or dropped
    in silence.

    `S = (P + 1) / (P + N + 2)` is deliberately not computed here. The Laplace ratio is what
    production trades, and a second definition of the expression the comparison rests on is the
    one thing worth not having.
    """

    def __init__(self, snippets: Snippets, *, universe: pl.DataFrame | None = None, key: str | None = None) -> None:
        """Describe an aggregation over the files a snippets read selects. Reaches no network.

        Args:
            snippets: The read to aggregate, narrowed however it is wanted. Its column selection
                is replaced by :data:`COUNT_COLUMNS`, which is all the aggregation reads.
            universe: The frame the assets came in on, kept so :meth:`by` can read another of its
                identifier columns. ``None`` when the caller passed ids rather than a frame.
            key: The identifier column :meth:`by` was asked for, or ``None`` for the store's own.
        """
        self._snippets = snippets.select(*COUNT_COLUMNS)
        self._universe = universe
        self._key = key

    def months(self, first: str | None = None, last: str | None = None) -> Counts:
        """Narrow to a span of months, both ends included.

        The same narrowing as :meth:`Snippets.months`, so a panel can be asked for without
        going through the snippets read to get at it.

        Args:
            first: Earliest month to keep, ``"YYYY-MM"``. ``None`` leaves the start open.
            last: Latest month to keep, ``"YYYY-MM"``. Left ``None`` it repeats *first*.

        Returns:
            A new :class:`Counts` over the narrowed span.

        Raises:
            ValueError: If a month is not ``"YYYY-MM"``, or the span runs backwards.
        """
        return self._with(self._snippets.months(first, last))

    def by(self, column: str, universe: pl.DataFrame | None = None) -> Counts:
        """Key the panel by another of the universe's identifiers instead of the ``yukka_id``.

            >>> data(universe).counts.by("ric").frame        # doctest: +SKIP

        The keying is **point in time**: each day takes the identifier the company held on that
        day, via :func:`as_of` over :func:`identifier_history`. A day no membership spell covers
        keeps the last identifier in force rather than being dropped — masking a universe is the
        caller's job, not a relabelling's. The store knows nothing about RICs, so the frame the
        assets arrived on is what this reads; passing ids rather than a frame means passing the
        frame here instead.

        Args:
            column: An identifier column of the universe frame, e.g. ``"ric"`` or ``"isin"``.
            universe: The frame to read it off, or ``None`` to use the one :func:`data` was given.

        Returns:
            A new :class:`Counts` whose :attr:`frame` carries *column* in place of ``entity``.

        Raises:
            ValueError: If no frame is available, or it has no such column.
        """
        frame = self._universe if universe is None else universe
        if frame is None:
            msg = f"by({column!r}) needs the universe frame: pass it here, or to data() instead of plain ids"
            raise ValueError(msg)
        identifier_history(frame, column)  # resolved here too, so a bad column fails now, not at the read
        return Counts(self._snippets, universe=frame, key=column)

    @property
    def snippets(self) -> Snippets:
        """The read underneath, for what it already knows about itself.

        Returns:
            The :class:`Snippets` this aggregates, so :attr:`Snippets.rows`,
            :attr:`Snippets.coverage` and :attr:`Snippets.missing` answer for the counts too —
            they describe the same files.
        """
        return self._snippets

    @property
    def frame(self) -> pl.DataFrame:
        """Run the aggregation and return the daily panel.

        Returns:
            One row per company and day, in that order: ``entity`` (the ``yukka_id``), ``day``,
            ``positive``, ``neutral``, ``negative``, ``uncharacterised`` — or, after :meth:`by`,
            the identifier that company held **on that day** in place of ``entity``.
            A day is the **UTC** calendar day of ``publish_time``,
            fixed by the query rather than left to the connection's timezone. A day a company has
            no article on is absent, not zero.

        Raises:
            NoSnippetsError: If the narrowing left no file to read.
        """
        import polars as pl

        with self._snippets._open() as connection:
            panel = pl.DataFrame(self.relation(connection))
        if self._key is None:
            return panel
        return as_of(panel, self._universe, self._key, on="day")

    def relation(self, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation:
        """The aggregation as a DuckDB relation, for a query that goes further than a panel.

        The escape hatch, as :meth:`Snippets.relation` is for the rows: a relation composes into
        SQL, so a join of these counts against prices or a window over them runs inside DuckDB
        and only its result is materialised.

            >>> with data("company:bmw").snippets.months("2025-01") as held:   # doctest: +SKIP
            ...     rel = held.counts.relation()
            ...     rel.aggregate("entity, sum(positive), sum(negative)", "entity").fetchall()

        Args:
            connection: An open connection, from :class:`~novetly_score.data.duck.OpenDuck`.
                Left ``None`` it is the connection the underlying read holds.

        A relation is keyed by ``entity`` whatever :meth:`by` asked for: the mapping to a RIC
        lives in the universe and not in the store, so relabelling happens where the panel is
        materialised. A query that wants it keyed otherwise joins :func:`identifiers` itself.

        Returns:
            A relation over the panel, keyed by ``entity``. Nothing is read until it is.

        Raises:
            NoSnippetsError: If the narrowing left no file to read.
            RuntimeError: If no connection was passed and none is held.
        """
        return self._snippets.relation(connection).query("snippets", _COUNTS_SQL)

    def __repr__(self) -> str:
        """Describe the aggregation without running it.

        Returns:
            A one-line summary of companies and span.
        """
        keyed = "" if self._key is None else f", by {self._key}"
        return f"Counts({len(self._snippets._names)} companies, months {self._snippets._span}{keyed})"

    def _with(self, snippets: Snippets) -> Counts:
        """Copy this aggregation over a differently narrowed read.

        Args:
            snippets: The narrowed read.

        Returns:
            A new :class:`Counts` keeping the universe and the key.
        """
        return Counts(snippets, universe=self._universe, key=self._key)


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

        A frame is kept as well as resolved. It is the only thing that knows a company's other
        identifiers, so keeping it is what lets :meth:`Counts.by` key a panel by RIC without this
        layer knowing what a RIC is.

        Args:
            assets: One asset, an iterable of them, or a frame with a ``yukka_id`` column.
            connection: An open connection for the reads to use, or ``None`` to open one per read.

        Raises:
            ValueError: If no asset was given.
        """
        self._names = entities(assets)
        self._connection = connection
        self._frame = assets if hasattr(assets, "columns") and hasattr(assets, "get_column") else None
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

    def counts(
        self,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
        *,
        key: str | None = None,
    ) -> pl.DataFrame:
        """Read the daily article counts for these assets over a span, and return them.

            >>> data(Index.STOXX600.frame).counts("2025-01", "2025-06", key="ric")   # doctest: +SKIP

        Unlike the rest of the layer this one *runs*: the span is the narrowing, so there is
        nothing left to describe. :attr:`Snippets.counts` is the same panel as a value that can
        still be narrowed, composed into SQL, or asked what it would cost.

        The store files snippets by month, so a day-bounded span reads whole months and trims the
        days afterwards — the bytes are the month's either way. It also reads **one month either
        side**, because the files are not UTC-day aligned: the ``2025-01`` file holds days through
        2025-02-03, so the month a day falls in is not reliably the file that holds it. Without
        the padding a day query would quietly miss the rows the neighbouring file kept.

        Args:
            start_date: Earliest month or day to include: ``"2025-01"``, ``"2025-01-15"`` or a
                :class:`datetime.date`. ``None`` for the store's earliest.
            end_date: Latest, included, in the same three shapes. ``None`` for the store's
                latest. Either end left open runs to the edge of what the store holds, so
                ``counts("2025-01")`` is 2025-01 onwards and ``counts()`` is everything.
            key: An identifier column of the universe frame to key the panel by instead of the
                ``yukka_id``, e.g. ``"ric"``. Needs the assets to have come in as a frame.

        Returns:
            One row per company and day: ``entity`` (or *key*), ``day``, ``positive``,
            ``neutral``, ``negative``, ``uncharacterised``.

        Raises:
            NoSnippetsError: If the span selects no file for any of these assets.
            ValueError: If a span end is neither ``YYYY-MM`` nor ``YYYY-MM-DD``, if the span runs
                backwards, or if *key* is not a column of the universe frame.
        """
        import polars as pl

        first, first_day = _span_end(start_date, "start_date")
        last, last_day = _span_end(end_date, "end_date")
        backwards = (first_day is not None and last_day is not None and last_day < first_day) or (
            first is not None and last is not None and last < first
        )
        if backwards:
            msg = f"counts({start_date!r}, {end_date!r}) runs backwards"
            raise ValueError(msg)
        # An end left as None stays None, which is the manifest's "no bound on that side"; a day
        # moves its month one out, because the files are not UTC-day aligned.
        read = self.snippets._with(
            first=first if first_day is None else _shift_month(first, -1),
            last=last if last_day is None else _shift_month(last, 1),
        )
        counts = Counts(read, universe=self._frame)
        frame = (counts if key is None else counts.by(key)).frame
        if first_day is not None:
            frame = frame.filter(pl.col("day") >= first_day)
        if last_day is not None:
            frame = frame.filter(pl.col("day") <= last_day)
        return frame

    def prices(
        self,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
        *,
        adjustment: str = "net",
        key: str | None = None,
    ) -> pl.DataFrame:
        """Read EUR levels for these assets over a span, and return them.

            >>> data(Index.STOXX600.frame).prices("2025-01", "2025-06", key="ric")   # doctest: +SKIP

        The companion to :meth:`counts`, and deliberately the same shape of call: the span is the
        argument, the frame is the answer, and either end left open runs to the edge of what the
        feed holds. See :mod:`novetly_score.data.prices` for what the three adjustments are and
        why a total-return level is only meaningful as a ratio.

        This needs the universe frame, not ids: the store knows what a company is, and only the
        frame knows which line its prices are filed under.

        Args:
            start_date: Earliest month or day, or ``None`` for the feed's history start.
            end_date: Latest month or day, **included**, or ``None`` for today.
            adjustment: ``"net"`` (dividends reinvested after withholding tax, the default and
                what a UCITS receives), ``"gross"`` (before it), or ``"raw"`` (the converted
                close, no dividends). Dividend ex-dates are a 1.5-3% step down in a raw series,
                clustered in the European spring, so a signal measured against ``raw`` is
                measured against that too.
            key: An identifier column of the universe frame to key by — point in time, as the
                counts are — or ``None`` for the ``yukka_id``.

        Returns:
            ``entity`` (or *key*), ``day``, ``close_price``, sorted by both. Days a name had no
            valid close are absent rather than forward-filled.

        Raises:
            ValueError: If the assets did not arrive as a frame, if *adjustment* is unknown, if a
                span end is malformed, or if the span runs backwards.
        """
        from novetly_score.data import prices as feed

        if self._frame is None:
            msg = "prices() needs the universe frame: pass it to data() instead of plain ids"
            raise ValueError(msg)
        first, last = feed.window(start_date, end_date)
        rics = self._frame.get_column("ric").drop_nulls().unique().to_list()
        return feed.keyed(feed.fetch(rics, first, last, adjustment), self._frame, key=key)

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


def _span_end(value: str | date | None, which: str) -> tuple[str | None, date | None]:
    """Read a span end as the month to read and, if a day was given, the day to trim to.

    Args:
        value: ``"YYYY-MM"``, ``"YYYY-MM-DD"``, a :class:`datetime.date`, or ``None``.
        which: Which end it is, for the error message.

    Returns:
        ``(month, day)`` — the month to pass to :meth:`Snippets.months`, and the day to filter by
        or ``None`` when the end was a whole month.

    Raises:
        ValueError: If *value* is a string in neither shape.
    """
    if value is None:
        return None, None
    if isinstance(value, str):
        text = value.strip()
        if MONTH.match(text):
            return text, None
        try:
            value = date.fromisoformat(text)
        except ValueError:
            msg = f"{which} {text!r} is not YYYY-MM or YYYY-MM-DD, e.g. '2025-01' or '2025-01-15'"
            raise ValueError(msg) from None
    return f"{value.year:04d}-{value.month:02d}", value


def _month_end(month: str) -> date:
    """The last day of *month*.

    Args:
        month: A month, ``"YYYY-MM"``.

    Returns:
        Its final day, so a month given as a span's end means all of it.
    """
    following = _shift_month(month, 1)
    return date.fromisoformat(f"{following}-01") - timedelta(days=1)


def _shift_month(month: str, delta: int) -> str:
    """The month *delta* months from *month*.

    Args:
        month: A month, ``"YYYY-MM"``.
        delta: How many months to move, negative for earlier.

    Returns:
        The shifted month, ``"YYYY-MM"``.
    """
    index = int(month[:4]) * 12 + int(month[5:7]) - 1 + delta
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


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
