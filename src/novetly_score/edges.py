"""Point-in-time nearest-neighbour edges between a company's snippets, on the trading-day grid.

Two stages, each a command, each safe to re-run::

    uv run python -m novetly_score.edges build     --entities airbus
    uv run python -m novetly_score.edges benchmark --entities airbus,volkswagen

``build`` writes three linked tables per company; ``benchmark`` reports throughput and peak
memory on the names that decide whether a sweep fits. A company is the unit of work *and* of
resume: ``build`` skips a company whose three outputs already exist, so an interrupted sweep
re-runs the same command rather than starting over.

**Nothing may see its own future.** Every snippet is assigned the trading day it could first have
been traded on, and a target on day *t* is compared only against sources in
``[t - LOOKBACK trading days, t)``. Same-day pairs are excluded by construction rather than by a
filter, so there is no arrangement of the data that leaks one.

**The unit is the distinct text on a trading day, not the snippet row.** The same text recurs
across articles and sources -- 2.6 rows per distinct text -- so one canonical representative is
kept per ``(trading_date, vector_key)``, the earliest by ``publish_time``, carrying
``same_day_count``. ``snippet_map`` records every raw row's route to its representative, so
nothing is lost by the collapse.

**``vector_key`` is the whole deduplication story.** It is ``md5(snippet)``, which is also the key
the embedding vectors are stored under, so identical text already has exactly one vector. The
upstream data carries no reprint or thread label -- there is no ``duplicate_group_id`` in these
twelve columns -- and hashing the 256-dim embedding would add a 58 GB scan to catch only distinct
texts that embed bit-identically. One axis, derivable by anyone, in any engine.

**Edges are keyed by an integer, not a UUID.** A random UUID is sixteen bytes of entropy that zstd
cannot touch, and measured at 16.6 bytes per edge against 6.8 for an ``INTEGER`` -- 54.8 GB against
22.5 GB across the whole universe at ``TOP_K`` 50. ``snippet_key`` is dense per company and
resolves to its UUID through ``snippet_meta``, with ``snippet_map`` reaching the raw rows behind it.

The output sits beside the vectors it reads, keyed by model, width, *k* and window::

    edges/snippets/text-embedding-3-small-d256/k50-w20/
        snippet_meta/<company>.parquet    one row per canonical snippet
        snippet_map/<company>.parquet     one row per raw snippet
        edges/<company>.parquet           up to TOP_K rows per canonical snippet

**A GPU is not what makes this fast.** The matmul is 256-dim and measured at roughly 0.1 h on a
T4 against 0.7 h on sixteen CPU cores for the entire universe; the cost is in reading vectors and
in the selection, which is why ``P`` is chunked and a single running ``topk`` is kept rather than
one pass per *k*. The device is chosen automatically and a CPU host is a perfectly good one.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import polars as pl

from novetly_score.data.duck import BUCKET, OpenDuck
from novetly_score.data.embed import DIMENSIONS, MODEL, upload
from novetly_score.data.embed import Layout as VectorLayout
from novetly_score.data.s3 import OpenClient
from novetly_score.data.snippets import MANIFEST, SNIPPETS_PREFIX, entities

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from duckdb import DuckDBPyConnection

#: Trading days in the trailing window, counted strictly before the target's own day.
LOOKBACK: Final = 20

#: Nearest sources kept per target. Narrowing later is a filter; widening later is a recompute,
#: and at 6.8 bytes an edge the whole universe is 22.5 GB, so the wide choice is the cheap one.
TOP_K: Final = 50

#: Columns of ``P`` per matmul block. Bounds the distance matrix rather than the window: at this
#: width the worst measured day peaks near 0.5 GB, which fits any device.
CHUNK: Final = 16_384

#: Local hour at or after which an article belongs to the next trading day.
CLOSE_HOUR: Final = 16

#: Zone the close is read in. The store's ``publish_time`` is timezone-aware, so this only has
#: to name the market's own clock, not the data's.
TIMEZONE: Final = "Europe/London"

#: The three tables ``build`` writes, and the order they are written in. ``edges`` is last
#: because its presence is what marks a company done.
STAGES: Final = ("snippet_meta", "snippet_map", "edges")

#: Columns of the edge list, in order.
EDGE_COLUMNS: Final = (
    "company_id",
    "target_snippet_key",
    "target_trading_date",
    "source_snippet_key",
    "source_trading_date",
    "trading_days_apart",
    "cosine_distance",
    "rank",
)

_SQL_FILE_LIST: Final = """
    SELECT entity, month FROM read_parquet($manifest)
    WHERE rows > 0 AND entity = $entity
      AND ($first IS NULL OR month >= $first) AND ($last IS NULL OR month <= $last)
    ORDER BY month
"""


class EdgeRunError(RuntimeError):
    """The run cannot proceed, or produced nothing where something was expected."""


@dataclass(frozen=True)
class Layout:
    """Where one run's three tables live.

    Keyed by the vectors' model and width as well as by *k* and the window, so a re-run under
    different parameters lands beside the first rather than on top of it.
    """

    model: str = MODEL
    dimensions: int = DIMENSIONS
    top_k: int = TOP_K
    lookback: int = LOOKBACK
    bucket: str = BUCKET

    @property
    def prefix(self) -> str:
        """The run's own prefix under the bucket.

        Returns:
            A key prefix, e.g. ``edges/snippets/text-embedding-3-small-d256/k50-w20``.
        """
        return f"edges/snippets/{self.model}-d{self.dimensions}/k{self.top_k}-w{self.lookback}"

    def key(self, stage: str, company: str) -> str:
        """Name one company's object in one stage.

        Args:
            stage: One of :data:`STAGES`.
            company: The snippet directory name, e.g. ``"airbus"``.

        Returns:
            The object key.
        """
        return f"{self.prefix}/{stage}/{company}.parquet"

    def glob(self, stage: str) -> str:
        """Name every company's object in one stage, as DuckDB reads them.

        Args:
            stage: One of :data:`STAGES`.

        Returns:
            An ``s3://`` glob.
        """
        return f"s3://{self.bucket}/{self.prefix}/{stage}/*.parquet"

    def url(self, stage: str, company: str) -> str:
        """Name one company's object as a URL.

        Args:
            stage: One of :data:`STAGES`.
            company: The snippet directory name.

        Returns:
            An ``s3://`` URL.
        """
        return f"s3://{self.bucket}/{self.key(stage, company)}"


def trading_date_sql(column: str = "publish_time", *, close_hour: int = CLOSE_HOUR, timezone: str = TIMEZONE) -> str:
    """The SQL expression mapping a timestamp to the trading day it can first be traded on.

    Two rolls, in order. A timestamp at or after *close_hour* in *timezone* missed that day's
    close, so it belongs to the next day. A day that has no close -- Saturday or Sunday, whether
    the article landed there directly or was rolled there by the first step -- moves on to
    Monday.

    Args:
        column: The timestamp column to read.
        close_hour: Local hour of the close. ``16`` means 16:00 is already too late.
        timezone: Zone to read the close in.

    Returns:
        A SQL expression of type ``DATE``.
    """
    local = f"({column} AT TIME ZONE '{timezone}')"
    rolled = (
        f"CAST({local} AS DATE) + "
        f"CASE WHEN extract('hour' FROM {local}) >= {close_hour} THEN INTERVAL 1 DAY ELSE INTERVAL 0 DAY END"
    )
    return (
        f"CAST({rolled} + CASE isodow({rolled}) "
        f"WHEN 6 THEN INTERVAL 2 DAY "
        f"WHEN 7 THEN INTERVAL 1 DAY "
        f"ELSE INTERVAL 0 DAY END AS DATE)"
    )


def device() -> Any:  # a torch type, importable only at runtime
    """Pick the fastest device present, preferring CUDA, then Apple's Metal, then the CPU.

    The CPU branch is not a degraded mode: the matmul is 256-dim and the whole universe measures
    about 0.7 h on sixteen cores against 0.1 h on a T4, so a CPU host is a reasonable choice and
    the only one CI ever takes.

    Returns:
        A ``torch.device``.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def peak_memory(target: Any) -> int:  # a torch type, see device
    """Report peak memory on *target*, in bytes, however that device measures it.

    Args:
        target: The device the run used.

    Only CUDA keeps a true high-water mark. Metal exposes the *current* allocation, which reads
    zero the moment the tensors are freed and so cannot size anything, and the CPU keeps none at
    all -- for both, the process's resident high-water mark is reported instead. That figure
    includes the interpreter and is not comparable with the CUDA one, which is why the benchmark
    prints the device alongside it.

    Args:
        target: The device the run used.

    Returns:
        Peak bytes, always greater than zero.
    """
    import torch

    if target.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if sys.platform == "win32":  # pragma: no cover — only the Windows CI job takes this branch
        return _windows_peak_working_set()
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return int(usage if sys.platform == "darwin" else usage * 1024)


def _windows_peak_working_set() -> int:  # pragma: no cover — Windows only
    """Peak working set of this process, Windows' equivalent of a peak RSS.

    ``resource`` is a Unix-only module, so the obvious implementation imports nothing on Windows
    and the whole benchmark command dies with ``ModuleNotFoundError``. ``GetProcessMemoryInfo``
    is the platform's own answer and reports a genuine high-water mark rather than a stub.

    Returns:
        Peak bytes of working set.
    """
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        """``PROCESS_MEMORY_COUNTERS``, in the order the API fills it."""

        _fields_ = (
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        )

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]  # ty: ignore
    ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]  # ty: ignore
        handle, ctypes.byref(counters), counters.cb
    )
    return int(counters.PeakWorkingSetSize)


def file_list(con: DuckDBPyConnection, company: str, first: str | None, last: str | None) -> list[str]:
    """Name one company's snippet files, from the manifest rather than a listing.

    Args:
        con: An open connection.
        company: The snippet directory name.
        first: Earliest month, ``"YYYY-MM"``, or ``None``.
        last: Latest month, ``"YYYY-MM"``, or ``None``.

    Returns:
        ``s3://`` paths, in month order.

    Raises:
        EdgeRunError: If the selection names no file.
    """
    rows = con.sql(
        _SQL_FILE_LIST,
        params={"manifest": MANIFEST, "entity": company, "first": first, "last": last},
    ).fetchall()
    if not rows:
        msg = f"no snippet file for {company!r} in that month range"
        raise EdgeRunError(msg)
    return [f"s3://{BUCKET}/{SNIPPETS_PREFIX}/{entity}/{month}.parquet" for entity, month in rows]


def existing(stage: str, layout: Layout) -> set[str]:
    """Name the companies already written for one stage.

    Args:
        stage: One of :data:`STAGES`.
        layout: The run's layout.

    Returns:
        Company directory names.
    """
    prefix = f"{layout.prefix}/{stage}/"
    found: set[str] = set()
    with OpenClient() as s3:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=layout.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                found.add(item["Key"].removeprefix(prefix).removesuffix(".parquet"))
    return found


def _calendar_sql(calendar: Sequence[date] | None) -> str:
    """The relation supplying the trading-day grid the window is counted over.

    Args:
        calendar: The trading days to index, or ``None``.

    Returns:
        A SQL relation yielding one ``trading_date`` column.
    """
    if calendar is not None:
        return "SELECT UNNEST($calendar::DATE[]) AS trading_date"
    # Every weekday in the data's own range. This counts a public holiday as a trading day and so
    # widens the window by however many holidays it spans; pass the price feed's own dates to
    # make the grid exact.
    return """
        SELECT day AS trading_date
        FROM (SELECT UNNEST(generate_series(
                  (SELECT min(trading_date) FROM raw),
                  (SELECT max(trading_date) FROM raw),
                  INTERVAL 1 DAY)) AS day)
        WHERE isodow(day) <= 5
    """


def canonical(
    con: DuckDBPyConnection,
    company: str,
    paths: Sequence[str],
    *,
    lookback: int = LOOKBACK,
    close_hour: int = CLOSE_HOUR,
    timezone: str = TIMEZONE,
    calendar: Sequence[date] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Collapse a company's snippets to one canonical row per text per trading day, and flag reprints.

    The reprint columns are strictly point-in-time: they come from window functions that look only
    backwards along the trading-day grid, so no row can be told about its own future. A reprint of
    a text first seen earlier is flagged; the first occurrence is not.

    Args:
        con: An open connection.
        company: The snippet directory name, used as ``company_id``.
        paths: The company's snippet files.
        lookback: Trading days a reprint is still "known" within.
        close_hour: Local hour of the close.
        timezone: Zone the close is read in.
        calendar: Trading days to index over, or ``None`` for every weekday in range.

    Returns:
        ``(meta, mapping, dropped)``: the canonical table, the raw-to-canonical lookup, and the
        number of raw rows whose trading day was absent from *calendar* and so could not be
        placed. *meta* carries a trailing ``idx``, the trading-day grid position, which the
        windowing needs and :func:`build` drops before writing.

    Raises:
        EdgeRunError: If the company has no usable snippet.
    """
    # No filter on `entity` is needed or wanted: `paths` already names one company's files, and
    # the column holds the full yukka_id (`company:airbus`) rather than the directory name
    # (`airbus`) that `company_id` and the CLI both use.
    # DuckDB rejects a bound parameter the query does not mention, so each statement gets only
    # the parameters it actually reads.
    trading_date = trading_date_sql(close_hour=close_hour, timezone=timezone)
    grid: dict[str, object] = {"calendar": list(calendar)} if calendar is not None else {}

    con.sql(
        f"""
        CREATE OR REPLACE TABLE raw AS
        SELECT snippet_id, md5(snippet) AS vector_key, publish_time, {trading_date} AS trading_date
        FROM read_parquet($paths)
        WHERE snippet IS NOT NULL AND length(snippet) > 0
        """,  # noqa: S608  # nosec B608 — the one interpolation is trading_date_sql's own output,
        # built from an int and a zone name; every path and id is a bound parameter
        params={"paths": list(paths)},
    ).fetchall()
    (raw_rows,) = con.sql("SELECT count(*) FROM raw").fetchall()[0]
    if not raw_rows:
        msg = f"{company!r} has no snippet with text"
        raise EdgeRunError(msg)

    con.sql(
        f"""
        CREATE OR REPLACE TABLE cal AS
        SELECT trading_date, DENSE_RANK() OVER (ORDER BY trading_date) AS idx
        FROM ({_calendar_sql(calendar)})
        """,  # noqa: S608  # nosec B608 — _calendar_sql returns one of two module literals
        params=grid,
    ).fetchall()

    # One canonical row per (trading_date, vector_key): the earliest publish_time, with
    # snippet_id breaking a tie so the choice is identical on every run.
    con.sql(
        """
        CREATE OR REPLACE TABLE canon AS
        SELECT trading_date, vector_key, snippet_id AS canonical_snippet_id, same_day_count
        FROM (
            SELECT *,
                   row_number() OVER (PARTITION BY trading_date, vector_key
                                      ORDER BY publish_time, snippet_id) AS rn,
                   count(*) OVER (PARTITION BY trading_date, vector_key) AS same_day_count
            FROM raw
        )
        WHERE rn = 1
        """
    ).fetchall()

    # The grid join is an inner join, so a day absent from a supplied calendar drops its rows.
    # Count them rather than letting them vanish.
    (placed,) = con.sql("SELECT count(*) FROM canon JOIN cal USING (trading_date)").fetchall()[0]
    (total,) = con.sql("SELECT count(*) FROM canon").fetchall()[0]
    dropped = int(total - placed)

    con.sql(
        """
        CREATE OR REPLACE TABLE meta AS
        SELECT $company AS company_id,
               CAST(row_number() OVER (ORDER BY trading_date, canonical_snippet_id) - 1 AS INTEGER) AS snippet_key,
               canonical_snippet_id,
               trading_date,
               idx,
               vector_key,
               CAST(same_day_count AS INTEGER) AS same_day_count,
               CAST(window_reprints AS INTEGER) AS window_reprint_count,
               window_reprints > 0 AS is_known_reprint,
               CASE WHEN window_reprints > 0 THEN CAST(idx - prev_idx AS INTEGER) END AS days_since_last_reprint,
               CASE WHEN window_reprints > 0 THEN prev_id END AS matched_prior_snippet_id
        FROM (
            SELECT c.*, g.idx,
                   count(*) OVER (PARTITION BY c.vector_key ORDER BY g.idx
                                  RANGE BETWEEN $lookback PRECEDING AND 1 PRECEDING) AS window_reprints,
                   lag(g.idx) OVER (PARTITION BY c.vector_key ORDER BY g.idx) AS prev_idx,
                   lag(c.canonical_snippet_id) OVER (PARTITION BY c.vector_key ORDER BY g.idx) AS prev_id
            FROM canon AS c
            JOIN cal AS g USING (trading_date)
        )
        """,
        params={"company": company, "lookback": lookback},
    ).fetchall()

    mapping = pl.DataFrame(
        con.sql(
            """
            SELECT $company AS company_id, r.snippet_id, m.canonical_snippet_id, m.snippet_key, m.trading_date
            FROM raw AS r
            JOIN meta AS m ON m.trading_date = r.trading_date AND m.vector_key = r.vector_key
            ORDER BY m.snippet_key, r.snippet_id
            """,
            params={"company": company},
        )
    )
    frame = pl.DataFrame(
        con.sql(
            """
            SELECT company_id, canonical_snippet_id, snippet_key, trading_date, vector_key,
                   same_day_count, window_reprint_count, is_known_reprint,
                   days_since_last_reprint, matched_prior_snippet_id, idx
            FROM meta ORDER BY snippet_key
            """
        )
    )
    return frame, mapping, dropped


def stage(
    companies: Sequence[str],
    into: str,
    layout: Layout,
    *,
    first: str | None = None,
    last: str | None = None,
) -> dict[str, int]:
    """Copy just the vectors the named companies need into one local parquet, in a single pass.

    This exists because of a measurement. A company's canonical texts are about 2% of the store,
    scattered uniformly across all 256 shards by their MD5, so nothing can be pruned by row-group
    statistics and the join reads essentially the whole 58 GB: **213 seconds for one month of one
    company**. Paid per company that is 37 hours of lookup before any distance is computed. Paid
    once for every company in the run, it is 213 seconds total.

    Run this before :func:`build` and pass the result as its *vectors*.

    Args:
        companies: Snippet directory names the run will cover.
        into: Local parquet path to write.
        layout: The run's layout, naming the vectors' model and width.
        first: Earliest month, or ``None``.
        last: Latest month, or ``None``.

    Returns:
        ``{"companies": …, "texts": …, "vectors": …}``.

    Raises:
        EdgeRunError: If the vectors cannot be read, or a text the companies need has no vector.
    """
    import duckdb

    source = VectorLayout(model=layout.model, dimensions=layout.dimensions).glob("vectors")
    with OpenDuck(memory_limit="8GB") as con:
        paths: list[str] = []
        for company in companies:
            paths.extend(file_list(con, company, first, last))
        print(f"staging vectors for {len(companies)} companies from {len(paths)} snippet files", flush=True)
        con.sql(
            """
            CREATE OR REPLACE TABLE wanted AS
            SELECT DISTINCT md5(snippet) AS snippet_id FROM read_parquet($paths)
            WHERE snippet IS NOT NULL AND length(snippet) > 0
            """,
            params={"paths": paths},
        ).fetchall()
        (texts,) = con.sql("SELECT count(*) FROM wanted").fetchall()[0]
        started = time.monotonic()
        try:
            con.sql(
                "COPY (SELECT v.snippet_id, v.embedding FROM read_parquet($vectors) AS v "
                "SEMI JOIN wanted AS w ON w.snippet_id = v.snippet_id) "
                "TO $into (FORMAT parquet, COMPRESSION zstd)",
                params={"vectors": source, "into": into},
            ).fetchall()
        except duckdb.Error as exc:
            msg = f"no vector could be read from {source!r}: {exc}"
            raise EdgeRunError(msg) from exc
        (written,) = con.sql("SELECT count(*) FROM read_parquet($into)", params={"into": into}).fetchall()[0]
    print(f"{texts:,} texts wanted, {written:,} vectors staged in {time.monotonic() - started:,.0f}s", flush=True)
    if written != texts:
        msg = f"{texts - written:,} of {texts:,} texts have no vector; embed them before building edges"
        raise EdgeRunError(msg)
    return {"companies": len(companies), "texts": int(texts), "vectors": int(written)}


def _tensor(con: DuckDBPyConnection, vectors: str, dimensions: int, target: Any) -> Any:  # torch types
    """Load every canonical snippet's vector into one contiguous unit-norm tensor.

    Row *i* of the result is ``snippet_key`` *i*, which is what lets a day's slice of the tensor
    be taken by position with no copy and no lookup table.

    Args:
        con: An open connection holding the ``meta`` table.
        vectors: Parquet glob holding ``snippet_id`` and ``embedding``.
        dimensions: Width of the embedding.
        target: The device to put the tensor on.

    Returns:
        A ``torch.float32`` tensor of shape ``(n_canonical, dimensions)``.

    Raises:
        EdgeRunError: If any canonical snippet has no vector.
    """
    import duckdb
    import numpy as np
    import torch

    # The stored embedding is a variable-length LIST, so the cast to a fixed-width ARRAY is what
    # makes a wrong width an error here rather than a wrong distance downstream.
    try:
        frame = pl.DataFrame(
            con.sql(
                f"""
                SELECT m.snippet_key, CAST(v.embedding AS FLOAT[{dimensions}]) AS vec
                FROM meta AS m
                LEFT JOIN read_parquet($vectors) AS v ON v.snippet_id = m.vector_key
                ORDER BY m.snippet_key
                """,  # noqa: S608  # nosec B608 — dimensions is a validated int
                params={"vectors": vectors},
            )
        )
    except duckdb.Error as exc:
        msg = f"no vector could be read from {vectors!r}: {exc}"
        raise EdgeRunError(msg) from exc
    missing = frame.get_column("vec").is_null().sum()
    if missing:
        msg = f"{missing:,} canonical snippets have no vector; embed them before building edges"
        raise EdgeRunError(msg)

    block = np.array(frame.get_column("vec").to_numpy(), dtype=np.float32, copy=True).reshape(-1, dimensions)
    tensor = torch.from_numpy(block).to(target)
    return torch.nn.functional.normalize(tensor, p=2.0, dim=1)


def _window_first(window_rows: Any, index: Any, codes: Any) -> Any:  # torch types
    """Keep one row per ``vector_key`` in a candidate window: the earliest on the grid.

    This is the point-in-time window deduplication. Without it a text reprinted on ten of the
    twenty days would occupy ten of a target's slots, crowding out the distinct stories the score
    is actually about. Keeping the *earliest* is what makes it point-in-time: the choice depends
    only on the window's own contents.

    Args:
        window_rows: Row indices in the candidate window.
        index: Trading-day index per row.
        codes: Integer ``vector_key`` code per row.

    Returns:
        A subset of *window_rows*, in grid order, with no repeated ``vector_key``.
    """
    import torch

    ordered = window_rows[torch.argsort(index[window_rows], stable=True)]
    seen = codes[ordered]
    # A stable sort by code leaves equal codes in their original (grid) order, so the first
    # entry of each run is the earliest occurrence. Only argsort, comparison and indexing are
    # used: scatter_reduce_ with "amin" would be the obvious way and is unimplemented on Metal.
    order = torch.argsort(seen, stable=True)
    grouped = seen[order]
    starts = torch.ones_like(grouped, dtype=torch.bool)
    starts[1:] = grouped[1:] != grouped[:-1]
    return ordered[torch.sort(order[starts]).values]


def _neighbours(
    embeddings: Any,
    codes: Any,
    target_rows: Any,
    source_rows: Any,
    top_k: int,
    chunk: int,
) -> tuple[Any, Any]:
    """Find each target's nearest sources, chunking the candidate side to bound memory.

    One running ``topk`` is kept across chunks rather than one pass per *k*, because the cost here
    is the selection and not the matmul. Pairs sharing a ``vector_key`` are masked to infinity
    before selection: identical text has an identical vector and so distance zero, and unmasked
    those would fill the top of every list with reprints ``snippet_meta`` already records.

    Args:
        embeddings: The company's unit-norm tensor.
        codes: Integer ``vector_key`` codes, one per row of *embeddings*.
        target_rows: Row indices of the target day.
        source_rows: Row indices of the candidate window, already deduplicated by ``vector_key``.
        top_k: Sources to keep per target.
        chunk: Columns of the candidate side per block.

    Returns:
        ``(distances, positions)``, each ``(len(target_rows), kept)``, where *positions* index
        into *source_rows*. Entries that stayed infinite are the caller's to drop.
    """
    import torch

    targets = embeddings[target_rows]
    target_codes = codes[target_rows]
    kept = min(top_k, int(source_rows.numel()))
    best_d = torch.full((targets.shape[0], 0), float("inf"), device=targets.device)
    best_i = torch.full((targets.shape[0], 0), -1, dtype=torch.long, device=targets.device)

    for start in range(0, int(source_rows.numel()), chunk):
        window = source_rows[start : start + chunk]
        distance = 1.0 - targets @ embeddings[window].T
        distance[target_codes[:, None] == codes[window][None, :]] = float("inf")
        width = min(kept, int(window.numel()))
        picked_d, picked_i = torch.topk(distance, width, dim=1, largest=False)
        merged_d = torch.cat([best_d, picked_d], dim=1)
        merged_i = torch.cat([best_i, picked_i + start], dim=1)
        best_d, order = torch.topk(merged_d, min(kept, merged_d.shape[1]), dim=1, largest=False)
        best_i = torch.gather(merged_i, 1, order)
    return best_d, best_i


def build(
    company: str,
    layout: Layout,
    *,
    first: str | None = None,
    last: str | None = None,
    vectors: str | None = None,
    calendar: Sequence[date] | None = None,
    chunk: int = CHUNK,
    write: bool = True,
) -> dict[str, int]:
    """Build and write one company's three tables.

    Args:
        company: The snippet directory name.
        layout: Where to write, and the *k* and window to use.
        first: Earliest month, or ``None``.
        last: Latest month, or ``None``.
        vectors: Parquet glob of vectors, or ``None`` for the layout's own model and width.
        calendar: Trading days to index over, or ``None`` for every weekday in range.
        chunk: Columns of the candidate side per matmul block.
        write: Whether to upload. ``False`` computes everything and keeps it, for a benchmark.

    Returns:
        ``{"canonical": …, "raw": …, "edges": …, "dropped": …, "bytes": …}``.

    Raises:
        EdgeRunError: If the company has no snippet, or a canonical snippet has no vector.
    """
    import torch

    source = vectors or VectorLayout(model=layout.model, dimensions=layout.dimensions).glob("vectors")
    target = device()
    with OpenDuck(memory_limit="8GB") as con:
        paths = file_list(con, company, first, last)
        meta, mapping, dropped = canonical(con, company, paths, lookback=layout.lookback, calendar=calendar)
        embeddings = _tensor(con, source, layout.dimensions, target)

    keys = meta.get_column("vector_key")
    codes = torch.from_numpy(keys.cast(pl.Categorical).to_physical().to_numpy().astype("int64")).to(target)
    index = torch.from_numpy(meta.get_column("idx").to_numpy().astype("int64")).to(target)
    dates = meta.get_column("trading_date").to_list()
    day_index = sorted({int(value) for value in meta.get_column("idx").to_list()})

    frames: list[pl.DataFrame] = []
    for day in day_index:
        target_rows = torch.nonzero(index == day, as_tuple=True)[0]
        window_rows = torch.nonzero((index < day) & (index >= day - layout.lookback), as_tuple=True)[0]
        if not int(window_rows.numel()):
            continue
        sources = _window_first(window_rows, index, codes)
        distance, position = _neighbours(embeddings, codes, target_rows, sources, layout.top_k, chunk)
        frames.append(_edge_frame(company, target_rows, sources, position, distance, index, dates, day))

    edges = pl.concat(frames, how="vertical") if frames else _empty_edges()
    written = 0
    if write:
        written += upload(meta.drop("idx"), layout.key("snippet_meta", company), layout.bucket)
        written += upload(mapping, layout.key("snippet_map", company), layout.bucket)
        written += upload(edges, layout.key("edges", company), layout.bucket)
    return {
        "canonical": meta.height,
        "raw": mapping.height,
        "edges": edges.height,
        "dropped": dropped,
        "bytes": written,
    }


def _empty_edges() -> pl.DataFrame:
    """An edge frame with the right schema and no rows, for a company with one trading day.

    Returns:
        An empty frame carrying :data:`EDGE_COLUMNS`.
    """
    return pl.DataFrame(
        schema={
            "company_id": pl.String,
            "target_snippet_key": pl.Int32,
            "target_trading_date": pl.Date,
            "source_snippet_key": pl.Int32,
            "source_trading_date": pl.Date,
            "trading_days_apart": pl.Int16,
            "cosine_distance": pl.Float32,
            "rank": pl.Int16,
        }
    )


def _edge_frame(
    company: str,
    target_rows: Any,
    sources: Any,
    position: Any,
    distance: Any,
    index: Any,
    dates: Sequence[date],
    day: int,
) -> pl.DataFrame:
    """Turn one target day's selection into rows, dropping the slots the window could not fill.

    Args:
        company: The company id.
        target_rows: Row indices of the target day.
        sources: Row indices of the candidate window.
        position: Selected positions into *sources*.
        distance: Selected distances, aligned with *position*.
        index: Trading-day index per row.
        dates: Trading date per row.
        day: The target day's index.

    Returns:
        Rows in :data:`EDGE_COLUMNS` order, ranked from 1 by increasing distance.
    """
    import torch

    finite = torch.isfinite(distance)
    rows, slots = torch.nonzero(finite, as_tuple=True)
    if not int(rows.numel()):
        return _empty_edges()
    source_rows = sources[position[rows, slots]]
    target_keys = target_rows[rows]
    return pl.DataFrame(
        {
            "company_id": [company] * int(rows.numel()),
            "target_snippet_key": target_keys.to("cpu").numpy().astype("int32"),
            "target_trading_date": [dates[int(key)] for key in target_keys.tolist()],
            "source_snippet_key": source_rows.to("cpu").numpy().astype("int32"),
            "source_trading_date": [dates[int(key)] for key in source_rows.tolist()],
            "trading_days_apart": (day - index[source_rows]).to("cpu").numpy().astype("int16"),
            "cosine_distance": distance[rows, slots].to("cpu").numpy().astype("float32"),
            "rank": (slots + 1).to("cpu").numpy().astype("int16"),
        },
        schema=_empty_edges().schema,
    )


def sweep(
    companies: Sequence[str],
    layout: Layout,
    *,
    first: str | None = None,
    last: str | None = None,
    force: bool = False,
    vectors: str | None = None,
) -> dict[str, int]:
    """Build every company that is not already done.

    Args:
        companies: Snippet directory names.
        layout: The run's layout.
        first: Earliest month, or ``None``.
        last: Latest month, or ``None``.
        force: Rebuild companies that already have an ``edges`` object.
        vectors: Staged parquet from :func:`stage`, or ``None`` to read the store's own glob --
            which costs a full 58 GB scan per company and is almost never what you want.

    Returns:
        ``{"companies": …, "skipped": …, "edges": …, "bytes": …}``.
    """
    done = set() if force else existing("edges", layout)
    pending = [name for name in companies if name not in done]
    print(f"{len(companies)} companies: {len(companies) - len(pending)} already done, {len(pending)} to do", flush=True)
    totals = {"companies": 0, "skipped": len(companies) - len(pending), "edges": 0, "bytes": 0}
    for position, company in enumerate(pending, start=1):
        started = time.monotonic()
        counts = build(company, layout, first=first, last=last, vectors=vectors)
        totals["companies"] += 1
        totals["edges"] += counts["edges"]
        totals["bytes"] += counts["bytes"]
        print(
            f"  {company} ({position}/{len(pending)}): {counts['canonical']:,} canonical from "
            f"{counts['raw']:,} raw, {counts['edges']:,} edges, {counts['bytes'] / 1e6:,.0f} MB, "
            f"{time.monotonic() - started:,.0f}s"
            + (f", {counts['dropped']:,} off-calendar" if counts["dropped"] else ""),
            flush=True,
        )
    return totals


def benchmark(
    companies: Sequence[str], layout: Layout, *, chunk: int = CHUNK, vectors: str | None = None
) -> list[dict[str, float]]:
    """Measure throughput and peak memory without writing anything.

    Args:
        companies: Snippet directory names, heaviest first for a worst case.
        layout: The run's layout.
        chunk: Columns of the candidate side per matmul block.
        vectors: Staged parquet from :func:`stage`, or ``None`` for the store's own glob.

    Returns:
        One report per company.
    """
    target = device()
    print(f"device: {target}", flush=True)
    reports: list[dict[str, float]] = []
    for company in companies:
        started = time.monotonic()
        counts = build(company, layout, chunk=chunk, write=False, vectors=vectors)
        elapsed = time.monotonic() - started
        report = {
            "seconds": elapsed,
            "canonical": float(counts["canonical"]),
            "edges": float(counts["edges"]),
            "edges_per_second": counts["edges"] / elapsed if elapsed else 0.0,
            "peak_bytes": float(peak_memory(target)),
        }
        reports.append(report)
        print(
            f"  {company}: {counts['canonical']:,} canonical, {counts['edges']:,} edges in "
            f"{elapsed:,.0f}s = {report['edges_per_second']:,.0f} edges/s, "
            f"peak {report['peak_bytes'] / 1e9:,.2f} GB",
            flush=True,
        )
    return reports


def _split(value: str | None) -> list[str] | None:
    """Split a comma-separated argument.

    Args:
        value: The argument, or ``None``.

    Returns:
        The items, or ``None`` if there were none.
    """
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _companies(value: str | None) -> list[str]:
    """Resolve a comma-separated company argument to snippet directory names.

    Args:
        value: The argument.

    Returns:
        Directory names.

    Raises:
        EdgeRunError: If no company was named.
    """
    items = _split(value)
    if not items:
        msg = "name at least one company with --entities"
        raise EdgeRunError(msg)
    return list(entities(items))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one stage from the command line.

    Args:
        argv: Arguments, or ``None`` for ``sys.argv``.

    Returns:
        ``0`` if the stage succeeded, ``1`` if it reported a problem.
    """
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)

    parser = argparse.ArgumentParser(prog="novetly_score.edges", description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL, help=f"embedding model the vectors are under (default {MODEL})")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"sources per target (default {TOP_K})")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help=f"trading days (default {LOOKBACK})")
    parser.add_argument("--dimensions", type=int, default=DIMENSIONS, help=f"vector width (default {DIMENSIONS})")
    stages = parser.add_subparsers(dest="stage", required=True)

    keep = stages.add_parser("stage", help="copy the vectors these companies need to one local parquet")
    keep.add_argument("--entities", help="comma-separated company names")
    keep.add_argument("--months", help="FIRST:LAST, e.g. 2025-01:2025-03 (default: every month)")
    keep.add_argument("--into", required=True, help="local parquet path to write")

    run = stages.add_parser("build", help="write the three tables for each company")
    run.add_argument("--entities", help="comma-separated company names")
    run.add_argument("--months", help="FIRST:LAST, e.g. 2025-01:2025-03 (default: every month)")
    run.add_argument("--force", action="store_true", help="rebuild companies already done")
    run.add_argument("--vectors", default=None, help="staged parquet from the stage command (strongly advised)")

    mark = stages.add_parser("benchmark", help="measure throughput and peak memory, writing nothing")
    mark.add_argument("--entities", help="comma-separated company names")
    mark.add_argument("--chunk", type=int, default=CHUNK, help=f"candidate columns per block (default {CHUNK})")
    mark.add_argument("--vectors", default=None, help="staged parquet from the stage command")

    args = parser.parse_args(argv)
    layout = Layout(model=args.model, dimensions=args.dimensions, top_k=args.top_k, lookback=args.lookback)
    try:
        companies = _companies(args.entities)
        if args.stage == "stage":
            first, last = _months(args.months)
            stage(companies, args.into, layout, first=first, last=last)
        elif args.stage == "build":
            first, last = _months(args.months)
            sweep(companies, layout, first=first, last=last, force=args.force, vectors=args.vectors)
        else:
            benchmark(companies, layout, chunk=args.chunk, vectors=args.vectors)
    except EdgeRunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _months(span: str | None) -> tuple[str | None, str | None]:
    """Split a ``FIRST:LAST`` month span.

    Args:
        span: ``"2025-01:2025-03"``, ``"2025-01"`` for one month, ``"2025-01:"`` for open-ended,
            or ``None`` for every month.

    Returns:
        ``(first, last)``.
    """
    if not span:
        return None, None
    first, separator, last = span.partition(":")
    return (first.strip() or None), ((last.strip() or None) if separator else first.strip() or None)


if __name__ == "__main__":  # pragma: no cover — the command line
    raise SystemExit(main())
