"""The counts panel, derived once and kept in the store instead of recomputed per caller.

    from novetly_score.data import write_counts

    write_counts()                              # once, after the store changes
    tone = data(universe).counts(key="ric")     # thereafter: one small object

:meth:`novetly_score.data.Universe.counts` aggregates 181 million snippet rows across 18,183
parquet files to produce 472 thousand rows. The aggregation runs in the store's engine, so the
text never crosses the network, but the row groups still do: measured at four minutes against an
idle store and twenty-six against a busy one, because it is 18,183 HTTP round trips whose latency
queues. Nothing about the result changes unless the store does, so paying that per notebook run —
or, with a reactive notebook, per edit — is the whole cost of this layer for none of the benefit.

**The panel is derived, so it can go stale, so it carries a stamp.** ``panels/counts.stamp.json``
records the file count, row count and last month the panel was built from, and :func:`read_panel`
compares it against the live manifest before handing the frame over. A store that has grown a
month since the panel was written raises :class:`StalePanelError` rather than quietly answering
with a panel that stops short. The check is one 190 KB read, which is the right price for not
publishing a stale baseline.

What is **not** baked in: the universe mask, the re-dating of news onto trading days, and the
sentiment ratio. Those are the experiment's choices and they cost a second over the materialised
panel (measured: 1.2s for all of them), so they stay where they can be changed — see
:mod:`novetly_score.align` and :func:`novetly_score.panel.sentiment`.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any, Final

from novetly_score.data.duck import BUCKET, OpenDuck
from novetly_score.data.s3 import OpenClient

if TYPE_CHECKING:
    import polars as pl
    from duckdb import DuckDBPyConnection

#: Key the materialised panel is written to.
COUNTS_KEY: Final = "panels/counts.parquet"

#: Key its provenance is written to, beside it.
STAMP_KEY: Final = "panels/counts.stamp.json"

#: Columns of the panel, in order. Keyed by the ``yukka_id`` the store files snippets under,
#: as the price feed is: a caller asking for another identifier gets it relabelled point in time
#: at read time, so one stored panel serves every key.
COLUMNS: Final = ("entity", "day", "positive", "neutral", "negative", "uncharacterised")


class StalePanelError(RuntimeError):
    """The materialised panel was derived from a store that has since changed."""


def stamp(connection: DuckDBPyConnection | None = None) -> dict[str, Any]:
    """What the store holds right now, as a provenance record for a derived panel.

    Read off ``snippets_manifest/file_stats.parquet`` rather than by walking the snippet keys, so
    it costs one small object. Three numbers, because any change to the store moves at least one
    of them: a new month moves all three, a backfill of an existing month moves the row count,
    and a re-written file moves the row count alone.

    Args:
        connection: An open connection, from :class:`~novetly_score.data.duck.OpenDuck`. Left
            ``None`` one is opened for the read and closed after.

    Returns:
        ``{"files": int, "rows": int, "last_month": str}``.
    """
    import polars as pl

    from novetly_score.data.snippets import MANIFEST

    # the path is a module constant, but it goes in as a parameter like every other path this
    # layer reads, so no query is ever assembled by interpolation
    query = "SELECT count(*) AS files, sum(rows) AS rows, max(month) AS last_month FROM read_parquet($manifest)"
    params = {"manifest": MANIFEST}
    if connection is not None:
        row = pl.DataFrame(connection.sql(query, params=params)).row(0, named=True)
    else:
        with OpenDuck() as opened:
            row = pl.DataFrame(opened.sql(query, params=params)).row(0, named=True)
    return {"files": int(row["files"]), "rows": int(row["rows"]), "last_month": str(row["last_month"])}


def write_counts(universe: pl.DataFrame | None = None, *, bucket: str = BUCKET) -> dict[str, Any]:
    """Derive the counts panel and put it in the store, with its stamp beside it.

        >>> write_counts()                                      # doctest: +SKIP

    The expensive call, run deliberately rather than as a side effect of a read: it is the same
    aggregation :func:`novetly_score.panel.counts` makes, so the panel it writes is that function's
    answer and not a second definition of it.

    Args:
        universe: The frame to read and to take RICs from. Defaults to the STOXX 600, which is a
            network call to YUKKA and needs a token.
        bucket: Bucket to write to.

    Returns:
        The stamp written, plus ``rows`` and ``bytes`` of the panel.

    Raises:
        NoSnippetsError: If the store holds no file for any company in the universe.
    """
    from novetly_score.data.snippets import data

    frame = universe
    if frame is None:
        from yukka.data import Index

        frame = Index.STOXX600.frame
    panel = data(frame).derive_counts()
    written = stamp()
    buffer = io.BytesIO()
    panel.write_parquet(buffer, compression="zstd")
    body = buffer.getvalue()
    with OpenClient() as client:
        client.put_object(Bucket=bucket, Key=COUNTS_KEY, Body=body)
        client.put_object(
            Bucket=bucket,
            Key=STAMP_KEY,
            Body=json.dumps(written, sort_keys=True).encode(),
            ContentType="application/json",
        )
    return {**written, "rows": panel.height, "bytes": len(body)}


def read_panel(*, check: bool = True, bucket: str = BUCKET) -> pl.DataFrame:
    """Read the stored counts panel, refusing one the store has outgrown.

    Reached through :meth:`novetly_score.data.Universe.counts`, which narrows it to a span and a
    universe and keys it; this is the whole panel, keyed by ``yukka_id``.

    Args:
        check: Compare the panel's stamp against the live manifest and refuse a mismatch. Turn it
            off only to look at a panel you already know is behind.
        bucket: Bucket to read from.

    Returns:
        :data:`COLUMNS` — the same frame :meth:`Universe.derive_counts` produces, in one read.

    Raises:
        StalePanelError: If *check* is on and the store has changed since the panel was written,
            or if no panel has been written yet.
    """
    import polars as pl

    with OpenClient() as client:
        try:
            body = client.get_object(Bucket=bucket, Key=COUNTS_KEY)["Body"].read()
        except client.exceptions.NoSuchKey as exc:
            msg = f"no panel at s3://{bucket}/{COUNTS_KEY}. Build it with write_counts()"
            raise StalePanelError(msg) from exc
        if check:
            try:
                written = json.loads(client.get_object(Bucket=bucket, Key=STAMP_KEY)["Body"].read())
            except client.exceptions.NoSuchKey as exc:
                msg = f"panel at s3://{bucket}/{COUNTS_KEY} has no stamp, so it cannot be trusted"
                raise StalePanelError(msg) from exc
            _compare(written, stamp())
    return pl.read_parquet(io.BytesIO(body))


def _compare(written: dict[str, Any], current: dict[str, Any]) -> None:
    """Refuse a panel whose stamp disagrees with the store, naming what moved.

    Args:
        written: The stamp the panel was written with.
        current: The store's stamp now, from :func:`stamp`.

    Raises:
        StalePanelError: On any disagreement.
    """
    moved = [
        f"{field}: panel {written.get(field)!r}, store {current[field]!r}"
        for field in ("files", "rows", "last_month")
        if written.get(field) != current[field]
    ]
    if moved:
        msg = "the counts panel is behind the store (" + "; ".join(moved) + "). Rebuild it with write_counts()"
        raise StalePanelError(msg)
