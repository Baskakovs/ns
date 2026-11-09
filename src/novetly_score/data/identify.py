"""Give every snippet row a stable id, by rewriting each snippet file with a ``snippet_id`` column.

    uv run python -m novetly_score.data.identify backfill --entities bmw --months 2025-01:2025-03
    uv run python -m novetly_score.data.identify verify
    uv run python -m novetly_score.data.identify manifest
    uv run python -m novetly_score.data.identify restore --entities bmw      # undo a backfill

The snippet files arrive without a key. The text is not one: 2.68 rows share a text in a
440k-row sample, so a text addresses a *sentence*, not an occurrence of it. This writes a
UUID per row, about 2.9 GB over the whole corpus at 16 bytes each.

**A random id is only as good as its first assignment.** It cannot be recomputed, so the file
*is* the record of it: a second run over a file that already has ids would hand the same rows
new ones and orphan everything joined to the old. So a file that already carries ``snippet_id``
is skipped, always, and no flag overrides that. Re-running is therefore safe and cheap, which is
what makes the backfill resumable across 18,183 files.

**The id is parquet's own UUID type**, 16 bytes, which DuckDB reads back as a ``UUID`` and so
cannot hold a malformed value at all — there is no well-formedness to check, only that the
column really is one. Reading it through :mod:`novetly_score.data.snippets` gives readable text,
because DuckDB converts a UUID on the way out through Arrow; a direct ``pl.read_parquet`` gives
the 16 bytes instead, which ``uuid.UUID(bytes=value)`` turns back into an id. DuckDB does the
writing because polars has no UUID dtype and would emit an untyped blob.

**The original is kept, and ``restore`` puts it back.** Each file is copied to
``Archive/snippets_pre_id/`` — server-side, so no bytes move through here — before the rewritten
one is put over it. Restoring is how to change the way ids are made after some files have them,
since a file that has ids is never rewritten. The rewrite itself is one
``PutObject`` per file, so a reader sees the old file or the new one and never a torn one; a
killed run leaves every file either done or untouched. Pass ``--no-backup`` once you trust it.

**Row order is preserved**, because the manifest records these files as sorted and a reader may
rely on it. polars reads, appends the column and writes; nothing re-sorts and nothing aggregates.

``verify`` is the gate: it checks every file has the column, that no id is null or malformed,
and — the one that matters — that the ids are unique across the whole corpus and not merely
within a file. ``manifest`` rebuilds ``snippets_manifest/file_stats.parquet``, whose ``bytes``
column the rewrite invalidates.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

import polars as pl

from novetly_score.data.duck import BUCKET, OpenDuck
from novetly_score.data.s3 import OpenClient
from novetly_score.data.snippets import SNIPPETS_PREFIX, entities

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import DuckDBPyConnection

#: The column this adds, and the name every join is expected to use.
ID_COLUMN: Final = "snippet_id"

#: Where an untouched original is kept before its file is replaced.
BACKUP_PREFIX: Final = "Archive/snippets_pre_id"

#: The parquet logical type a sound id column carries. DuckDB reads such a column back as its
#: own ``UUID`` type, which is what makes the id well-formed by construction rather than by a
#: length check — there is no way to put a non-UUID in it.
ID_TYPE: Final = "UUIDType()"

#: Where the file manifest lives within a bucket.
MANIFEST_KEY: Final = "snippets_manifest/file_stats.parquet"


def manifest_path(bucket: str = BUCKET) -> str:
    """Name the file manifest of one bucket.

    The manifest belongs to the bucket it describes, so every command here takes a bucket and
    derives the manifest from it rather than reaching for one fixed path.

    Args:
        bucket: The bucket.

    Returns:
        An ``s3://`` URL.
    """
    return f"s3://{bucket}/{MANIFEST_KEY}"


_SQL_FILES: Final = """
    SELECT entity, month, rows FROM read_parquet($manifest)
    WHERE ($entities IS NULL OR list_contains($entities, entity))
      AND ($first IS NULL OR month >= $first) AND ($last IS NULL OR month <= $last)
    ORDER BY entity, month
"""

_SQL_SCHEMA: Final = "SELECT name FROM parquet_schema($path)"

# union_by_name so a file that has not been backfilled yet reads its missing column as NULL
# rather than failing to bind: "verify before backfill" is the first thing anyone runs, and it
# should come back as a count of rows without ids, not as a binder error.
# Each counter means one thing: a null is counted as a row without an id and not also as a
# collision, so the message can name the actual cause rather than two at once.
_SQL_ID_HEALTH: Final = """
    SELECT count(*) AS rows,
           count(*) FILTER (snippet_id IS NOT NULL) AS with_id,
           count(DISTINCT snippet_id) AS ids,
           count(*) FILTER (snippet_id IS NULL) AS null_ids
    FROM read_parquet($paths, union_by_name=true)
"""

# A file whose id column is a string or a blob rather than a UUID would still join, since
# DuckDB casts across them, and would be invisible to every row-level check. The logical type
# is the only place that drift shows.
_SQL_ID_TYPE: Final = """
    SELECT count(*) FILTER (logical_type <> $expected) AS wrong_type, count(*) AS files
    FROM parquet_schema($paths) WHERE name = $column
"""

_SQL_ADD_IDS: Final = "SELECT uuid() AS snippet_id, * FROM read_parquet($source)"

_SQL_COUNT: Final = "SELECT count(*) FROM read_parquet($path)"

_SQL_REBUILD_MANIFEST: Final = """
    SELECT regexp_extract(filename, '{prefix}/([^/]+)/', 1) AS entity,
           regexp_extract(filename, '([0-9]{{4}}-[0-9]{{2}})\\.parquet', 1) AS month,
           count(*) AS rows, count(DISTINCT article_id) AS articles
    FROM read_parquet($paths, filename=true) GROUP BY 1, 2
"""


class BackfillError(RuntimeError):
    """The backfill cannot proceed, or the corpus did not come out sound."""


def key_of(entity: str, month: str) -> str:
    """Name one snippet file's object key.

    Args:
        entity: Snippet directory name.
        month: ``"YYYY-MM"``.

    Returns:
        The object key.
    """
    return f"{SNIPPETS_PREFIX}/{entity}/{month}.parquet"


def files(
    names: Sequence[str] | None,
    first: str | None,
    last: str | None,
    bucket: str = BUCKET,
) -> list[tuple[str, str, int]]:
    """List the snippet files a run covers, from the manifest rather than a listing.

    Empty files are included: a file with no rows still needs the column, or a reader that
    selects it across the corpus fails on the one file that lacks it.

    Args:
        names: Snippet directory names, or ``None`` for every company.
        first: Earliest month, or ``None``.
        last: Latest month, or ``None``.
        bucket: The bucket whose manifest to read.

    Returns:
        ``(entity, month, rows)`` triples, in entity then month order.

    Raises:
        BackfillError: If the selection names no file.
    """
    with OpenDuck(bucket=bucket) as con:
        rows = con.sql(
            _SQL_FILES,
            params={
                "manifest": manifest_path(bucket),
                "entities": list(names) if names else None,
                "first": first,
                "last": last,
            },
        ).fetchall()
    if not rows:
        msg = "that selection of companies and months holds no snippet file"
        raise BackfillError(msg)
    return [(entity, month, count) for entity, month, count in rows]


def add_ids(con: DuckDBPyConnection, source: Path, target: Path) -> int:
    """Write *source* to *target* with a UUID column in front, row order intact.

    DuckDB does the writing, not polars, because only it emits parquet's ``UUID`` logical type
    — polars has no UUID dtype and would write 16 opaque bytes that come back as a ``BLOB``.
    The relation's own ``write_parquet`` takes the path as an argument, so no file path is ever
    formatted into SQL.

    Row order survives: ``preserve_insertion_order`` is on by default, and this is checked over
    500,000 rows by a test, because the manifest records these files as sorted and a reader may
    rely on it.

    Args:
        con: An open connection.
        source: The parquet to read.
        target: The parquet to write.

    Returns:
        The rows written.
    """
    relation = con.sql(_SQL_ADD_IDS, params={"source": str(source)})
    relation.write_parquet(str(target), compression="zstd")
    return con.sql(_SQL_COUNT, params={"path": str(target)}).fetchall()[0][0]


def backfill(
    names: Sequence[str] | None = None,
    first: str | None = None,
    last: str | None = None,
    *,
    backup: bool = True,
    bucket: str = BUCKET,
    dry_run: bool = False,
) -> dict[str, int]:
    """Add the id column to every selected snippet file that has not got one.

    Args:
        names: Snippet directory names, or ``None`` for every company.
        first: Earliest month, or ``None``.
        last: Latest month, or ``None``.
        backup: Copy each original under :data:`BACKUP_PREFIX` before replacing it.
        bucket: The bucket.
        dry_run: Report what would be rewritten and change nothing.

    Returns:
        ``{"files": …, "rewritten": …, "skipped": …, "rows": …, "bytes": …}``.
    """
    selection = files(names, first, last, bucket)
    counts = {"files": len(selection), "rewritten": 0, "skipped": 0, "rows": 0, "bytes": 0}
    print(f"{len(selection):,} snippet files selected", flush=True)
    with OpenDuck(bucket=bucket) as con, OpenClient() as s3:
        for index, (entity, month, expected) in enumerate(selection, start=1):
            key = key_of(entity, month)
            if _has_id(con, bucket, key):
                counts["skipped"] += 1
                continue
            if dry_run:
                counts["rewritten"] += 1
                counts["rows"] += expected
                continue
            written, rows = _rewrite(con, s3, bucket, key, backup=backup)
            counts["rewritten"] += 1
            counts["rows"] += rows
            counts["bytes"] += written
            if counts["rewritten"] % 50 == 0 or index == len(selection):
                print(
                    f"  {index:,}/{len(selection):,}: {counts['rewritten']:,} rewritten, "
                    f"{counts['skipped']:,} already had ids, {counts['rows']:,} rows",
                    flush=True,
                )
    verb = "would rewrite" if dry_run else "rewrote"
    print(
        f"{verb} {counts['rewritten']:,} files ({counts['rows']:,} rows); {counts['skipped']:,} already carried ids",
        flush=True,
    )
    return counts


def _has_id(con: DuckDBPyConnection, bucket: str, key: str) -> bool:
    """Say whether a file already carries the id column, by reading its schema.

    The schema and not the bytes: a snippet whose own text contains "snippet_id" would fool a
    search of the footer, and the cost of that mistake is a file that is skipped for ever and
    never gets ids. Reading ``parquet_schema`` costs the footer, which is a range read, so this
    is no more expensive than the mistake would have been.

    Args:
        con: An open connection.
        bucket: The bucket.
        key: The object key.

    Returns:
        ``True`` if :data:`ID_COLUMN` is one of the file's columns.

    Raises:
        BackfillError: If the file cannot be read at all.
    """
    import duckdb

    try:
        columns = con.sql(_SQL_SCHEMA, params={"path": f"s3://{bucket}/{key}"}).fetchall()
    except duckdb.Error as exc:
        msg = f"cannot read the schema of s3://{bucket}/{key}: {exc}"
        raise BackfillError(msg) from exc
    return any(name == ID_COLUMN for (name,) in columns)


def _rewrite(
    con: DuckDBPyConnection,
    s3,  # noqa: ANN001 — a boto3 client; see OpenClient
    bucket: str,
    key: str,
    *,
    backup: bool,
) -> tuple[int, int]:
    """Rewrite one file with ids, keeping the original first.

    Args:
        con: An open connection, which does the writing.
        s3: An open S3 client.
        bucket: The bucket.
        key: The object key.
        backup: Copy the original under :data:`BACKUP_PREFIX` first.

    Returns:
        ``(bytes_written, rows)``.
    """
    if backup:
        spare = f"{BACKUP_PREFIX}/{key.removeprefix(f'{SNIPPETS_PREFIX}/')}"
        if not _exists(s3, bucket, spare):
            s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": key}, Key=spare)
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "in.parquet"
        target = Path(directory) / "out.parquet"
        s3.download_file(bucket, key, str(source))
        rows = add_ids(con, source, target)
        size = target.stat().st_size
        s3.upload_file(str(target), bucket, key)
    return size, rows


def _exists(s3, bucket: str, key: str) -> bool:  # noqa: ANN001 — boto3 client
    """Say whether an object is already there.

    Args:
        s3: An open S3 client.
        bucket: The bucket.
        key: The object key.

    Returns:
        ``True`` if the object exists.
    """
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return False
    return True


def verify(bucket: str = BUCKET) -> dict[str, int]:
    """Prove every snippet row has one sound id, unique across the corpus.

    Uniqueness is checked over every file at once, not file by file: ids drawn independently
    per file are only useful if they do not collide between files, and that is the claim a
    per-file check cannot make. There is no well-formedness check, because a UUID column cannot
    hold anything but a UUID; what replaces it is a check that the column really is one, since
    DuckDB casts a string or a blob to a UUID silently when joining and such a file would
    otherwise pass every row-level test.

    Args:
        bucket: The bucket.

    Returns:
        ``{"rows": …, "ids": …, "collisions": …, "null_ids": …, "wrong_type": …}``. A row in a
        file that has not been backfilled counts under ``null_ids``; ``wrong_type`` counts files
        whose id column is a string or a blob rather than a UUID.

    Raises:
        BackfillError: If any row lacks a sound, unique id.
    """
    paths = f"s3://{bucket}/{SNIPPETS_PREFIX}/*/*.parquet"
    with OpenDuck(bucket=bucket) as con:
        # union_by_name can only union a column that at least one file has, so the
        # nothing-migrated-yet case has to be answered before the health query, not by it.
        columns = {name for (name,) in con.sql(_SQL_SCHEMA, params={"path": paths}).fetchall()}
        if ID_COLUMN not in columns:
            msg = f"no snippet file in s3://{bucket} carries {ID_COLUMN} — run the backfill"
            raise BackfillError(msg)
        wrong_type, _files = con.sql(
            _SQL_ID_TYPE, params={"paths": paths, "expected": ID_TYPE, "column": ID_COLUMN}
        ).fetchall()[0]
        if wrong_type:
            # Stop here rather than counting rows: with a mixed id type the corpus cannot be
            # read as one table at all, so there are no row counts to report.
            msg = (
                f"{wrong_type:,} file(s) hold {ID_COLUMN} as something other than a {ID_TYPE}. "
                f"Restore them from {BACKUP_PREFIX} and run the backfill again"
            )
            raise BackfillError(msg)
        rows, with_id, ids, nulls = con.sql(_SQL_ID_HEALTH, params={"paths": paths}).fetchall()[0]
    report = {
        "rows": rows,
        "ids": ids,
        "collisions": with_id - ids,
        "null_ids": nulls,
        "wrong_type": wrong_type,
    }
    for name, value in report.items():
        print(f"  {name:<12} {value:,}", flush=True)
    if report["collisions"] or nulls:
        cause = (
            "run the backfill — those rows are in files that have not been rewritten yet"
            if nulls and not report["collisions"] and not wrong_type
            else f"a file rewritten twice is the likely cause; the originals are under {BACKUP_PREFIX}"
        )
        msg = f"unsound ids: {report['collisions']:,} collisions, {nulls:,} without an id. {cause}"
        raise BackfillError(msg)
    print(f"sound: {rows:,} rows, {ids:,} distinct ids, one per row", flush=True)
    return report


def restore(
    names: Sequence[str] | None = None,
    first: str | None = None,
    last: str | None = None,
    *,
    bucket: str = BUCKET,
    dry_run: bool = False,
) -> dict[str, int]:
    """Put the kept originals back over the live files, undoing a backfill.

    The copy is server-side, so undoing costs no bandwidth. Afterwards the files carry no id
    column again and :func:`backfill` will key them afresh — which is the only way to change
    how ids are made once some files already have them, since a file that has ids is skipped.

    Args:
        names: Snippet directory names, or ``None`` for every company that has a backup.
        first: Earliest month, or ``None``.
        last: Latest month, or ``None``.
        bucket: The bucket.
        dry_run: Report what would be restored and change nothing.

    Returns:
        ``{"restored": …, "missing": …}`` — files put back, and files with no backup to put back.
    """
    counts = {"restored": 0, "missing": 0}
    wanted = {(entity, month) for entity, month, _ in files(names, first, last, bucket)}
    with OpenClient() as s3:
        kept: dict[tuple[str, str], str] = {}
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{BACKUP_PREFIX}/"):
            for item in page.get("Contents", []):
                parts = item["Key"].removeprefix(f"{BACKUP_PREFIX}/").split("/")
                if len(parts) == 2 and parts[1].endswith(".parquet"):
                    kept[(parts[0], parts[1].removesuffix(".parquet"))] = item["Key"]
        for entity, month in sorted(wanted):
            spare = kept.get((entity, month))
            if spare is None:
                counts["missing"] += 1
                continue
            if not dry_run:
                s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": spare}, Key=key_of(entity, month))
            counts["restored"] += 1
    verb = "would restore" if dry_run else "restored"
    print(
        f"{verb} {counts['restored']:,} files from {BACKUP_PREFIX}; {counts['missing']:,} had no backup",
        flush=True,
    )
    return counts


def manifest(bucket: str = BUCKET) -> int:
    """Rebuild the file manifest, whose byte counts the rewrite invalidated.

    Args:
        bucket: The bucket.

    Returns:
        The number of files the rebuilt manifest describes.
    """
    paths = f"s3://{bucket}/{SNIPPETS_PREFIX}/*/*.parquet"
    sql = _SQL_REBUILD_MANIFEST.format(prefix=SNIPPETS_PREFIX)
    with OpenDuck(bucket=bucket) as con:
        counted = pl.DataFrame(con.sql(sql, params={"paths": paths}))
    sizes: dict[tuple[str, str], int] = {}
    with OpenClient() as s3:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{SNIPPETS_PREFIX}/"):
            for item in page.get("Contents", []):
                parts = item["Key"].split("/")
                if len(parts) == 3 and parts[2].endswith(".parquet"):
                    sizes[(parts[1], parts[2].removesuffix(".parquet"))] = item["Size"]
    frame = (
        pl.DataFrame(
            [{"entity": entity, "month": month, "bytes": size} for (entity, month), size in sorted(sizes.items())]
        )
        .join(counted, on=["entity", "month"], how="left")
        .with_columns(
            rows=pl.col("rows").fill_null(0),
            articles=pl.col("articles").fill_null(0),
            sorted=pl.lit(True),
            schema_ok=pl.lit(True),
        )
        .select("entity", "month", "rows", "articles", "bytes", "sorted", "schema_ok")
        .sort("entity", "month")
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "file_stats.parquet"
        frame.write_parquet(path, compression="zstd")
        with OpenClient() as s3:
            s3.upload_file(str(path), bucket, MANIFEST_KEY)
    print(f"rebuilt the manifest: {frame.height:,} files, {frame.get_column('rows').sum():,} rows", flush=True)
    return frame.height


def _months(span: str | None) -> tuple[str | None, str | None]:
    """Split a ``FIRST:LAST`` month span.

    Args:
        span: ``"2025-01:2025-03"``, ``"2025-01"``, ``"2025-01:"``, or ``None``.

    Returns:
        ``(first, last)``.
    """
    if not span:
        return None, None
    first, separator, last = span.partition(":")
    return (first.strip() or None), ((last.strip() or None) if separator else first.strip() or None)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command from the command line.

    Args:
        argv: Arguments, or ``None`` for ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` if the corpus is not sound.
    """
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)

    parser = argparse.ArgumentParser(prog="novetly_score.data.identify", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bucket",
        default=BUCKET,
        help=f"the bucket to work on (default {BUCKET}); point it at a copy to rehearse",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fill = commands.add_parser("backfill", help="add the id column to files that have not got one")
    fill.add_argument("--entities", help="comma-separated company names (default: every company)")
    fill.add_argument("--months", help="FIRST:LAST, e.g. 2025-01:2025-03 (default: every month)")
    fill.add_argument("--no-backup", action="store_true", help="do not keep the originals")
    fill.add_argument("--dry-run", action="store_true", help="report what would be rewritten and stop")
    commands.add_parser("verify", help="prove every row has one sound, corpus-unique id")
    commands.add_parser("manifest", help="rebuild file_stats.parquet after a rewrite")
    back = commands.add_parser("restore", help="put the kept originals back, undoing a backfill")
    back.add_argument("--entities", help="comma-separated company names (default: every company)")
    back.add_argument("--months", help="FIRST:LAST, e.g. 2025-01:2025-03 (default: every month)")
    back.add_argument("--dry-run", action="store_true", help="report what would be restored and stop")

    args = parser.parse_args(argv)
    try:
        if args.command == "backfill":
            first, last = _months(args.months)
            names = [item.strip() for item in args.entities.split(",") if item.strip()] if args.entities else None
            backfill(
                list(entities(names)) if names else None,
                first,
                last,
                backup=not args.no_backup,
                bucket=args.bucket,
                dry_run=args.dry_run,
            )
        elif args.command == "restore":
            first, last = _months(args.months)
            names = [item.strip() for item in args.entities.split(",") if item.strip()] if args.entities else None
            restore(
                list(entities(names)) if names else None,
                first,
                last,
                bucket=args.bucket,
                dry_run=args.dry_run,
            )
        elif args.command == "verify":
            verify(bucket=args.bucket)
        else:
            manifest(bucket=args.bucket)
    except BackfillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover — the command line
    raise SystemExit(main())
