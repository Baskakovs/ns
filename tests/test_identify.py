"""Tests for the id backfill: that ids appear once, never twice, and that verify catches it.

The corpus lives in its own bucket, because ``verify`` and ``manifest`` are deliberately
corpus-wide — they glob every snippet file there is — and moto's backend is global to the
process, so sharing a bucket with another test module would mean checking its files too.
"""

from __future__ import annotations

import boto3
import polars as pl
import pytest
from moto.server import ThreadedMotoServer

from novetly_score.data.identify import (
    BACKUP_PREFIX,
    ID_COLUMN,
    ID_TYPE,
    MANIFEST_KEY,
    BackfillError,
    add_ids,
    backfill,
    files,
    key_of,
    main,
    manifest,
    restore,
    verify,
)
from novetly_score.data.s3 import ACCESS_KEY_VAR, ENDPOINT_VAR, SECRET_VAR

#: A bucket of this module's own.
BUCKET = "migrate-experiments"

#: The corpus: (entity, month, snippet texts). One file is empty, one text repeats, and one
#: snippet's own text contains the id column's name — the thing that once fooled the skip check.
CORPUS = (
    ("mig_one", "2025-01", ["alpha", "beta", "alpha"]),
    ("mig_one", "2025-02", ["gamma mentions snippet_id in its text"]),
    ("mig_two", "2025-01", ["delta", "epsilon"]),
    ("mig_two", "2025-02", []),
)

#: Rows across the whole corpus.
ROWS = sum(len(texts) for _, _, texts in CORPUS)


@pytest.fixture(scope="session")
def server():
    """A local S3-compatible store, as an endpoint URL."""
    moto = ThreadedMotoServer(port=0, verbose=False)
    moto.start()
    host, port = moto.get_host_and_port()
    yield f"http://{host}:{port}"
    moto.stop()


@pytest.fixture
def env(monkeypatch, tmp_path, server):
    """An environment pointing the commands at the store, with no .env within reach."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENDPOINT_VAR, server)
    monkeypatch.setenv(ACCESS_KEY_VAR, "key")
    monkeypatch.setenv(SECRET_VAR, "secret")
    return monkeypatch


@pytest.fixture
def corpus(env, tmp_path):
    """A snippet corpus with no id column, rebuilt from nothing, as the real one is pre-backfill.

    Rebuilt per test rather than per session: the point of several of these tests is what a
    second run does to ids a first run assigned, so one test must not inherit another's.
    """
    import contextlib
    import os

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ[ENDPOINT_VAR],
        aws_access_key_id="key",
        aws_secret_access_key="secret",  # noqa: S106 — the mock store's fixed credentials
        region_name="us-east-1",
    )
    with contextlib.suppress(s3.exceptions.BucketAlreadyOwnedByYou):
        s3.create_bucket(Bucket=BUCKET)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        for item in page.get("Contents", []):
            s3.delete_object(Bucket=BUCKET, Key=item["Key"])

    for entity, month, texts in CORPUS:
        path = tmp_path / f"{entity}-{month}.parquet"
        pl.DataFrame(
            {
                "entity": [f"company:{entity}"] * len(texts),
                "sentiment": ["pos"] * len(texts),
                "snippet": texts,
                "article_id": [f"{entity}-{month}-{index}" for index in range(len(texts))],
                "lang": ["en"] * len(texts),
            },
            schema={
                "entity": pl.String,
                "sentiment": pl.String,
                "snippet": pl.String,
                "article_id": pl.String,
                "lang": pl.String,
            },
        ).write_parquet(path)
        s3.upload_file(str(path), BUCKET, key_of(entity, month))

    stats = tmp_path / "file_stats.parquet"
    pl.DataFrame(
        {
            "entity": [entity for entity, _, _ in CORPUS],
            "month": [month for _, month, _ in CORPUS],
            "rows": [len(texts) for _, _, texts in CORPUS],
            "articles": [len(texts) for _, _, texts in CORPUS],
            "bytes": [1000] * len(CORPUS),
            "sorted": [True] * len(CORPUS),
            "schema_ok": [True] * len(CORPUS),
        }
    ).write_parquet(stats)
    s3.upload_file(str(stats), BUCKET, MANIFEST_KEY)
    return s3


def read(s3, key, tmp_path):
    """Read one object back as a frame."""
    local = tmp_path / "read.parquet"
    s3.download_file(BUCKET, key, str(local))
    return pl.read_parquet(local)


class TestOneIdPerRow:
    """What the column holds once a file has been rewritten, checked on local files only."""

    @pytest.fixture
    def con(self):
        """A plain in-memory connection: adding ids touches no network."""
        import duckdb

        with duckdb.connect() as connection:
            yield connection

    def written(self, con, tmp_path, frame):
        """Run a frame through add_ids and read the result back with DuckDB."""
        source, target = tmp_path / "in.parquet", tmp_path / "out.parquet"
        frame.write_parquet(source)
        rows = add_ids(con, source, target)
        return rows, target

    def test_it_puts_a_unique_id_on_every_row(self, con, tmp_path):
        """One id per row, and no two the same — identical text included."""
        rows, target = self.written(con, tmp_path, pl.DataFrame({"snippet": ["a", "b", "a"]}))
        assert rows == 3
        counted = con.sql(
            "SELECT count(*), count(DISTINCT snippet_id) FROM read_parquet($p)", params={"p": str(target)}
        ).fetchall()[0]
        assert counted == (3, 3)

    def test_the_id_is_a_native_uuid(self, con, tmp_path):
        """The point of the type: a UUID column cannot hold a malformed value at all."""
        _rows, target = self.written(con, tmp_path, pl.DataFrame({"snippet": ["a"]}))
        kind = con.sql(
            "SELECT logical_type FROM parquet_schema($p) WHERE name = $c",
            params={"p": str(target), "c": ID_COLUMN},
        ).fetchall()
        assert kind == [(ID_TYPE,)]

    def test_polars_can_recover_the_id(self, con, tmp_path):
        """The 16 bytes polars sees must convert back to the id DuckDB wrote, faithfully."""
        import uuid

        _rows, target = self.written(con, tmp_path, pl.DataFrame({"snippet": ["a"]}))
        raw = pl.read_parquet(target).get_column(ID_COLUMN).item()
        assert isinstance(raw, bytes)
        (want,) = con.sql("SELECT snippet_id::VARCHAR FROM read_parquet($p)", params={"p": str(target)}).fetchall()[0]
        assert str(uuid.UUID(bytes=raw)) == want

    def test_the_id_comes_first(self, con, tmp_path):
        """The key reads better at the front, and a reader may select by position."""
        _rows, target = self.written(con, tmp_path, pl.DataFrame({"entity": ["x"], "snippet": ["a"]}))
        assert pl.read_parquet(target).columns == [ID_COLUMN, "entity", "snippet"]

    def test_the_rows_keep_their_order(self, con, tmp_path):
        """The manifest records these files as sorted, so a reader may rely on it."""
        size = 500_000  # large enough that a parallel scan would show up as a reordering
        frame = pl.DataFrame({"seq": list(range(size))})
        _rows, target = self.written(con, tmp_path, frame)
        assert pl.read_parquet(target).get_column("seq").to_list() == list(range(size))

    def test_an_empty_file_gets_the_column(self, con, tmp_path):
        """A file with no rows still needs the column, or a corpus-wide select fails on it."""
        rows, target = self.written(con, tmp_path, pl.DataFrame({"snippet": []}, schema={"snippet": pl.String}))
        assert rows == 0
        assert pl.read_parquet(target).columns == [ID_COLUMN, "snippet"]


class TestTheBackfill:
    """What rewriting a corpus does, and does not, change."""

    def test_every_file_gains_the_column(self, corpus, tmp_path):
        """Including the empty one."""
        counts = backfill(bucket=BUCKET)
        assert counts["rewritten"] == len(CORPUS)
        assert counts["rows"] == ROWS
        for entity, month, _ in CORPUS:
            assert ID_COLUMN in read(corpus, key_of(entity, month), tmp_path).columns

    def test_the_other_columns_are_untouched(self, corpus, tmp_path):
        """A migration that edits the data is not a migration."""
        before = read(corpus, key_of("mig_one", "2025-01"), tmp_path)
        backfill(bucket=BUCKET)
        after = read(corpus, key_of("mig_one", "2025-01"), tmp_path)
        assert after.drop(ID_COLUMN).equals(before)

    def test_ids_do_not_collide_between_files(self, corpus, tmp_path):
        """Each file is rewritten alone, so the interesting claim is the one across files."""
        backfill(bucket=BUCKET)
        seen = [
            value
            for entity, month, _ in CORPUS
            for value in read(corpus, key_of(entity, month), tmp_path).get_column(ID_COLUMN)
        ]
        assert len(seen) == ROWS
        assert len(set(seen)) == ROWS

    def test_a_selection_narrows_the_rewrite(self, corpus, tmp_path):
        """A corpus of 18,183 files has to be migratable in slices."""
        backfill(["mig_one"], "2025-01", "2025-01", bucket=BUCKET)
        assert ID_COLUMN in read(corpus, key_of("mig_one", "2025-01"), tmp_path).columns
        assert ID_COLUMN not in read(corpus, key_of("mig_two", "2025-01"), tmp_path).columns

    def test_a_dry_run_changes_nothing(self, corpus, tmp_path):
        """The count has to be available before 18,183 files are rewritten."""
        counts = backfill(bucket=BUCKET, dry_run=True)
        assert counts["rewritten"] == len(CORPUS)
        assert ID_COLUMN not in read(corpus, key_of("mig_one", "2025-01"), tmp_path).columns

    def test_an_empty_selection_is_refused(self, corpus):
        """Rewriting nothing silently would look like a finished migration."""
        with pytest.raises(BackfillError, match="holds no snippet file"):
            backfill(["mig_one"], "2030-01", "2030-01", bucket=BUCKET)


class TestIdsAreNeverReassigned:
    """The invariant the whole design rests on."""

    def test_a_second_run_rewrites_nothing(self, corpus, tmp_path):
        """A random id cannot be recomputed, so reassigning one orphans every join to it."""
        backfill(bucket=BUCKET)
        first = read(corpus, key_of("mig_one", "2025-01"), tmp_path).get_column(ID_COLUMN).to_list()
        counts = backfill(bucket=BUCKET)
        assert counts["rewritten"] == 0
        assert counts["skipped"] == len(CORPUS)
        assert read(corpus, key_of("mig_one", "2025-01"), tmp_path).get_column(ID_COLUMN).to_list() == first

    def test_a_resumed_run_finishes_the_rest(self, corpus, tmp_path):
        """A killed run must be finishable without touching what it already did."""
        backfill(["mig_one"], bucket=BUCKET)
        kept = read(corpus, key_of("mig_one", "2025-01"), tmp_path).get_column(ID_COLUMN).to_list()
        counts = backfill(bucket=BUCKET)
        assert counts["rewritten"] == 2
        assert counts["skipped"] == 2
        assert read(corpus, key_of("mig_one", "2025-01"), tmp_path).get_column(ID_COLUMN).to_list() == kept

    def test_a_snippet_naming_the_column_does_not_fool_the_check(self, corpus, tmp_path):
        """The check reads the schema: a footer search once skipped such a file for ever."""
        backfill(bucket=BUCKET)
        frame = read(corpus, key_of("mig_one", "2025-02"), tmp_path)
        assert "snippet_id" in frame.get_column("snippet").item()
        assert frame.get_column(ID_COLUMN).item() is not None


class TestTheOriginalIsKept:
    """What is recoverable after an in-place rewrite of 18,183 files."""

    def test_the_original_is_copied_aside(self, corpus, tmp_path):
        """The rewrite is irreversible; the copy is what makes it not."""
        before = read(corpus, key_of("mig_one", "2025-01"), tmp_path)
        backfill(bucket=BUCKET)
        spare = f"{BACKUP_PREFIX}/mig_one/2025-01.parquet"
        assert read(corpus, spare, tmp_path).equals(before)

    def test_the_backup_can_be_declined(self, corpus):
        """Once trusted, 6.65 GB of duplicate corpus is not worth keeping."""
        backfill(bucket=BUCKET, backup=False)
        listing = corpus.list_objects_v2(Bucket=BUCKET, Prefix=BACKUP_PREFIX)
        assert listing.get("KeyCount", 0) == 0

    def test_a_backup_is_not_overwritten(self, corpus, tmp_path):
        """A rerun must not replace a good original with an already-migrated file."""
        backfill(["mig_one"], "2025-01", "2025-01", bucket=BUCKET)
        spare = f"{BACKUP_PREFIX}/mig_one/2025-01.parquet"
        kept = read(corpus, spare, tmp_path)
        backfill(["mig_one"], "2025-01", "2025-01", bucket=BUCKET)
        assert read(corpus, spare, tmp_path).equals(kept)
        assert ID_COLUMN not in kept.columns


class TestRestore:
    """How a backfill is undone, which is the only way to change how ids are made."""

    def test_it_puts_the_original_back(self, corpus, tmp_path):
        """And the restored file has no id column, so the backfill will key it afresh."""
        backfill(bucket=BUCKET)
        counts = restore(bucket=BUCKET)
        assert counts["restored"] == len(CORPUS)
        assert ID_COLUMN not in read(corpus, key_of("mig_one", "2025-01"), tmp_path).columns

    def test_a_restored_file_is_keyed_again(self, corpus, tmp_path):
        """Restore then backfill is the migration path when the id type changes."""
        backfill(bucket=BUCKET)
        before = read(corpus, key_of("mig_one", "2025-01"), tmp_path).get_column(ID_COLUMN).to_list()
        restore(bucket=BUCKET)
        backfill(bucket=BUCKET)
        after = read(corpus, key_of("mig_one", "2025-01"), tmp_path).get_column(ID_COLUMN).to_list()
        assert len(after) == len(before)
        assert set(after).isdisjoint(before), "a re-key must produce new ids, not the old ones"

    def test_a_dry_run_restores_nothing(self, corpus, tmp_path):
        """Undoing a migration of 18,183 files deserves a rehearsal."""
        backfill(bucket=BUCKET)
        counts = restore(bucket=BUCKET, dry_run=True)
        assert counts["restored"] == len(CORPUS)
        assert ID_COLUMN in read(corpus, key_of("mig_one", "2025-01"), tmp_path).columns

    def test_a_file_with_no_backup_is_reported(self, corpus):
        """--no-backup leaves nothing to restore, and that has to be said rather than assumed."""
        backfill(bucket=BUCKET, backup=False)
        counts = restore(bucket=BUCKET)
        assert counts["restored"] == 0
        assert counts["missing"] == len(CORPUS)


class TestVerify:
    """What counts as a sound corpus."""

    def test_a_migrated_corpus_verifies(self, corpus):
        """The happy path has to be reachable."""
        backfill(bucket=BUCKET)
        report = verify(bucket=BUCKET)
        assert report["rows"] == ROWS
        assert report["ids"] == ROWS
        assert report["collisions"] == 0

    def test_a_collision_is_caught(self, corpus, tmp_path):
        """Two rows with one id is the failure a per-file check cannot see."""
        backfill(bucket=BUCKET)
        key = key_of("mig_one", "2025-01")
        frame = read(corpus, key, tmp_path)
        clashing = frame.with_columns(pl.lit(frame.get_column(ID_COLUMN).first()).alias(ID_COLUMN))
        local = tmp_path / "clash.parquet"
        clashing.write_parquet(local)
        corpus.upload_file(str(local), BUCKET, key)
        with pytest.raises(BackfillError, match="collisions"):
            verify(bucket=BUCKET)

    def test_a_null_id_is_caught(self, corpus, tmp_path):
        """A null id is a row that cannot be addressed at all.

        Planted with DuckDB rather than polars so the column stays a UUID: a polars column of
        nulls would be a string, and the type check would fire first and mask this one.
        """
        import duckdb

        backfill(bucket=BUCKET)
        key = key_of("mig_two", "2025-01")
        source, target = tmp_path / "was.parquet", tmp_path / "null.parquet"
        corpus.download_file(BUCKET, key, str(source))
        with duckdb.connect() as con:
            con.sql(
                "SELECT NULL::UUID AS snippet_id, * EXCLUDE (snippet_id) FROM read_parquet($p)",
                params={"p": str(source)},
            ).write_parquet(str(target))
        corpus.upload_file(str(target), BUCKET, key)
        with pytest.raises(BackfillError, match="without an id"):
            verify(bucket=BUCKET)

    def test_an_id_column_of_the_wrong_type_is_caught(self, corpus, tmp_path):
        """DuckDB casts a string to a UUID silently when joining, so only the type shows it."""
        backfill(bucket=BUCKET)
        key = key_of("mig_two", "2025-01")
        frame = read(corpus, key, tmp_path).with_columns(pl.lit("not-a-uuid").alias(ID_COLUMN))
        local = tmp_path / "bad.parquet"
        frame.write_parquet(local)
        corpus.upload_file(str(local), BUCKET, key)
        with pytest.raises(BackfillError, match="other than a"):
            verify(bucket=BUCKET)


class TestTheManifest:
    """What the rewrite invalidates, and what rebuilding it restores."""

    def test_it_records_every_file(self, corpus):
        """The file list is what every other stage reads; it has to be complete."""
        backfill(bucket=BUCKET)
        assert manifest(bucket=BUCKET) == len(CORPUS)

    def test_the_byte_counts_follow_the_rewrite(self, corpus, tmp_path):
        """The old manifest's bytes are stale the moment a file is rewritten."""
        backfill(bucket=BUCKET)
        manifest(bucket=BUCKET)
        rebuilt = read(corpus, MANIFEST_KEY, tmp_path)
        listing = corpus.head_object(Bucket=BUCKET, Key=key_of("mig_one", "2025-01"))
        row = rebuilt.filter((pl.col("entity") == "mig_one") & (pl.col("month") == "2025-01"))
        assert row.get_column("bytes").item() == listing["ContentLength"]
        assert row.get_column("rows").item() == 3

    def test_an_empty_file_keeps_a_row_with_no_rows(self, corpus, tmp_path):
        """It exists, so it belongs in the file list with a count of zero."""
        backfill(bucket=BUCKET)
        manifest(bucket=BUCKET)
        rebuilt = read(corpus, MANIFEST_KEY, tmp_path)
        row = rebuilt.filter((pl.col("entity") == "mig_two") & (pl.col("month") == "2025-02"))
        assert row.height == 1
        assert row.get_column("rows").item() == 0

    def test_the_file_list_comes_from_the_manifest(self, corpus):
        """And the manifest it reads is the one in the bucket it was asked about."""
        assert [(entity, month) for entity, month, _ in files(None, None, None, BUCKET)] == [
            (entity, month) for entity, month, _ in CORPUS
        ]


class TestTheCommandLine:
    """That the flags reach the commands."""

    def test_dry_run_reaches_the_command(self, corpus, tmp_path):
        """--dry-run must change nothing; the sibling script once parsed it and ignored it."""
        assert main(["--bucket", BUCKET, "backfill", "--entities", "mig_one", "--dry-run"]) == 0
        assert ID_COLUMN not in read(corpus, key_of("mig_one", "2025-01"), tmp_path).columns

    def test_the_backfill_reaches_the_command(self, corpus, tmp_path):
        """And without --dry-run it must actually rewrite."""
        assert main(["--bucket", BUCKET, "backfill", "--entities", "mig_one"]) == 0
        assert ID_COLUMN in read(corpus, key_of("mig_one", "2025-01"), tmp_path).columns

    def test_a_sound_corpus_exits_zero(self, corpus):
        """The exit code is what a pipeline reads."""
        backfill(bucket=BUCKET)
        assert main(["--bucket", BUCKET, "verify"]) == 0

    def test_an_unmigrated_corpus_exits_nonzero(self, corpus):
        """Verifying before the backfill must fail, and say so rather than raise a binder error."""
        assert main(["--bucket", BUCKET, "verify"]) == 1

    def test_an_unmigrated_corpus_says_run_the_backfill(self, corpus):
        """The first thing anyone runs is verify, before the backfill. It must say so plainly."""
        with pytest.raises(BackfillError, match="run the backfill"):
            verify(bucket=BUCKET)

    def test_a_partly_migrated_corpus_counts_the_rest(self, corpus):
        """Mid-migration, the rows still waiting must be a number rather than an error."""
        backfill(["mig_one"], bucket=BUCKET)
        with pytest.raises(BackfillError, match="without an id"):
            verify(bucket=BUCKET)
