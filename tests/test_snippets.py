"""Tests for the snippets reader: assets in, the right files read, nothing silently short."""

from __future__ import annotations

import boto3
import duckdb
import polars as pl
import pytest
from moto.server import ThreadedMotoServer
from yukka import Asset
from yukka.data import IndexConstituent

from novetly_score.data import COLUMNS, NoSnippetsError, entities, entity, paths, snippets
from novetly_score.data.duck import BUCKET
from novetly_score.data.s3 import ACCESS_KEY_VAR, ENDPOINT_VAR, SECRET_VAR

#: The fixture store's contents: one entry per snippet file, with the rows it holds.
FILES = (
    ("bmw", "2024-07", 5),
    ("bmw", "2024-08", 7),
    ("bmw", "2024-09", 3),
    ("sap", "2024-07", 4),
    ("sap", "2024-09", 6),
    ("airbus", "2024-08", 2),
    ("airbus", "2024-09", 0),  # exists, holds nothing: must be skipped, not opened
)

#: A company the fixture store has no directory for at all.
UNCOVERED = "nestle"


def rows_of(*names):
    """Total the fixture rows of the named companies, over every month."""
    return sum(rows for name, _, rows in FILES if name in names)


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    """A local S3-compatible store holding the real snippets layout and its manifest.

    Written with DuckDB so ``snippet_id`` is a native UUID, as the backfilled corpus has it.
    """
    server = ThreadedMotoServer(port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="key",
        aws_secret_access_key="secret",  # noqa: S106 — the mock store's fixed credentials
        region_name="us-east-1",
    )
    s3.create_bucket(Bucket=BUCKET)

    workdir = tmp_path_factory.mktemp("snippets")
    con = duckdb.connect()
    for name, month, rows in FILES:
        path = workdir / f"{name}-{month}.parquet"
        con.execute(
            """COPY (
                 SELECT uuid() AS snippet_id,
                        'company:' || $entity AS entity,
                        ['pos', 'neu', 'neg'][(i % 3) + 1] AS sentiment,
                        'snippet text ' || i AS snippet,
                        $entity || '-' || $month || '-' || i AS article_id,
                        ($month || '-01')::TIMESTAMPTZ + INTERVAL (i) HOUR AS publish_time,
                        'professional' AS news_type,
                        'provider:opoint' AS provider,
                        'source:example' AS source,
                        'http://example.com/' || i AS read_more_link,
                        false AS implied,
                        ['en', 'de'][(i % 2) + 1] AS lang
                 FROM range($rows) AS t(i)
               ) TO $path (FORMAT parquet)""",
            {"entity": name, "month": month, "rows": rows, "path": str(path)},
        )
        s3.upload_file(str(path), BUCKET, f"snippets/{name}/{month}.parquet")

    manifest = workdir / "file_stats.parquet"
    con.execute(
        """COPY (
             SELECT entity, month, rows::BIGINT AS rows, rows::BIGINT AS articles,
                    (rows * 100)::BIGINT AS bytes, true AS sorted, rows > 0 AS schema_ok
             FROM (SELECT unnest($entities) AS entity, unnest($months) AS month, unnest($rows) AS rows)
           ) TO $path (FORMAT parquet)""",
        {
            "entities": [name for name, _, _ in FILES],
            "months": [month for _, month, _ in FILES],
            "rows": [rows for _, _, rows in FILES],
            "path": str(manifest),
        },
    )
    s3.upload_file(str(manifest), BUCKET, "snippets_manifest/file_stats.parquet")
    con.close()

    yield endpoint
    server.stop()


@pytest.fixture
def env(monkeypatch, tmp_path, store):
    """An environment pointing the reader at the fixture store, with no .env within reach."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENDPOINT_VAR, store)
    monkeypatch.setenv(ACCESS_KEY_VAR, "key")
    monkeypatch.setenv(SECRET_VAR, "secret")
    return monkeypatch


class TestResolvingAssets:
    """What a universe has to look like for the reader to take it. Reaches no network."""

    def test_a_yukka_id_names_a_directory(self):
        """The directory is the id without its kind, which is the whole mapping."""
        assert entity("company:bmw") == "bmw"

    def test_a_directory_name_passes_through(self):
        """A caller holding only the bare name needs no yukka import."""
        assert entity("bmw") == "bmw"

    def test_a_yukka_asset_resolves(self):
        """Asset is what the yukka client takes, so it is what this reader takes."""
        assert entity(Asset.from_yukka_id("company:bmw")) == "bmw"

    def test_an_index_constituent_resolves(self):
        """Index.assets hands out IndexConstituents, so they must work as they come."""
        constituent = IndexConstituent(name="BMW", isin="DE0005190003", ric="BMWG.DE", yukka_id="company:bmw")
        assert entity(constituent) == "bmw"

    def test_a_universe_frame_resolves(self):
        """Index.frame is a polars frame with a yukka_id column, so it is accepted whole."""
        frame = pl.DataFrame({"name": ["BMW", "SAP"], "yukka_id": ["company:bmw", "company:sap"]})
        assert entities(frame) == ("bmw", "sap")

    def test_a_frame_without_an_id_is_skipped(self):
        """A universe drops constituents it could not match; a null id names no directory."""
        assert entities(pl.DataFrame({"yukka_id": ["company:bmw", None, "company:sap"]})) == ("bmw", "sap")

    def test_repeats_collapse_once_in_order(self):
        """A universe repeats a company per membership interval; reading it twice doubles rows."""
        assert entities(["company:sap", "company:bmw", "company:sap"]) == ("sap", "bmw")

    def test_a_single_asset_needs_no_list(self):
        """One company is a common case and must not need wrapping."""
        assert entities("company:bmw") == ("bmw",)

    def test_a_non_company_is_refused(self):
        """The snippets hold one directory per company, so another kind is a mistake, not a miss."""
        with pytest.raises(ValueError, match="is a person"):
            entity("person:angela_merkel")

    def test_something_that_is_not_an_asset_is_refused(self):
        """A tuple or a number would otherwise resolve to nonsense."""
        with pytest.raises(TypeError, match="neither a YUKKA id"):
            entity(42)

    def test_a_blank_id_is_refused(self):
        """An empty string would name the snippets directory itself."""
        with pytest.raises(ValueError, match="blank"):
            entity("   ")

    def test_an_empty_universe_is_refused(self):
        """A read of nothing is a mistake worth catching before it looks like a result."""
        with pytest.raises(ValueError, match="no assets given"):
            entities([])


class TestWhichFilesAreRead:
    """The file list, decided from the manifest rather than by listing the bucket."""

    def test_every_month_of_every_company_asked_for(self, env):
        """Unbounded, the read covers each company's whole history."""
        assert len(paths(["company:bmw", "company:sap"])) == 5

    def test_a_file_holding_no_rows_is_left_out(self, env):
        """309 such pairs exist in the real store; opening them buys nothing."""
        assert paths("company:airbus") == (f"s3://{BUCKET}/snippets/airbus/2024-08.parquet",)

    def test_the_span_drops_whole_files(self, env):
        """Month narrowing is file pruning, which is what keeps a wide read affordable."""
        assert len(paths(["company:bmw", "company:sap"], "2024-08", "2024-09")) == 3

    def test_one_month_is_both_ends(self, env):
        """Passing the same month twice is how a single month is asked for."""
        assert paths("company:bmw", "2024-08", "2024-08") == (f"s3://{BUCKET}/snippets/bmw/2024-08.parquet",)

    @pytest.mark.parametrize("month", ["2024-13", "2024-1", "202407", "July 2024", ""])
    def test_a_month_that_is_not_a_month_is_refused(self, env, month):
        """A malformed month would otherwise silently select no file."""
        with pytest.raises(ValueError, match="is not YYYY-MM"):
            paths("company:bmw", month)

    def test_a_backwards_span_is_refused(self, env):
        """Swapped ends select nothing, which looks like a company with no news."""
        with pytest.raises(ValueError, match="runs backwards"):
            paths("company:bmw", "2024-09", "2024-07")


class TestACompanyWithoutSnippetsIsReported:
    """A universe is wider than the store, and the difference must be visible."""

    def test_an_uncovered_company_is_warned_about(self, env):
        """Of the real STOXX 600, a third has no snippets; silence there is a wrong result."""
        with pytest.warns(UserWarning, match=f"company:{UNCOVERED}"):
            paths([f"company:{UNCOVERED}", "company:bmw"])

    def test_a_covered_company_warns_about_nothing(self, env):
        """No warning when the store has it all, or the warning means nothing."""
        import warnings as warn

        with warn.catch_warnings():
            warn.simplefilter("error")
            paths("company:bmw")

    def test_an_uncovered_company_does_not_block_the_others(self, env):
        """A read of a whole universe must still return the companies that are there."""
        with pytest.warns(UserWarning, match="no snippets for"):
            frame = snippets([f"company:{UNCOVERED}", "company:bmw"])
        assert frame.height == rows_of("bmw")


class TestWhatIsRead:
    """The rows that come back, and the columns they come back in."""

    def test_the_whole_history_of_one_company(self, env):
        """The headline case: every month the store has, with no arguments but the asset."""
        assert snippets("company:bmw").height == rows_of("bmw")

    def test_the_whole_history_of_a_universe(self, env):
        """Several companies at once, which is what a universe is."""
        assert snippets(["company:bmw", "company:sap", "company:airbus"]).height == rows_of("bmw", "sap", "airbus")

    def test_every_column_comes_back_by_default(self, env):
        """An unnarrowed read is the whole row, so a caller sees the schema as it is."""
        assert snippets("company:bmw", "2024-07").columns == list(COLUMNS)

    def test_the_entity_column_holds_the_yukka_id(self, env):
        """It joins straight back onto a YUKKA universe, so it must stay the id."""
        assert snippets("company:bmw", "2024-07").get_column("entity").unique().to_list() == ["company:bmw"]

    def test_the_row_id_arrives_readable(self, env):
        """Stored as a native UUID, it still reaches a frame as text: Arrow converts it.

        Worth pinning: polars has no UUID dtype, so a direct ``pl.read_parquet`` of the same
        file gives 16 opaque bytes. It is the DuckDB-to-Arrow hop this reader goes through that
        makes the id readable, and a change of engine would quietly take that away.
        """
        import uuid

        frame = snippets("company:bmw", "2024-07", columns=["snippet_id"])
        assert frame.schema["snippet_id"] == pl.String
        assert uuid.UUID(frame.get_column("snippet_id").item(0)).version == 4

    def test_a_projection_fetches_only_what_it_names(self, env):
        """Dropping the snippet text is the difference between megabytes and gigabytes."""
        frame = snippets("company:bmw", "2024-07", columns=["entity", "publish_time"])
        assert frame.columns == ["entity", "publish_time"]

    def test_a_projection_keeps_the_order_named(self, env):
        """Columns come back as asked for, not in file order."""
        assert snippets("company:bmw", "2024-07", columns=["lang", "entity"]).columns == ["lang", "entity"]

    def test_the_projection_is_pushed_into_the_read(self, env):
        """Dropped columns must be dropped by the parquet reader, not after it hands rows over."""
        from novetly_score.data import OpenDuck

        with OpenDuck() as con:
            relation = con.sql(
                "SELECT * FROM read_parquet($paths)", params={"paths": list(paths("company:bmw", "2024-07"))}
            ).project("entity, lang")
            plan = relation.explain()
        reader = plan[plan.index("READ_PARQUET") :]
        assert "entity" in reader
        assert "snippet" not in reader, "the reader must not be asked for columns the read dropped"

    def test_an_unknown_column_is_refused(self, env):
        """Named before the read, a typo costs no round-trip and names the real columns."""
        with pytest.raises(ValueError, match="no such snippet column"):
            snippets("company:bmw", columns=["entity", "sentimnet"])

    def test_an_empty_projection_is_refused(self, env):
        """An empty column list is not a read of nothing, it is a mistake."""
        with pytest.raises(ValueError, match="at least one"):
            snippets("company:bmw", columns=[])

    def test_the_span_narrows_the_rows(self, env):
        """File pruning has to show up in the result, not only in the file list."""
        assert snippets("company:bmw", "2024-08", "2024-09").height == 7 + 3


class TestAnEmptyReadSaysWhy:
    """An empty frame is indistinguishable from a quiet company, so it is refused."""

    def test_a_company_the_store_does_not_hold(self, env):
        """The message names the span the store does hold, so the cause is diagnosable."""
        with pytest.raises(NoSnippetsError, match="2024-07 to 2024-09"):
            snippets(f"company:{UNCOVERED}")

    def test_a_span_the_store_does_not_reach(self, env):
        """A covered company asked for over months that are not."""
        with pytest.raises(NoSnippetsError, match="2020-01"):
            snippets("company:bmw", "2020-01", "2020-02")

    def test_a_month_with_no_rows(self, env):
        """A file that exists and holds nothing is as empty as no file at all."""
        with pytest.raises(NoSnippetsError):
            snippets("company:airbus", "2024-09", "2024-09")


@pytest.mark.integration
def test_reads_the_stoxx600_from_the_real_store():
    """Read one month of the deepest-covered company of the real STOXX 600.

    Skipped unless the S3 variables and a YUKKA token are set, so the suite stays runnable
    without either.
    """
    from novetly_score.data import ConnectionFailedError

    yukka_data = pytest.importorskip("yukka.data")
    try:
        universe = yukka_data.Index.STOXX600.frame
    except Exception as exc:  # noqa: BLE001 — any token or access failure is a skip, not a failure
        pytest.skip(f"STOXX 600 unavailable: {exc}")

    try:
        with pytest.warns(UserWarning, match="no snippets for"):
            selected = paths(universe, "2025-01", "2025-01")
        assert len(selected) > 0
        frame = snippets("company:volkswagen", "2025-01", "2025-01", columns=["entity", "lang"])
    except ConnectionFailedError as exc:
        if "not set" in str(exc):
            pytest.skip(str(exc))
        raise
    assert frame.get_column("entity").unique().to_list() == ["company:volkswagen"]
