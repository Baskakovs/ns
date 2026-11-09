"""Tests for the snippets layer: assets in, the right files read, and nothing read too early."""

from __future__ import annotations

import boto3
import duckdb
import polars as pl
import pytest
from moto.server import ThreadedMotoServer
from yukka import Asset
from yukka.data import IndexConstituent

from novetly_score.data import NoSnippetsError, OpenDuck, Snippets, data, entities, entity
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


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    """A local S3-compatible store holding a small snippets layout, as an endpoint URL.

    The layout is the real one: ``snippets/<entity>/<YYYY-MM>.parquet`` plus the manifest the
    layer reads its file list from, with the same columns and one zero-row file. Written with
    DuckDB and uploaded with boto3, so the files under test are real parquet read over real HTTP
    range requests rather than a stand-in for them.
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
                 SELECT uuid()::VARCHAR AS snippet_id,
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
    """An environment pointing the layer at the fixture store, with no .env within reach."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENDPOINT_VAR, store)
    monkeypatch.setenv(ACCESS_KEY_VAR, "key")
    monkeypatch.setenv(SECRET_VAR, "secret")
    return monkeypatch


def rows_of(*names):
    """Total the fixture rows of the named companies, over every month.

    Args:
        *names: Snippet directory names.

    Returns:
        How many rows those companies hold in the fixture store.
    """
    return sum(rows for name, _, rows in FILES if name in names)


class TestResolvingAssets:
    """What a universe has to look like for the layer to take it."""

    def test_a_yukka_id_names_a_directory(self):
        """The directory is the id without its kind, which is the whole mapping."""
        assert entity("company:bmw") == "bmw"

    def test_a_directory_name_passes_through(self):
        """A caller holding only the bare name needs no yukka import."""
        assert entity("bmw") == "bmw"

    def test_a_yukka_asset_resolves(self):
        """Asset is what the yukka client takes, so it is what this layer takes."""
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
        frame = pl.DataFrame({"yukka_id": ["company:bmw", None, "company:sap"]})
        assert entities(frame) == ("bmw", "sap")

    def test_repeats_collapse_once_in_order(self):
        """A universe repeats a company per membership interval; reading it twice would double rows."""
        assert entities(["company:sap", "company:bmw", "company:sap"]) == ("sap", "bmw")

    def test_a_single_asset_needs_no_list(self):
        """One company is a common case and must not need wrapping."""
        assert entities("company:bmw") == ("bmw",)

    def test_a_non_company_is_refused(self):
        """The snippets hold one directory per company, so another kind is a mistake, not a miss."""
        with pytest.raises(ValueError, match="is a person"):
            entity("person:angela_merkel")

    def test_something_that_is_not_an_asset_is_refused(self):
        """A frame column or a tuple would otherwise resolve to nonsense."""
        with pytest.raises(TypeError, match="neither a YUKKA id"):
            entity(42)

    def test_a_blank_id_is_refused(self):
        """An empty string would name the snippets directory itself."""
        with pytest.raises(ValueError, match="blank"):
            entity("   ")

    def test_an_empty_universe_is_refused(self):
        """A read of nothing is a mistake worth catching before it looks like a result."""
        with pytest.raises(ValueError, match="no assets given"):
            data([])


class TestAUniverseDescribesItself:
    """What a universe says about itself, without reading anything."""

    def test_it_counts_its_companies(self, monkeypatch, tmp_path):
        """A universe repeats constituents; what matters is how many companies it names."""
        monkeypatch.chdir(tmp_path)
        assert len(data(["company:bmw", "company:sap", "company:bmw"])) == 2

    def test_it_names_the_directories(self, monkeypatch, tmp_path):
        """The resolution is inspectable, so a surprise is traceable to the mapping."""
        monkeypatch.chdir(tmp_path)
        assert data(["company:bmw", "company:sap"]).names == ("bmw", "sap")

    def test_it_prints_its_size(self, monkeypatch, tmp_path):
        """A repr worth reading in a notebook."""
        monkeypatch.chdir(tmp_path)
        assert repr(data("company:bmw")) == "Universe(1 companies)"


class TestNarrowingReachesNoNetwork:
    """Describing a read is free; only asking for its rows is not."""

    def test_describing_a_read_needs_no_credentials(self, monkeypatch, tmp_path):
        """With no store configured at all, a read can still be built and printed."""
        monkeypatch.chdir(tmp_path)
        for variable in (ENDPOINT_VAR, ACCESS_KEY_VAR, SECRET_VAR):
            monkeypatch.delenv(variable, raising=False)
        snippets = data(["company:bmw", "company:sap"]).snippets.months("2024-07", "2024-08").select("entity", "lang")
        assert "2 companies" in repr(snippets)
        assert "2024-07:2024-08" in repr(snippets)

    def test_narrowing_returns_a_new_read(self, monkeypatch, tmp_path):
        """A narrowing is a value, so the same read can be narrowed two ways independently."""
        monkeypatch.chdir(tmp_path)
        wide = data("company:bmw").snippets
        narrow = wide.months("2024-07")
        assert wide is not narrow
        assert "…:…" in repr(wide)

    def test_an_empty_span_is_the_whole_history(self, monkeypatch, tmp_path):
        """months() with nothing to narrow by narrows nothing, rather than selecting nothing."""
        monkeypatch.chdir(tmp_path)
        assert "…:…" in repr(data("company:bmw").snippets.months())

    def test_one_month_needs_one_argument(self, monkeypatch, tmp_path):
        """months("2024-07") is that month, not everything from it onwards."""
        monkeypatch.chdir(tmp_path)
        assert "2024-07:2024-07" in repr(data("company:bmw").snippets.months("2024-07"))

    @pytest.mark.parametrize("month", ["2024-13", "2024-1", "202407", "July 2024", ""])
    def test_a_month_that_is_not_a_month_is_refused(self, monkeypatch, tmp_path, month):
        """A malformed month would otherwise silently select no file."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="is not YYYY-MM"):
            data("company:bmw").snippets.months(month)

    def test_a_backwards_span_is_refused(self, monkeypatch, tmp_path):
        """Swapped ends select nothing, which looks like a company with no news."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="runs backwards"):
            data("company:bmw").snippets.months("2024-09", "2024-07")

    def test_an_unknown_column_is_refused(self, monkeypatch, tmp_path):
        """Named before the read, a typo costs no round-trip and names the real columns."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="no such snippet column"):
            data("company:bmw").snippets.select("entity", "sentimnet")

    def test_select_needs_a_column(self, monkeypatch, tmp_path):
        """An empty projection is not a read of nothing, it is a mistake."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="at least one"):
            data("company:bmw").snippets.select()


class TestTheFileListComesFromTheManifest:
    """Which files a read would open, decided without listing the bucket."""

    def test_every_month_of_every_company_asked_for(self, env):
        """Unnarrowed, the read covers each company's whole history."""
        files = data(["company:bmw", "company:sap"]).snippets.files
        assert sorted(files.get_column("month").to_list()) == ["2024-07", "2024-07", "2024-08", "2024-09", "2024-09"]

    def test_a_file_holding_no_rows_is_left_out(self, env):
        """309 such pairs exist in the real store; opening them buys nothing."""
        files = data("company:airbus").snippets.files
        assert files.get_column("month").to_list() == ["2024-08"]

    def test_the_span_drops_whole_files(self, env):
        """Month narrowing is file pruning, which is what keeps a read small."""
        snippets = data(["company:bmw", "company:sap"]).snippets.months("2024-08", "2024-09")
        assert snippets.files.get_column("month").to_list() == ["2024-08", "2024-09", "2024-09"]

    def test_the_paths_are_the_real_layout(self, env):
        """A path built here must be a path the store answers to."""
        assert data("company:sap").snippets.months("2024-07").paths == (f"s3://{BUCKET}/snippets/sap/2024-07.parquet",)

    def test_the_row_count_is_known_before_the_read(self, env):
        """The point of the manifest: how much a read returns, for the price of one small object."""
        assert data(["company:bmw", "company:sap"]).snippets.rows == rows_of("bmw", "sap")

    def test_coverage_summarises_each_company(self, env):
        """What a universe actually brings, deepest first, before anything is pulled."""
        coverage = data(["company:bmw", "company:sap", "company:airbus"]).snippets.coverage
        assert coverage.get_column("entity").to_list() == ["company:bmw", "company:sap", "company:airbus"]
        assert coverage.get_column("months").to_list() == [3, 2, 1]
        assert coverage.row(0, named=True)["rows"] == rows_of("bmw")
        assert coverage.row(0, named=True)["first_month"] == "2024-07"


class TestACompanyWithoutSnippetsIsReported:
    """A universe is wider than the store, and the difference must be visible."""

    def test_an_uncovered_company_is_named(self, env):
        """Of the real STOXX 600, a third has no snippets; silence there is a wrong result."""
        snippets = data([f"company:{UNCOVERED}", "company:bmw"]).snippets
        assert snippets.missing == (f"company:{UNCOVERED}",)

    def test_a_covered_company_is_not_named(self, env):
        """Nothing is reported missing when the store has it all."""
        assert data("company:bmw").snippets.missing == ()

    def test_an_uncovered_company_does_not_block_the_others(self, env):
        """A read of a whole universe must still return the companies that are there."""
        frame = data([f"company:{UNCOVERED}", "company:bmw"]).snippets.frame
        assert frame.height == rows_of("bmw")

    def test_a_company_outside_the_span_is_reported_missing(self, env):
        """Missing is about this read: sap has no 2024-08, so this read has no sap."""
        snippets = data(["company:bmw", "company:sap"]).snippets.months("2024-08")
        assert snippets.missing == ("company:sap",)


class TestWhatIsRead:
    """The rows that come back, and the columns they come back in."""

    def test_the_manifest_count_is_the_row_count(self, env):
        """The promise made before the read has to be the one the read keeps."""
        snippets = data(["company:bmw", "company:sap", "company:airbus"]).snippets
        assert snippets.frame.height == snippets.rows

    def test_the_entity_column_holds_the_yukka_id(self, env):
        """It joins straight back onto a YUKKA universe, so it must stay the id."""
        assert data("company:bmw").snippets.months("2024-07").frame.get_column("entity").unique().to_list() == [
            "company:bmw"
        ]

    def test_a_projection_fetches_only_what_it_names(self, env):
        """Dropping the snippet text is the difference between megabytes and gigabytes."""
        frame = data("company:bmw").snippets.months("2024-07").select("entity", "publish_time").frame
        assert frame.columns == ["entity", "publish_time"]

    def test_a_projection_keeps_the_order_named(self, env):
        """Columns come back as asked for, not in file order."""
        frame = data("company:bmw").snippets.months("2024-07").select("lang", "entity").frame
        assert frame.columns == ["lang", "entity"]

    def test_every_column_comes_back_by_default(self, env):
        """An unnarrowed read is the whole row, so a caller sees the schema as it is."""
        frame = data("company:bmw").snippets.months("2024-07").frame
        assert frame.columns == list(Snippets(()).select(*frame.columns)._columns)
        assert frame.height == rows_of("bmw") - 7 - 3  # 2024-07 alone

    def test_the_row_id_is_a_column_like_any_other(self, env):
        """snippet_id addresses the occurrence; it must be selectable on its own."""
        frame = data("company:bmw").snippets.months("2024-07").select("snippet_id").frame
        assert frame.columns == ["snippet_id"]
        assert frame.get_column("snippet_id").n_unique() == frame.height

    def test_the_span_narrows_the_rows(self, env):
        """File pruning has to show up in the result, not only in the file list."""
        assert data("company:bmw").snippets.months("2024-08", "2024-09").frame.height == 7 + 3

    def test_the_projection_is_pushed_into_the_read(self, env):
        """Dropped columns must be dropped by the parquet reader, not after it hands rows over."""
        with data("company:bmw").snippets.select("entity", "lang") as held:
            plan = held.relation().explain()
        reader = plan[plan.index("READ_PARQUET") :]
        assert "entity" in reader
        assert "lang" in reader
        assert "snippet" not in reader, "the reader must not be asked for columns the read dropped"

    def test_a_column_cannot_reach_a_query_unchecked(self, env):
        """Checked where columns enter, not only in select(), whatever built the read."""
        with pytest.raises(ValueError, match="no such snippet column"):
            Snippets(("bmw",), columns=("entity", "drop table"))


class TestOneConnectionForABlock:
    """Reads in a block share a connection; reads outside one each open and close their own."""

    def test_the_block_serves_several_reads(self, env):
        """The with form is for loops: open once, read many."""
        with data(["company:bmw", "company:sap"]).snippets as held:
            assert held.rows == rows_of("bmw", "sap")
            assert held.months("2024-07").frame.height == 5 + 4
            assert held.coverage.height == 2

    def test_the_relation_escape_hatch_needs_no_argument_in_a_block(self, env):
        """A relation composes into SQL; inside a block the connection under it is the held one."""
        with data("company:bmw").snippets as held:
            counted = held.relation().aggregate("entity, count(*) AS n", "entity").fetchall()
        assert counted == [("company:bmw", rows_of("bmw"))]

    def test_a_relation_outside_a_block_says_what_it_needs(self, env):
        """A connection opened for the call would close before the relation could be read."""
        with pytest.raises(RuntimeError, match="needs a connection"):
            data("company:bmw").snippets.relation()

    def test_a_connection_can_be_passed_in(self, env):
        """A caller already holding one must be able to keep using it."""
        with OpenDuck() as con:
            snippets = data("company:bmw", connection=con).snippets
            assert pl.DataFrame(snippets.relation()).height == rows_of("bmw")
            assert snippets.rows == rows_of("bmw")


class TestAnEmptyReadSaysWhy:
    """An empty frame is indistinguishable from a quiet company, so it is refused."""

    def test_a_company_the_store_does_not_hold(self, env):
        """The cause is the universe, not the span, and the message has to say so."""
        with pytest.raises(NoSnippetsError, match="no snippets at all"):
            _ = data(f"company:{UNCOVERED}").snippets.frame

    def test_a_span_the_store_does_not_reach(self, env):
        """A company that is covered, asked for over months that are not: a different mistake."""
        with pytest.raises(NoSnippetsError, match="2020-01") as raised:
            _ = data("company:bmw").snippets.months("2020-01", "2020-02").frame
        assert "2024-07 to 2024-09" in str(raised.value), "the message must name the span the store does hold"

    def test_a_month_with_no_rows(self, env):
        """A file that exists and holds nothing is as empty as no file at all."""
        with pytest.raises(NoSnippetsError, match="selects no file"):
            _ = data("company:airbus").snippets.months("2024-09").frame


@pytest.mark.integration
def test_reads_the_stoxx600_from_the_real_store():
    """Resolve the STOXX 600 through yukka and read one month of one company's snippets.

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
        snippets = data(universe).snippets
        covered = snippets.coverage
        assert covered.height > 0, "the real store holds snippets for part of the index"
        assert snippets.rows > 0
        deepest = covered.row(0, named=True)["entity"]
        frame = data(deepest).snippets.months(covered.row(0, named=True)["last_month"]).select("entity", "lang").frame
    except ConnectionFailedError as exc:
        if "not set" in str(exc):
            pytest.skip(str(exc))
        raise
    assert frame.get_column("entity").unique().to_list() == [deepest]
