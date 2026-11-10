"""Tests for the snippets layer: assets in, the right files read, and nothing read too early."""

from __future__ import annotations

from datetime import date

import boto3
import duckdb
import polars as pl
import pytest
from moto.server import ThreadedMotoServer
from yukka import Asset
from yukka.data import IndexConstituent

from novetly_score.data import (
    Counts,
    NoSnippetsError,
    OpenDuck,
    Snippets,
    as_of,
    data,
    entities,
    entity,
    identifier_history,
)
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

#: A company whose file is handcrafted for the counts aggregation. The layout above writes one
#: snippet per article, which is the one shape that cannot show what the counts are counting.
TONE_ENTITY = "tonecorp"
TONE_MONTH = "2024-07"

#: ``(article, publish_time, sentiments)`` — every rule the aggregation holds, one article each.
TONE_ARTICLES = (
    ("a1", "2024-07-15 08:00:00+00", ("pos", "pos")),  # one article, however many snippets
    ("a2", "2024-07-15 09:00:00+00", ("pos", "neg")),  # sums to zero: neutral
    ("a3", "2024-07-15 10:00:00+00", ("neg", "neu", "neg")),  # the sign, not the majority label
    ("a4", "2024-07-15 11:00:00+00", ("neu",)),  # neutral
    ("a5", "2024-07-15 12:00:00+00", (None,)),  # no tone at all: uncharacterised
    ("a6", "2024-07-15 23:30:00+00", ("pos",)),  # the 15th in UTC, the 16th east of it
    ("a7", "2024-07-16 00:30:00+00", ("neg",)),  # the 16th in UTC, the 15th west of it
)

#: What :attr:`Counts.frame` has to make of :data:`TONE_ARTICLES`.
TONE_PANEL = (
    {
        "entity": f"company:{TONE_ENTITY}",
        "day": date(2024, 7, 15),
        "positive": 2,
        "neutral": 2,
        "negative": 1,
        "uncharacterised": 1,
    },
    {
        "entity": f"company:{TONE_ENTITY}",
        "day": date(2024, 7, 16),
        "positive": 0,
        "neutral": 0,
        "negative": 1,
        "uncharacterised": 0,
    },
)

#: A universe frame over the fixture store, shaped like :attr:`yukka.data.Index.frame`: one row
#: per membership spell, each carrying the identifier that company was filed under *then*.
#:
#: ``bmw`` changes line inside the fixture's months, which is what makes point-in-time keying
#: observable: 2024-08-02 is the first day on the new one. ``sap`` changed long before the store's
#: history, so every day of it keys to the later line. ``unilever`` is the concurrent case — two
#: lines from the same day, the longer-running one surviving.
UNIVERSE = pl.DataFrame(
    {
        "yukka_id": [
            "company:bmw",
            "company:bmw",
            "company:sap",
            "company:sap",
            "company:unilever",
            "company:unilever",
            f"company:{UNCOVERED}",
        ],
        "ric": ["BMWG.L", "BMWG.DE", "SAPG.OLD", "SAPG.DE", "ULVR.L", "UNA.AS", "NESN.S"],
        "start_date": [
            date(2015, 10, 13),
            date(2024, 8, 2),
            date(2015, 10, 13),
            date(2020, 1, 2),
            date(2015, 10, 13),
            date(2015, 10, 13),
            date(2016, 1, 1),
        ],
        "end_date": [
            date(2024, 8, 1),
            date(2026, 9, 21),
            date(2020, 1, 1),
            date(2026, 9, 21),
            date(2026, 9, 21),
            date(2020, 12, 21),
            date(2026, 9, 21),
        ],
    }
)

#: The snippet rows :data:`TONE_ARTICLES` spells out, one per sentiment.
TONE_ROWS = tuple(
    (article, stamp, sentiment) for article, stamp, sentiments in TONE_ARTICLES for sentiment in sentiments
)


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
                 SELECT 'company:' || $entity AS entity,
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

    tone_path = workdir / f"{TONE_ENTITY}-{TONE_MONTH}.parquet"
    con.execute(
        """COPY (
             SELECT 'company:' || $entity AS entity,
                    sentiment,
                    'snippet text of ' || article AS snippet,
                    article AS article_id,
                    stamp::TIMESTAMPTZ AS publish_time,
                    'professional' AS news_type,
                    'provider:opoint' AS provider,
                    'source:example' AS source,
                    'http://example.com/' || article AS read_more_link,
                    false AS implied,
                    'en' AS lang
             FROM (SELECT unnest($articles) AS article,
                          unnest($stamps) AS stamp,
                          unnest($sentiments) AS sentiment)
           ) TO $path (FORMAT parquet)""",
        {
            "entity": TONE_ENTITY,
            "articles": [article for article, _, _ in TONE_ROWS],
            "stamps": [stamp for _, stamp, _ in TONE_ROWS],
            "sentiments": [sentiment for _, _, sentiment in TONE_ROWS],
            "path": str(tone_path),
        },
    )
    s3.upload_file(str(tone_path), BUCKET, f"snippets/{TONE_ENTITY}/{TONE_MONTH}.parquet")

    manifest = workdir / "file_stats.parquet"
    con.execute(
        """COPY (
             SELECT entity, month, rows::BIGINT AS rows, rows::BIGINT AS articles,
                    (rows * 100)::BIGINT AS bytes, true AS sorted, rows > 0 AS schema_ok
             FROM (SELECT unnest($entities) AS entity, unnest($months) AS month, unnest($rows) AS rows)
           ) TO $path (FORMAT parquet)""",
        {
            "entities": [name for name, _, _ in FILES] + [TONE_ENTITY],
            "months": [month for _, month, _ in FILES] + [TONE_MONTH],
            "rows": [rows for _, _, rows in FILES] + [len(TONE_ROWS)],
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


class TestTheIdentifierHistoryIsPointInTime:
    """A company's RIC is a fact about a date, not about the company."""

    def test_every_spell_is_a_row(self, env):
        """A company that changed line has two identifiers, and both have to survive."""
        history = identifier_history(UNIVERSE, "ric")
        assert history.filter(pl.col("yukka_id") == "company:bmw").get_column("ric").to_list() == [
            "BMWG.L",
            "BMWG.DE",
        ]

    def test_the_spells_come_back_in_order(self, env):
        """An as-of join needs them ordered, so the order is part of what this returns."""
        history = identifier_history(UNIVERSE, "ric")
        bmw = history.filter(pl.col("yukka_id") == "company:bmw").get_column("start_date").to_list()
        assert bmw == sorted(bmw)

    def test_concurrent_spells_resolve_to_the_survivor(self, env):
        """Two lines from the same day would key one day twice; the longer-running one wins."""
        unilever = identifier_history(UNIVERSE, "ric").filter(pl.col("yukka_id") == "company:unilever")
        assert unilever.height == 1
        assert unilever.get_column("ric").to_list() == ["ULVR.L"], "the line that survived, not the delisted one"

    def test_the_end_dates_come_too(self, env):
        """Masking a universe is somebody else's job, but it needs these dates to do it."""
        assert set(identifier_history(UNIVERSE, "ric").columns) == {"yukka_id", "ric", "start_date", "end_date"}

    def test_a_company_without_the_identifier_is_left_out(self, env):
        """There is nothing to file its rows under, and a null key joins to nothing useful."""
        frame = UNIVERSE.with_columns(
            ric=pl.when(pl.col("yukka_id") == "company:bmw").then(None).otherwise(pl.col("ric"))
        )
        assert "company:bmw" not in identifier_history(frame, "ric").get_column("yukka_id").to_list()

    def test_an_unknown_column_is_refused(self, env):
        """Named before a read rather than after it, and the message says what the frame has."""
        with pytest.raises(ValueError, match="no isin column"):
            identifier_history(UNIVERSE, "isin")

    def test_a_frame_without_spells_still_works(self, env):
        """A plain bridge of one row per company is a history with no dates in it."""
        frame = UNIVERSE.filter(~pl.col("ric").is_in(["BMWG.L", "SAPG.OLD", "UNA.AS"])).drop("start_date", "end_date")
        history = identifier_history(frame, "ric")
        assert history.height == 4
        assert history.get_column("start_date").null_count() == 4

    def test_ambiguity_without_spell_dates_is_refused(self, env):
        """With no dates there is no most recent line, so resolving it would be a guess."""
        with pytest.raises(ValueError, match="two rics"):
            identifier_history(UNIVERSE.drop("start_date", "end_date"), "ric")


class TestRelabellingIsAsOfTheDay:
    """`as_of` is the join that makes a panel's key point-in-time."""

    def panel(self, *rows):
        """A dated panel.

        Args:
            *rows: ``(entity, day, value)`` triples.

        Returns:
            A frame shaped like the counts panel, with one payload column.
        """
        return pl.DataFrame(
            {
                "entity": [row[0] for row in rows],
                "day": [row[1] for row in rows],
                "value": [row[2] for row in rows],
            }
        )

    def test_each_day_takes_the_line_in_force(self, env):
        """The whole point: a day before the change keys to the old line, after it to the new."""
        panel = self.panel(("company:bmw", date(2024, 8, 1), 1), ("company:bmw", date(2024, 8, 2), 2))
        assert as_of(panel, UNIVERSE, "ric").to_dicts() == [
            {"ric": "BMWG.DE", "day": date(2024, 8, 2), "value": 2},
            {"ric": "BMWG.L", "day": date(2024, 8, 1), "value": 1},
        ]

    def test_no_row_is_dropped(self, env):
        """Masking a universe is the caller's job, so relabelling keeps every row it was given."""
        panel = self.panel(("company:bmw", date(2024, 8, 1), 1), ("company:sap", date(2024, 8, 1), 2))
        assert as_of(panel, UNIVERSE, "ric").height == panel.height

    def test_a_day_no_spell_covers_keeps_the_last_line(self, env):
        """A company out of the index still has an identity, and its counts still have a key."""
        panel = self.panel(("company:sap", date(2020, 1, 1), 1))  # sap's first spell ends 2020-01-01
        assert as_of(panel, UNIVERSE, "ric").get_column("ric").to_list() == ["SAPG.OLD"]

    def test_a_day_before_every_spell_takes_the_first_line(self, env):
        """There is no earlier identifier to use, and dropping the row would be masking."""
        panel = self.panel(("company:bmw", date(2000, 1, 1), 1))
        assert as_of(panel, UNIVERSE, "ric").get_column("ric").to_list() == ["BMWG.L"]

    def test_the_payload_columns_survive_untouched(self, env):
        """Relabelling is a rename of the key, not a transformation of the rows."""
        panel = self.panel(("company:bmw", date(2024, 8, 2), 7))
        assert as_of(panel, UNIVERSE, "ric").get_column("value").to_list() == [7]

    def test_a_company_the_universe_cannot_key_is_absent(self, env):
        """A row with no identifier cannot be keyed by one; it is left out, not keyed to null."""
        panel = self.panel(("company:unknown", date(2024, 8, 2), 1), ("company:bmw", date(2024, 8, 2), 2))
        assert as_of(panel, UNIVERSE, "ric").get_column("ric").to_list() == ["BMWG.DE"]

    def test_the_date_column_can_be_named_anything(self, env):
        """The counts call it day; a price panel calls it date."""
        panel = self.panel(("company:bmw", date(2024, 8, 2), 1)).rename({"day": "date"})
        assert as_of(panel, UNIVERSE, "ric", on="date").columns == ["ric", "date", "value"]


class TestTheUniverseReadsTheCounts:
    """`data(assets).counts(start, end)` — the span is the argument, and the frame is the answer."""

    def test_a_span_of_months(self, env):
        """The headline call: two months in, one panel out."""
        frame = data(UNIVERSE).counts("2024-07", "2024-08")
        assert frame.columns == ["entity", "day", "positive", "neutral", "negative", "uncharacterised"]
        assert frame.get_column("entity").n_unique() == 2

    def test_an_open_end_runs_to_the_store(self, env):
        """A span with one end given runs to the edge of what the store holds, not to the other end."""
        onwards = data(UNIVERSE).counts("2024-08")
        assert onwards.equals(data(UNIVERSE).counts("2024-08", "2024-09"))
        assert onwards.height > data(UNIVERSE).counts("2024-08", "2024-08").height

    def test_an_open_start_runs_from_the_store(self, env):
        """The same on the other side, so end_date alone is a span and not an empty one."""
        until = data(UNIVERSE).counts(end_date="2024-08")
        assert until.equals(data(UNIVERSE).counts("2024-07", "2024-08"))

    def test_no_arguments_is_every_month(self, env):
        """Both ends open is everything, which is the whole fixture store."""
        everything = data(UNIVERSE).counts()
        assert everything.equals(data(UNIVERSE).counts("2024-07", "2024-09"))
        assert everything.height > data(UNIVERSE).counts("2024-08").height

    def test_a_day_trims_the_span(self, env):
        """A caller who thinks in dates gets the days asked for, not the months they fall in.

        The days are read off the panel rather than written down: a month's files hold whatever
        UTC days their articles fall on, which near a month boundary is not only that month.
        """
        month = data(UNIVERSE).counts("2024-07")
        days = month.get_column("day").unique().sort().to_list()
        assert len(days) > 1, "the fixture month has to span two days for trimming to show"
        trimmed = data(UNIVERSE).counts(days[-1], days[-1])
        assert trimmed.height > 0
        assert trimmed.get_column("day").unique().to_list() == [days[-1]]
        assert trimmed.height < month.height, "the month is read whole and trimmed, not read short"

    def test_a_day_as_an_open_start_trims_from_that_day(self, env):
        """A day on its own is a start, not a single day: it runs on to the store's latest."""
        days = data(UNIVERSE).counts().get_column("day").unique().sort().to_list()
        onwards = data(UNIVERSE).counts(days[1])
        assert onwards.get_column("day").min() == days[1]
        assert onwards.get_column("day").max() == days[-1]

    def test_a_date_object_is_a_span_end(self, env):
        """A caller holding dates should not have to format them first."""
        day = data(UNIVERSE).counts("2024-07").get_column("day").max()
        assert data(UNIVERSE).counts(day).equals(data(UNIVERSE).counts(day.isoformat()))

    def test_a_day_is_found_in_whichever_file_holds_it(self, env):
        """The files are not UTC-day aligned, so a day query reads one month either side.

        The fixture's earliest day sits in a file whose month it is not — exactly the case the
        real store shows at every month boundary. Without the padding this reads nothing.
        """
        day = data(UNIVERSE).counts("2024-07").get_column("day").min()
        found = data(UNIVERSE).counts(day, day)
        assert found.height > 0, "a day must not be missed because a neighbouring file holds it"
        assert found.get_column("day").unique().to_list() == [day]

    def test_a_backwards_day_span_is_refused(self, env):
        """Backwards is backwards whether the ends are months or days."""
        with pytest.raises(ValueError, match="runs backwards"):
            data(UNIVERSE).counts("2024-08-02", "2024-08-01")

    def test_a_span_end_that_is_neither_shape_is_refused(self, env):
        """The message has to name both shapes, because either one is reasonable to try."""
        with pytest.raises(ValueError, match="not YYYY-MM or YYYY-MM-DD"):
            data(UNIVERSE).counts("July 2024")

    def test_a_backwards_span_is_refused(self, env):
        """The read's own check, which still applies when the span arrives as two arguments."""
        with pytest.raises(ValueError, match="runs backwards"):
            data(UNIVERSE).counts("2024-09", "2024-07")

    def test_the_panel_can_be_keyed_by_a_ric(self, env):
        """What a price panel is keyed by, so the two join without a bridge per caller."""
        frame = data(UNIVERSE).counts("2024-07", "2024-09", key="ric")
        assert frame.columns[0] == "ric"
        assert set(frame.get_column("ric")) == {"BMWG.L", "BMWG.DE", "SAPG.DE"}

    def test_the_key_is_the_one_held_on_the_day(self, env):
        """The bmw line changes on 2024-08-02, and the panel has to change with it."""
        changed = date(2024, 8, 2)  # the day UNIVERSE moves bmw onto its new line
        frame = data(UNIVERSE).counts("2024-07", "2024-09", key="ric").filter(pl.col("ric").str.starts_with("BMWG"))
        before = frame.filter(pl.col("day") < changed)
        after = frame.filter(pl.col("day") >= changed)
        assert before.height > 0, "the fixture has to have days before the change"
        assert after.height > 0, "the fixture has to have days after it"
        assert before.get_column("ric").unique().to_list() == ["BMWG.L"]
        assert after.get_column("ric").unique().to_list() == ["BMWG.DE"]

    def test_keying_keeps_the_counts_intact(self, env):
        """Relabelling is a rename of the key, not an aggregation: every row has to survive it."""
        order = ["day", "positive", "neutral", "negative", "uncharacterised"]
        plain = data(UNIVERSE).counts("2024-07", "2024-09").drop("entity").sort(order)
        keyed = data(UNIVERSE).counts("2024-07", "2024-09", key="ric").drop("ric").sort(order)
        assert keyed.equals(plain)

    def test_a_line_the_company_has_left_is_not_used_after_it_left(self, env):
        """The sap line moved off SAPG.OLD in 2020, long before the store's history starts."""
        assert "SAPG.OLD" not in data(UNIVERSE).counts("2024-07", key="ric").get_column("ric").to_list()

    def test_ids_alone_cannot_be_keyed(self, env):
        """The store knows no RICs, so the frame is the only thing that could answer — say so."""
        with pytest.raises(ValueError, match="needs the universe frame"):
            data("company:bmw").counts("2024-07", key="ric")

    def test_an_unknown_key_fails_before_the_read(self, env):
        """A typo must not cost a read of the store first."""
        with pytest.raises(ValueError, match="no isin column"):
            data(UNIVERSE).counts("2024-07", key="isin")

    def test_an_empty_span_says_why(self, env):
        """A panel of nothing is as indistinguishable from a quiet company as a frame of nothing."""
        with pytest.raises(NoSnippetsError, match="no snippets at all"):
            data(f"company:{UNCOVERED}").counts("2024-07")

    def test_the_value_form_is_the_same_panel(self, env):
        """Snippets.counts is the same reduction, so over the same span the two must agree."""
        method = data(UNIVERSE).counts("2024-07", "2024-07")
        assert method.equals(data(UNIVERSE).snippets.months("2024-07").counts.frame)

    def test_the_two_forms_read_an_omitted_end_differently(self, env):
        """And where they differ, deliberately: months() repeats its start, counts() opens the span.

        Snippets.months("2024-07") is that one month, because a narrowing names the files it
        wants. counts("2024-07") is 2024-07 onwards, because a span's open end is the store's
        edge. Both are documented; this is here so neither drifts into the other silently.
        """
        assert data(UNIVERSE).counts("2024-07").height > data(UNIVERSE).snippets.months("2024-07").counts.frame.height


class TestCountsReduceSnippetsToArticles:
    """The daily panel: what a day's articles about a company reduce to, counted in the store."""

    def test_the_whole_panel(self, env):
        """The rules only matter together, so the fixture asserts them together as well."""
        panel = data(f"company:{TONE_ENTITY}").snippets.counts.frame
        assert panel.to_dicts() == list(TONE_PANEL)

    def test_an_article_counts_once_however_many_snippets_it_carries(self, env):
        """The grain is the article. Counting snippet rows would scale P and N by the fan-out."""
        panel = data(f"company:{TONE_ENTITY}").snippets.counts.frame.row(0, named=True)
        counted = panel["positive"] + panel["neutral"] + panel["negative"] + panel["uncharacterised"]
        assert counted == 6, "six articles on the 15th, from ten snippets"

    def test_a_balanced_article_is_neutral(self, env):
        """Tone is the sign of the sum, so positive and negative in one article cancel."""
        assert data(f"company:{TONE_ENTITY}").snippets.counts.frame.row(0, named=True)["neutral"] == 2

    def test_an_article_with_no_sentiment_is_not_neutral(self, env):
        """A company named but not characterised has no tone, which is not a tone of zero."""
        first = data(f"company:{TONE_ENTITY}").snippets.counts.frame.row(0, named=True)
        assert first["uncharacterised"] == 1
        assert first["neutral"] == 2, "the uncharacterised article must not have landed here"

    def test_the_day_is_the_utc_day(self, env):
        """Two articles sit either side of midnight UTC; a session timezone must not move them."""
        with OpenDuck() as con:
            con.execute("SET TimeZone = 'Pacific/Kiritimati'")  # UTC+14: both would fall on the 16th
            panel = data(f"company:{TONE_ENTITY}", connection=con).snippets.counts.frame
        assert panel.get_column("day").to_list() == [date(2024, 7, 15), date(2024, 7, 16)]

    def test_the_counts_come_back_as_integers(self, env):
        """DuckDB counts in HUGEINT, which arrives as a decimal and will not divide as expected."""
        panel = data(f"company:{TONE_ENTITY}").snippets.counts.frame
        assert [panel.schema[column] for column in ("positive", "neutral", "negative")] == [pl.Int64] * 3

    def test_several_companies_are_one_panel(self, env):
        """A panel is addressed by a universe, so the companies share one read and one frame."""
        panel = data(["company:bmw", "company:sap"]).snippets.months("2024-07").counts.frame
        assert panel.get_column("entity").unique().sort().to_list() == ["company:bmw", "company:sap"]

    def test_the_span_narrows_the_panel(self, env):
        """The counts are aggregated from the files the read selects, not from the whole history."""
        bmw = data("company:bmw").snippets
        narrowed = bmw.months("2024-07").counts.frame
        assert set(narrowed.get_column("day")) < set(bmw.counts.frame.get_column("day"))
        counted = narrowed.select("positive", "neutral", "negative", "uncharacterised").sum().row(0)
        assert sum(counted) == 5, "the five articles of 2024-07, and none of the other months'"

    def test_the_text_is_never_fetched(self, env):
        """The panel is affordable over the whole universe only if the snippet column stays put."""
        with data(f"company:{TONE_ENTITY}").snippets as held:
            plan = held.counts.relation().explain()
        reader = plan[plan.index("READ_PARQUET") :]
        assert "sentiment" in reader
        assert "snippet" not in reader, "the aggregation must not pull the text to count articles"

    def test_a_column_selection_is_overridden(self, env):
        """A counts read needs its four columns, whatever the read it was asked of selected."""
        counts = data(f"company:{TONE_ENTITY}").snippets.select("snippet").counts
        assert counts.frame.to_dicts() == list(TONE_PANEL)

    def test_the_read_underneath_is_reachable(self, env):
        """How much a panel will cost is a question about its files, which Snippets answers."""
        counts = data([f"company:{TONE_ENTITY}", f"company:{UNCOVERED}"]).snippets.counts
        assert counts.snippets.rows == len(TONE_ROWS)
        assert counts.snippets.missing == (f"company:{UNCOVERED}",)

    def test_describing_a_panel_reaches_no_network(self, monkeypatch, tmp_path):
        """Narrowing and asking for the counts is description; only .frame is a read."""
        monkeypatch.chdir(tmp_path)
        for variable in (ENDPOINT_VAR, ACCESS_KEY_VAR, SECRET_VAR):
            monkeypatch.delenv(variable, raising=False)
        counts = data("company:bmw").snippets.months("2024-07").counts
        assert isinstance(counts, Counts)
        assert repr(counts) == "Counts(1 companies, months 2024-07:2024-07)"

    def test_a_relation_outside_a_block_says_what_it_needs(self, env):
        """A connection opened for the call would close before the relation could be read."""
        with pytest.raises(RuntimeError, match="needs a connection"):
            data("company:bmw").snippets.counts.relation()

    def test_the_relation_composes_into_sql(self, env):
        """The escape hatch: the panel joins and windows inside DuckDB, not in Python."""
        with data(f"company:{TONE_ENTITY}").snippets as held:
            totalled = held.counts.relation().aggregate("sum(positive) AS p, sum(negative) AS n").fetchall()
        assert totalled == [(2, 2)]

    def test_an_empty_read_says_why(self, env):
        """An empty panel is as indistinguishable from a quiet company as an empty frame is."""
        with pytest.raises(NoSnippetsError, match="no snippets at all"):
            _ = data(f"company:{UNCOVERED}").snippets.counts.frame


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
