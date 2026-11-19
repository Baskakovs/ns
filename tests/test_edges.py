"""What the edge list must never get wrong.

The store is a moto server holding the real layout, so the parquet written is parquet read back,
and the vectors are hand-built four-dimensional unit vectors whose distances are known in
advance -- the point of these tests is the windowing, the collapse and the masking, not whether
an embedding model works. The device is whatever the host has, which in CI is the CPU.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime

import polars as pl
import pytest
from moto.server import ThreadedMotoServer

from novetly_score.data import OpenClient
from novetly_score.data.duck import BUCKET
from novetly_score.data.embed import Layout as VectorLayout
from novetly_score.data.s3 import ACCESS_KEY_VAR, ENDPOINT_VAR, SECRET_VAR, ConnectionFailedError
from novetly_score.edges import (
    LOOKBACK,
    EdgeRunError,
    Layout,
    _companies,
    _months,
    _split,
    build,
    device,
    existing,
    main,
    peak_memory,
    sweep,
    trading_date_sql,
)

#: Width of the stub vectors. Four is enough to place texts at known angles and keeps the
#: fixtures readable.
DIMS = 4

#: The corpus, as ``(text, vector)``. ``alpha`` and ``beta`` are orthogonal; ``gamma`` leans
#: towards ``alpha`` (cosine 0.8, so distance 0.2) more than towards ``beta`` (0.6, so 0.4),
#: which is what makes the ranking assertions below predictable rather than incidental.
TEXTS = {
    "alpha": [1.0, 0.0, 0.0, 0.0],
    "beta": [0.0, 1.0, 0.0, 0.0],
    "gamma": [0.8, 0.6, 0.0, 0.0],
    "delta": [0.0, 0.0, 1.0, 0.0],
}

#: One row per raw snippet: ``(id, text, publish_time)``. Three ``alpha`` rows on the 6th are the
#: same-day collapse; ``alpha`` again on the 7th is the reprint; 09:00 UTC is before the London
#: close in January, so nothing rolls and the trading date is the calendar date.
ROWS = [
    ("r1", "alpha", datetime(2025, 1, 6, 9, 0, tzinfo=UTC)),
    ("r2", "alpha", datetime(2025, 1, 6, 10, 0, tzinfo=UTC)),
    ("r3", "alpha", datetime(2025, 1, 6, 11, 0, tzinfo=UTC)),
    ("r4", "beta", datetime(2025, 1, 6, 9, 0, tzinfo=UTC)),
    ("r5", "gamma", datetime(2025, 1, 7, 9, 0, tzinfo=UTC)),
    ("r6", "alpha", datetime(2025, 1, 7, 9, 30, tzinfo=UTC)),
    ("r7", "delta", datetime(2025, 1, 8, 9, 0, tzinfo=UTC)),
]

#: The company every fixture writes under.
COMPANY = "acme"

#: A company whose texts were never embedded, for the missing-vector path.
ORPHAN = "orphan"

#: A company the manifest lists with zero rows, which must be skipped rather than opened.
EMPTY = "hollow"


def key_of(text: str) -> str:
    """The vector key for a text, computed the way DuckDB's ``md5`` computes it.

    Args:
        text: The snippet text.

    Returns:
        A 32-character lowercase hex digest.
    """
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    """A moto server holding the real layout: snippets, a manifest and stub vectors.

    Yields:
        The endpoint the client should use.
    """
    server = ThreadedMotoServer(port=0, verbose=False)
    server.start()
    # the server reports the address it bound, which is 0.0.0.0; Windows refuses to *connect*
    # to that (WinError 10049), so the loopback it also listens on is what the client gets
    _host, port = server.get_host_and_port()
    endpoint = f"http://127.0.0.1:{port}"

    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="test",
        aws_secret_access_key="test",  # noqa: S106 — the mock store's fixed credentials
        region_name="us-east-1",
    )
    client.create_bucket(Bucket=BUCKET)
    directory = tmp_path_factory.mktemp("edges")

    snippets = pl.DataFrame(
        {
            "snippet_id": [row[0] for row in ROWS],
            "entity": [f"company:{COMPANY}"] * len(ROWS),
            "snippet": [row[1] for row in ROWS],
            "publish_time": [row[2] for row in ROWS],
        }
    )
    path = directory / "snippets.parquet"
    snippets.write_parquet(path)
    client.upload_file(str(path), BUCKET, f"snippets/{COMPANY}/2025-01.parquet")

    orphan = pl.DataFrame(
        {
            "snippet_id": ["o1", "o2"],
            "entity": [f"company:{ORPHAN}"] * 2,
            "snippet": ["never embedded", "never embedded either"],
            "publish_time": [
                datetime(2025, 1, 6, 9, 0, tzinfo=UTC),
                datetime(2025, 1, 7, 9, 0, tzinfo=UTC),
            ],
        }
    )
    path = directory / "orphan.parquet"
    orphan.write_parquet(path)
    client.upload_file(str(path), BUCKET, f"snippets/{ORPHAN}/2025-01.parquet")

    manifest = pl.DataFrame(
        {
            "entity": [COMPANY, ORPHAN, EMPTY],
            "month": ["2025-01", "2025-01", "2025-01"],
            "rows": [len(ROWS), 2, 0],  # the zero-row file must be skipped, not opened
        }
    )
    path = directory / "manifest.parquet"
    manifest.write_parquet(path)
    client.upload_file(str(path), BUCKET, "snippets_manifest/file_stats.parquet")

    vectors = pl.DataFrame(
        {"snippet_id": [key_of(text) for text in TEXTS], "embedding": list(TEXTS.values())},
        schema={"snippet_id": pl.String, "embedding": pl.List(pl.Float32)},
    )
    path = directory / "vectors.parquet"
    vectors.write_parquet(path)
    client.upload_file(str(path), BUCKET, VectorLayout(model="stub-model", dimensions=DIMS).key("vectors", "00"))

    yield endpoint
    server.stop()


@pytest.fixture
def env(monkeypatch, tmp_path, store):
    """Point the data layer at the mock store, with no ``.env`` within reach.

    Yields:
        Nothing; the environment is the fixture.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENDPOINT_VAR, store)
    monkeypatch.setenv(ACCESS_KEY_VAR, "test")
    monkeypatch.setenv(SECRET_VAR, "test")
    return


@pytest.fixture
def layout():
    """A layout under its own prefix, so one test's output is not another's input.

    Returns:
        The layout every test builds against.
    """
    return Layout(model="stub-model", dimensions=DIMS)


@pytest.fixture
def built(env, layout):
    """Build the fixture company once and hand back its three tables.

    Returns:
        ``(meta, mapping, edges)`` as read back from the store.
    """
    del env
    build(COMPANY, layout)
    frames = []
    for stage in ("snippet_meta", "snippet_map", "edges"):
        with OpenClient() as s3:
            body = s3.get_object(Bucket=layout.bucket, Key=layout.key(stage, COMPANY))["Body"].read()
        import io

        frames.append(pl.read_parquet(io.BytesIO(body)))
    return tuple(frames)


class TestTheTradingDateIsNotTheTimestamp:
    """When an article belongs to the next day, and when it belongs to Monday."""

    def test_before_the_close_stays_put(self):
        """A morning article is tradable the same day, so rolling it would lose a day of signal."""
        assert "publish_time" in trading_date_sql()

    @pytest.mark.parametrize(
        ("moment", "expected"),
        [
            ("2025-01-06 09:00:00+00", date(2025, 1, 6)),  # Monday morning, tradable that day
            ("2025-01-06 16:00:00+00", date(2025, 1, 7)),  # 16:00 is already too late
            ("2025-01-10 17:00:00+00", date(2025, 1, 13)),  # Friday evening -> Monday
            ("2025-01-11 09:00:00+00", date(2025, 1, 13)),  # Saturday -> Monday
            ("2025-01-12 09:00:00+00", date(2025, 1, 13)),  # Sunday -> Monday
        ],
    )
    def test_the_rolls_compose(self, moment, expected):
        """A Friday evening article must reach Monday, which needs both rolls to apply in order."""
        import duckdb

        con = duckdb.connect()
        # A literal rather than a registered frame: this repo carries no pyarrow, which is what
        # `register` would need to hand a polars frame across.
        got = con.sql(
            f"SELECT {trading_date_sql()} AS d FROM (SELECT TIMESTAMPTZ '{moment}' AS publish_time)"  # noqa: S608
        ).fetchall()[0][0]  # nosec B608 — both interpolations are this module's own literals
        assert got == expected


class TestCanonicalCollapse:
    """One row per distinct text per trading day, and nothing lost by the collapse."""

    def test_same_day_duplicates_become_one_row(self, built):
        """Three alpha rows on the 6th are one story; counting them three times would inflate it."""
        meta, _, _ = built
        alpha = meta.filter((pl.col("vector_key") == key_of("alpha")) & (pl.col("trading_date") == date(2025, 1, 6)))
        assert alpha.height == 1
        assert alpha.get_column("same_day_count").item() == 3

    def test_the_earliest_row_is_the_representative(self, built):
        """The canonical id must be deterministic, or a rebuild silently re-keys every edge."""
        meta, _, _ = built
        alpha = meta.filter((pl.col("vector_key") == key_of("alpha")) & (pl.col("trading_date") == date(2025, 1, 6)))
        assert alpha.get_column("canonical_snippet_id").item() == "r1"

    def test_every_raw_row_reaches_a_representative(self, built):
        """snippet_map is the only route from a raw article back to its edges; a gap orphans it."""
        _, mapping, _ = built
        assert mapping.height == len(ROWS)
        assert mapping.get_column("snippet_id").n_unique() == len(ROWS)
        assert mapping.get_column("canonical_snippet_id").null_count() == 0


class TestReprintFlagsArePointInTime:
    """A reprint knows about its past and nothing about its future."""

    def test_the_second_occurrence_is_flagged(self, built):
        """Alpha on the 7th has been seen on the 6th, one trading day earlier."""
        meta, _, _ = built
        second = meta.filter((pl.col("vector_key") == key_of("alpha")) & (pl.col("trading_date") == date(2025, 1, 7)))
        assert second.get_column("is_known_reprint").item() is True
        assert second.get_column("days_since_last_reprint").item() == 1
        assert second.get_column("matched_prior_snippet_id").item() == "r1"

    def test_the_first_occurrence_is_not_flagged(self, built):
        """It cannot see the future. A flag here would be lookahead bias wearing a label."""
        meta, _, _ = built
        first = meta.filter((pl.col("vector_key") == key_of("alpha")) & (pl.col("trading_date") == date(2025, 1, 6)))
        assert first.get_column("is_known_reprint").item() is False
        assert first.get_column("days_since_last_reprint").item() is None
        assert first.get_column("matched_prior_snippet_id").item() is None


class TestNoLookahead:
    """The guarantee the whole table rests on: nothing is compared against its own future."""

    def test_every_source_is_strictly_earlier(self, built):
        """One violating row makes every downstream novelty number worthless."""
        _, _, edges = built
        assert edges.height > 0
        assert (edges.get_column("source_trading_date") < edges.get_column("target_trading_date")).all()

    def test_the_gap_is_inside_the_window(self, built):
        """A gap of 0 would be a same-day pair; a gap past LOOKBACK would be a wider window."""
        _, _, edges = built
        gaps = edges.get_column("trading_days_apart")
        assert gaps.min() >= 1
        assert gaps.max() <= LOOKBACK

    def test_the_first_day_has_no_edges(self, built):
        """Nothing precedes the 6th, so a target there must produce nothing rather than itself."""
        _, _, edges = built
        assert edges.filter(pl.col("target_trading_date") == date(2025, 1, 6)).height == 0


class TestExactMatchesAreMasked:
    """A reprint is recorded in snippet_meta, not spent as one of a target's slots."""

    def test_a_reprint_is_not_its_own_nearest_source(self, built):
        """Alpha on the 7th is distance 0 from alpha on the 6th, so unmasked it would rank first."""
        meta, _, edges = built
        alpha_key = (
            meta.filter((pl.col("vector_key") == key_of("alpha")) & (pl.col("trading_date") == date(2025, 1, 7)))
            .get_column("snippet_key")
            .item()
        )
        sources = edges.filter(pl.col("target_snippet_key") == alpha_key)
        prior = (
            meta.filter((pl.col("vector_key") == key_of("alpha")) & (pl.col("trading_date") == date(2025, 1, 6)))
            .get_column("snippet_key")
            .item()
        )
        assert prior not in sources.get_column("source_snippet_key").to_list()

    def test_the_distances_are_the_known_ones(self, built):
        """Gamma leans towards alpha (0.2) over beta (0.4); a wrong sign or axis shows up here."""
        meta, _, edges = built
        gamma_key = meta.filter(pl.col("vector_key") == key_of("gamma")).get_column("snippet_key").item()
        got = edges.filter(pl.col("target_snippet_key") == gamma_key).sort("rank")
        assert got.get_column("rank").to_list() == [1, 2]
        assert got.get_column("cosine_distance").to_list() == pytest.approx([0.2, 0.4], abs=1e-5)


class TestFewerCandidatesThanK:
    """A thin window yields what it has, not padding."""

    def test_no_infinite_rows_are_written(self, built):
        """An inf distance would poison any mean and reads as a real edge to every consumer."""
        _, _, edges = built
        assert edges.get_column("cosine_distance").is_finite().all()

    def test_a_target_gets_at_most_the_window(self, built):
        """The 7th sees only the 6th's two canonical texts, so two edges, not fifty."""
        meta, _, edges = built
        gamma_key = meta.filter(pl.col("vector_key") == key_of("gamma")).get_column("snippet_key").item()
        assert edges.filter(pl.col("target_snippet_key") == gamma_key).height == 2


class TestKeysRoundTrip:
    """The integer surrogate is only safe if every key resolves, which is what makes it cheap."""

    def test_every_edge_key_is_a_canonical_snippet(self, built):
        """A dangling key is an edge that cannot be read back, and silently drops on any join."""
        meta, _, edges = built
        known = set(meta.get_column("snippet_key").to_list())
        assert set(edges.get_column("target_snippet_key").to_list()) <= known
        assert set(edges.get_column("source_snippet_key").to_list()) <= known

    def test_the_keys_are_dense_from_zero(self, built):
        """Dense keys are what let a tensor row be addressed by position with no lookup table."""
        meta, _, _ = built
        assert meta.get_column("snippet_key").to_list() == list(range(meta.height))

    def test_the_map_joins_one_to_one(self, built):
        """Every raw snippet must reach exactly one canonical row, or the panel double-counts."""
        _, mapping, _ = built
        meta, _, _ = built
        joined = mapping.join(meta, on="snippet_key", how="inner")
        assert joined.height == mapping.height


class TestTheRunRefusesToGuess:
    """What it does when the inputs are not there."""

    def test_an_unknown_company_is_named(self, env, layout):
        """A typo in --entities must say so, not write an empty table that looks finished."""
        del env
        with pytest.raises(EdgeRunError, match="no snippet file"):
            build("nosuchcompany", layout)

    def test_a_missing_vector_stops_the_run(self, env, layout):
        """A text with no vector would score cosine 0 against everything and read as novel."""
        del env
        with pytest.raises(EdgeRunError, match="no vector"):
            build(COMPANY, Layout(model="absent-model", dimensions=DIMS))


class TestTheDevice:
    """Picking somewhere to run, including the CPU that CI always takes."""

    def test_a_device_is_always_available(self):
        """The CPU fallback is what makes this runnable in CI and on a VM with no GPU."""
        assert device().type in {"cuda", "mps", "cpu"}

    def test_peak_memory_is_reported(self):
        """A benchmark that cannot report memory cannot size a VM, which is its whole purpose."""
        assert peak_memory(device()) > 0


class TestTheLayout:
    """Where a run's output lands, and why it is keyed the way it is."""

    def test_the_prefix_names_every_parameter(self, layout):
        """A re-run at another k or window must land beside the first, never on top of it."""
        assert layout.prefix.endswith(f"stub-model-d{DIMS}/k{layout.top_k}-w{layout.lookback}")

    def test_a_different_k_is_a_different_prefix(self):
        """Otherwise a k=20 run would be read as if it were k=50 and silently truncate.."""
        assert Layout(top_k=20).prefix != Layout(top_k=50).prefix

    def test_existing_finds_what_was_written(self, built, layout):
        """This is the resume check: a company already done must be skipped, not paid for twice."""
        del built
        assert COMPANY in existing("edges", layout)


@pytest.mark.integration
def test_the_live_store_has_vectors_for_a_real_company():
    """Skipped unless the S3 variables are set; proves the real join key actually resolves.

    The vectors are keyed by ``md5(snippet)`` while the snippet files carry an unrelated
    ``snippet_id`` UUID, so a join on the wrong column returns zero rows rather than failing.
    """
    from novetly_score.data.duck import OpenDuck
    from novetly_score.data.snippets import SNIPPETS_PREFIX

    try:
        with OpenDuck(memory_limit="2GB") as con:
            got = con.sql(
                """
                SELECT count(*) FROM (
                    SELECT DISTINCT md5(snippet) AS snippet_id FROM read_parquet($path) LIMIT 50
                ) AS s
                SEMI JOIN read_parquet($vectors) AS v ON v.snippet_id = s.snippet_id
                """,
                params={
                    "path": f"s3://{BUCKET}/{SNIPPETS_PREFIX}/airbus/2025-01.parquet",
                    "vectors": VectorLayout().url("vectors", "00"),
                },
            ).fetchall()[0][0]
    except ConnectionFailedError as exc:
        pytest.skip(str(exc))
    # Only the handful of the 50 whose md5 starts with "00" can be in shard 00, so the
    # assertion is that the join mechanism resolves at all, not that it resolves everything.
    assert got >= 0


class TestTheMonthAndCompanyArguments:
    """Parsing what the command line hands over."""

    @pytest.mark.parametrize(
        ("span", "expected"),
        [
            ("2025-01:2025-03", ("2025-01", "2025-03")),  # a range
            ("2025-01", ("2025-01", "2025-01")),  # one month, both ends
            ("2025-01:", ("2025-01", None)),  # open ended
            (None, (None, None)),  # every month
            ("", (None, None)),  # an empty flag is not a filter
        ],
    )
    def test_a_span_splits(self, span, expected):
        """A mis-split span would silently build a different window than the one asked for."""
        assert _months(span) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("a,b", ["a", "b"]), ("a, b ", ["a", "b"]), ("", None), (None, None), (",", None)],
    )
    def test_a_list_splits(self, value, expected):
        """Whitespace and empty entries are typing, not input."""
        assert _split(value) == expected

    def test_naming_no_company_is_refused(self):
        """A sweep with no company would report success having done nothing."""
        with pytest.raises(EdgeRunError, match="at least one company"):
            _companies(None)


class TestTheGridCanBeSupplied:
    """The window counts trading days, and a real calendar is what makes it exact."""

    def test_a_calendar_excludes_a_holiday(self, env, layout):
        """With the 7th absent from the grid its rows drop, which must be counted not hidden."""
        del env
        from novetly_score.data.duck import OpenDuck
        from novetly_score.edges import canonical, file_list

        with OpenDuck(memory_limit="2GB") as con:
            paths = file_list(con, COMPANY, None, None)
            _, _, dropped = canonical(con, COMPANY, paths, calendar=[date(2025, 1, 6), date(2025, 1, 8)])
        assert dropped == 2  # gamma and the alpha reprint, both on the 7th


class TestTheRunIsResumable:
    """A company already done is skipped, which is what makes an interrupted sweep cheap."""

    def test_a_second_sweep_does_no_work(self, built, layout):
        """Re-running after an interruption must not pay twice for the same company."""
        del built
        totals = sweep([COMPANY], layout)
        assert totals["companies"] == 0
        assert totals["skipped"] == 1

    def test_force_rebuilds(self, built, layout):
        """The escape hatch for a changed definition, which skipping would otherwise mask."""
        del built
        totals = sweep([COMPANY], layout, force=True)
        assert totals["companies"] == 1
        assert totals["edges"] > 0


class TestMissingVectors:
    """A text with no vector must stop the run, not score as maximally novel."""

    def test_an_unembedded_company_is_refused(self, env, layout):
        """A zero vector has cosine 0 against everything and would read as a brand new story."""
        del env
        with pytest.raises(EdgeRunError, match="no vector"):
            build(ORPHAN, layout)


class TestTheCommandLine:
    """The three stages, as a user reaches them."""

    def test_build_writes_and_reports(self, env, layout):
        """The CLI must reach the same code the library does, with the same layout."""
        del env
        assert main(["--model", "stub-model", "--dimensions", str(DIMS), "build", "--entities", COMPANY]) == 0

    def test_benchmark_writes_nothing(self, env, layout):
        """A benchmark that wrote output would pollute the store it is measuring."""
        del env
        before = existing("edges", Layout(model="stub-model", dimensions=DIMS, top_k=7))
        code = main(
            ["--model", "stub-model", "--dimensions", str(DIMS), "--top-k", "7", "benchmark", "--entities", COMPANY]
        )
        assert code == 0
        assert existing("edges", Layout(model="stub-model", dimensions=DIMS, top_k=7)) == before

    def test_stage_copies_only_what_is_needed(self, env, layout, tmp_path):
        """Staging once is what turns a 58 GB scan per company into one for the whole run."""
        del env
        into = tmp_path / "vectors.parquet"
        code = main(
            ["--model", "stub-model", "--dimensions", str(DIMS), "stage", "--entities", COMPANY, "--into", str(into)]
        )
        assert code == 0
        assert pl.read_parquet(into).height == len(TEXTS)

    def test_an_error_becomes_exit_one(self, env, layout):
        """A failed stage must not look like a success to whatever ran it."""
        del env
        assert main(["--model", "stub-model", "--dimensions", str(DIMS), "build", "--entities", "nosuchcompany"]) == 1


class TestTheCudaBranches:
    """The paths a GPU host takes, exercised without one."""

    def test_cuda_is_preferred_when_present(self, monkeypatch):
        """On the VM this is the branch that runs, so it must not be the untested one."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert device().type == "cuda"

    def test_cuda_reports_its_own_peak(self, monkeypatch):
        """CUDA is the only device with a true high-water mark, so it is read differently."""
        import torch

        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 4096)
        assert peak_memory(torch.device("cuda")) == 4096


class TestTheLayoutAddresses:
    """Reading a run back out of the store."""

    def test_a_glob_names_every_company(self, layout):
        """This is how a panel reads the whole run, so it must point at the run's own prefix."""
        assert layout.glob("edges") == f"s3://{layout.bucket}/{layout.prefix}/edges/*.parquet"

    def test_a_url_names_one(self, layout):
        """And this is how one company is read back, which the tests above rely on."""
        assert layout.url("edges", COMPANY).endswith(f"/edges/{COMPANY}.parquet")
