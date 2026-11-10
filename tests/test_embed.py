"""Tests for the embedding stages: what gets embedded, what resume skips, what verify refuses.

The OpenAI client is a stub throughout — the script's own batching, retrying, sharding and
reconciliation are what these tests are about, and none of it needs a real request. The store
is a moto server holding the real layout, so the parquet written is parquet read back.
"""

from __future__ import annotations

import boto3
import duckdb
import polars as pl
import pytest
from moto.server import ThreadedMotoServer

from novetly_score.data import OpenClient
from novetly_score.data.duck import BUCKET
from novetly_score.data.embed import (
    MAX_ATTEMPTS,
    MAX_CHARS,
    EmbeddingRunError,
    Layout,
    Limiter,
    batches,
    embed,
    embed_batch,
    existing,
    inventory,
    main,
    verify,
)
from novetly_score.data.s3 import ACCESS_KEY_VAR, ENDPOINT_VAR, SECRET_VAR

#: The fixture corpus: (entity, month, snippet texts). "shared" appears under two companies and
#: twice under one of them, so the deduplication has something to collapse.
CORPUS = (
    ("bmw", "2025-01", ["bmw one", "bmw two", "shared", "shared"]),
    ("bmw", "2025-02", ["bmw three"]),
    ("sap", "2025-01", ["sap one", "shared"]),
)

#: Distinct texts across the whole fixture corpus, and total rows before the collapse.
DISTINCT = 5
ROWS = 7

#: Width the stub answers in, small enough to read in a failure message.
DIMS = 4


class StubEmbeddings:
    """The ``client.embeddings`` namespace, answering with a vector per input."""

    def __init__(self, owner):
        """Hold the client that records the calls."""
        self.owner = owner

    def create(self, *, model, dimensions, input):  # noqa: A002 — the SDK's parameter name
        """Answer one request, or raise whatever the test scripted for this call."""
        self.owner.calls.append({"model": model, "dimensions": dimensions, "inputs": list(input)})
        if self.owner.errors:
            raise self.owner.errors.pop(0)
        # Unit-norm, like the real model's, and distinct per input: a one-hot vector.
        data = [
            type(
                "Item",
                (),
                {
                    "index": index,
                    "embedding": [1.0 if slot == index % dimensions else 0.0 for slot in range(dimensions)],
                },
            )()
            for index in range(len(input))
        ]
        usage = type("Usage", (), {"total_tokens": sum(len(text) for text in input)})()
        return type("Answer", (), {"data": data, "usage": usage})()


class StubClient:
    """An OpenAI client stand-in that records every request and can be told to fail."""

    def __init__(self, errors=None):
        """Hold the scripted failures, raised one per call before any answer."""
        self.calls = []
        self.errors = list(errors or [])
        self.embeddings = StubEmbeddings(self)


def api_error(status):
    """An exception shaped like the SDK's, carrying *status*."""
    return type("APIError", (Exception,), {"status_code": status})(f"HTTP {status}")


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    """A local S3-compatible store holding the fixture snippet corpus and its manifest."""
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

    workdir = tmp_path_factory.mktemp("embed")
    con = duckdb.connect()
    for entity, month, texts in CORPUS:
        path = workdir / f"{entity}-{month}.parquet"
        pl.DataFrame(
            {
                "entity": [f"company:{entity}"] * len(texts),
                "sentiment": ["pos"] * len(texts),
                "snippet": texts,
                "article_id": [f"{entity}-{month}-{i}" for i in range(len(texts))],
                "lang": ["en"] * len(texts),
            }
        ).write_parquet(path)
        s3.upload_file(str(path), BUCKET, f"snippets/{entity}/{month}.parquet")

    manifest = workdir / "file_stats.parquet"
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
    ).write_parquet(manifest)
    s3.upload_file(str(manifest), BUCKET, "snippets_manifest/file_stats.parquet")
    con.close()

    yield endpoint
    server.stop()


@pytest.fixture
def env(monkeypatch, tmp_path, store):
    """An environment pointing the stages at the fixture store, with no .env within reach."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENDPOINT_VAR, store)
    monkeypatch.setenv(ACCESS_KEY_VAR, "key")
    monkeypatch.setenv(SECRET_VAR, "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return monkeypatch


@pytest.fixture
def layout():
    """A layout under its own prefix, so one test's output is not another's input."""
    return Layout(model="stub-model", dimensions=DIMS)


@pytest.fixture
def stub(monkeypatch):
    """Install a StubClient in place of the OpenAI one, and hand it back for inspection."""

    def install(errors=None):
        client = StubClient(errors)
        monkeypatch.setattr("novetly_score.data.embed._client", lambda _key: client)
        return client

    return install


def clear(layout):
    """Delete every object of a layout, so a test starts from nothing."""
    with OpenClient() as s3:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=layout.prefix):
            for item in page.get("Contents", []):
                s3.delete_object(Bucket=BUCKET, Key=item["Key"])


class TestTheUnitIsTheDistinctText:
    """What the inventory decides to embed, and what it collapses."""

    def test_duplicates_collapse(self, env, layout):
        """Four bmw rows holding three distinct texts must be three inputs, not four."""
        clear(layout)
        counts = inventory(["bmw"], None, None, layout)
        assert counts["rows"] == 5  # four rows in 2025-01, one in 2025-02
        assert counts["texts"] == 4  # "shared" appears twice in the same month

    def test_the_same_text_under_two_companies_is_one_input(self, env, layout):
        """A snippet naming two companies appears in both files and is still one vector."""
        clear(layout)
        counts = inventory(None, None, None, layout)
        assert counts["rows"] == ROWS
        assert counts["texts"] == DISTINCT

    def test_occurrences_record_the_collapse(self, env, layout):
        """How many rows a vector stands for has to survive, or the dedupe is unauditable."""
        clear(layout)
        inventory(None, None, None, layout)
        frame = _read(layout, "inventory")
        shared = frame.filter(pl.col("snippet") == "shared")
        assert shared.height == 1
        assert shared.get_column("occurrences").item() == 3

    def test_the_id_is_the_text_md5(self, env, layout):
        """Anyone must be able to recompute the key from the text alone, in any engine."""
        clear(layout)
        inventory(["sap"], None, None, layout)
        frame = _read(layout, "inventory")
        with duckdb.connect() as con:
            for row in frame.iter_rows(named=True):
                (want,) = con.execute("SELECT md5(?)", [row["snippet"]]).fetchone()
                assert row["snippet_id"] == want

    def test_a_month_narrows_the_inventory(self, env, layout):
        """The span must reach the file list, not only the report."""
        clear(layout)
        assert inventory(["bmw"], "2025-02", "2025-02", layout)["texts"] == 1

    def test_a_selection_with_no_file_is_refused(self, env, layout):
        """An empty selection would otherwise write an empty inventory and look finished."""
        with pytest.raises(EmbeddingRunError, match="holds no snippet file"):
            inventory(["bmw"], "2030-01", "2030-01", layout)


class TestBatching:
    """How a shard is cut into requests."""

    def test_a_batch_respects_the_input_count(self):
        """Over 512 inputs must become more than one request."""
        frame = pl.DataFrame({"snippet_id": [f"{i:032x}" for i in range(1300)], "snippet": ["x"] * 1300})
        assert [part.height for part in batches(frame)] == [512, 512, 276]

    def test_a_batch_respects_the_token_estimate(self):
        """A few huge snippets must not ride in one request because they are few."""
        frame = pl.DataFrame({"snippet_id": [f"{i:032x}" for i in range(10)], "snippet": ["x" * 200_000] * 10})
        assert len(list(batches(frame))) > 1

    def test_every_row_is_in_exactly_one_batch(self):
        """A row dropped here is a snippet silently never embedded."""
        frame = pl.DataFrame({"snippet_id": [f"{i:032x}" for i in range(1000)], "snippet": ["x" * 100] * 1000})
        seen = [key for part in batches(frame) for key in part.get_column("snippet_id")]
        assert seen == frame.get_column("snippet_id").to_list()

    def test_an_empty_frame_makes_no_request(self):
        """Nothing to embed must cost nothing."""
        assert (
            list(
                batches(
                    pl.DataFrame(
                        {"snippet_id": [], "snippet": []}, schema={"snippet_id": pl.String, "snippet": pl.String}
                    )
                )
            )
            == []
        )


class TestOneRequest:
    """What embed_batch sends, and what it does when the API says no."""

    def test_it_sends_the_model_and_the_width(self, layout):
        """The width is the difference between 67 GB and 401 GB; it must be on the request."""
        client = StubClient()
        frame = pl.DataFrame({"snippet_id": ["a" * 32], "snippet": ["hello"]})
        vectors, tokens = embed_batch(client, frame, layout, Limiter(10_000, 10_000_000))
        assert client.calls[0]["model"] == "stub-model"
        assert client.calls[0]["dimensions"] == DIMS
        assert len(vectors[0]) == DIMS
        assert tokens == len("hello")

    def test_an_overlong_snippet_is_truncated(self, layout):
        """8192 tokens is the hard limit; one long snippet must not fail its whole batch."""
        client = StubClient()
        frame = pl.DataFrame({"snippet_id": ["a" * 32], "snippet": ["x" * (MAX_CHARS + 500)]})
        embed_batch(client, frame, layout, Limiter(10_000, 10_000_000))
        assert len(client.calls[0]["inputs"][0]) == MAX_CHARS

    def test_a_rate_limit_is_retried(self, layout):
        """429 is weather, not an error."""
        client = StubClient(errors=[api_error(429)])
        frame = pl.DataFrame({"snippet_id": ["a" * 32], "snippet": ["hello"]})
        vectors, _ = embed_batch(client, frame, layout, Limiter(10_000, 10_000_000))
        assert len(vectors) == 1
        assert len(client.calls) == 2

    def test_a_bad_request_is_not_retried(self, layout):
        """400 means the batch is wrong and will stay wrong; retrying burns the rate limit."""
        client = StubClient(errors=[api_error(400)])
        frame = pl.DataFrame({"snippet_id": ["a" * 32], "snippet": ["hello"]})
        with pytest.raises(EmbeddingRunError, match="failed after"):
            embed_batch(client, frame, layout, Limiter(10_000, 10_000_000))
        assert len(client.calls) == 1

    def test_it_gives_up_eventually(self, layout):
        """A permanently sick endpoint must end the batch, not the run."""
        client = StubClient(errors=[api_error(500)] * MAX_ATTEMPTS)
        frame = pl.DataFrame({"snippet_id": ["a" * 32], "snippet": ["hello"]})
        with pytest.raises(EmbeddingRunError):
            embed_batch(client, frame, layout, Limiter(10_000, 10_000_000))
        assert len(client.calls) == MAX_ATTEMPTS


class TestTheRunIsResumable:
    """What a second run does with the work the first one finished."""

    def test_it_embeds_every_inventoried_text(self, env, layout, stub):
        """One vector per distinct snippet, no more and no fewer."""
        clear(layout)
        inventory(None, None, None, layout)
        stub()
        counts = embed(layout, workers=2)
        assert counts["embedded"] == DISTINCT
        assert _read(layout, "vectors").height == DISTINCT

    def test_a_second_run_embeds_nothing(self, env, layout, stub):
        """Re-running after success must cost nothing — this is what makes a crash cheap."""
        clear(layout)
        inventory(None, None, None, layout)
        stub()
        embed(layout, workers=2)
        client = stub()
        counts = embed(layout, workers=2)
        assert counts["embedded"] == 0
        assert client.calls == []

    def test_a_missing_shard_is_the_only_work(self, env, layout, stub):
        """A killed run loses one shard; the rerun must redo that one and no other."""
        clear(layout)
        inventory(None, None, None, layout)
        stub()
        embed(layout, workers=2)
        shards = sorted(existing("vectors", layout))
        dropped = shards[0]
        with OpenClient() as s3:
            s3.delete_object(Bucket=BUCKET, Key=layout.key("vectors", dropped))
        client = stub()
        embed(layout, workers=2)
        assert sorted(existing("vectors", layout)) == shards
        sent = sum(len(call["inputs"]) for call in client.calls)
        assert 0 < sent < DISTINCT

    def test_dry_run_sends_nothing(self, env, layout, stub):
        """A way to see the pending work without opening the tap."""
        clear(layout)
        inventory(None, None, None, layout)
        client = stub()
        counts = embed(layout, dry_run=True)
        assert counts["shards"] > 0
        assert client.calls == []

    def test_embedding_before_inventory_is_refused(self, env, layout):
        """Nothing to embed is a mistake in the order of the stages, not an empty result."""
        clear(layout)
        with pytest.raises(EmbeddingRunError, match="run the inventory command first"):
            embed(layout)


class TestFailuresAreRecorded:
    """What happens to a batch that cannot be embedded at all."""

    def test_a_failed_batch_lands_in_failed(self, env, layout, stub):
        """A snippet that could not be embedded has to be nameable afterwards."""
        clear(layout)
        inventory(["bmw"], "2025-02", "2025-02", layout)
        stub(errors=[api_error(400)])
        counts = embed(layout, workers=1)
        assert counts["failed"] == 1
        assert existing("failed", layout)

    def test_a_failure_forces_its_shard_to_be_redone(self, env, layout, stub):
        """Otherwise the shard looks done and the snippet is lost in silence."""
        clear(layout)
        inventory(["bmw"], "2025-02", "2025-02", layout)
        stub(errors=[api_error(400)])
        embed(layout, workers=1)
        client = stub()
        counts = embed(layout, workers=1)
        assert counts["embedded"] == 1
        assert client.calls, "the shard with a failure must be attempted again"
        assert not existing("failed", layout), "a failure made untrue must not stay on record"


class TestVerifyRefusesToBeFooled:
    """What counts as complete."""

    def test_a_complete_run_verifies(self, env, layout, stub):
        """The happy path has to be reachable, or the check is useless."""
        clear(layout)
        inventory(None, None, None, layout)
        stub()
        embed(layout, workers=2)
        report = verify(layout)
        assert report["wanted"] == DISTINCT
        assert report["written"] == DISTINCT
        assert report["missing"] == 0

    def test_a_missing_vector_is_named(self, env, layout, stub):
        """The anti-join is the whole point: a shortfall must be a number, not a hunch."""
        clear(layout)
        inventory(None, None, None, layout)
        stub()
        embed(layout, workers=2)
        with OpenClient() as s3:
            s3.delete_object(Bucket=BUCKET, Key=layout.key("vectors", sorted(existing("vectors", layout))[0]))
        with pytest.raises(EmbeddingRunError, match="without a vector"):
            verify(layout)

    def test_a_vector_of_the_wrong_width_is_caught(self, env, layout, stub):
        """A truncated width reads as a working run and scores nonsense."""
        clear(layout)
        inventory(["bmw"], "2025-02", "2025-02", layout)
        stub()
        embed(layout, workers=1)
        shard = sorted(existing("vectors", layout))[0]
        frame = _read(layout, "vectors").with_columns(embedding=pl.lit([1.0, 2.0]))
        _write(frame, layout, "vectors", shard)
        with pytest.raises(EmbeddingRunError, match="wrong width"):
            verify(layout)

    def test_an_all_zero_vector_is_caught(self, env, layout, stub):
        """A zero vector has cosine 0 against everything and would read as maximally novel."""
        clear(layout)
        inventory(["bmw"], "2025-02", "2025-02", layout)
        stub()
        embed(layout, workers=1)
        shard = sorted(existing("vectors", layout))[0]
        frame = _read(layout, "vectors").with_columns(embedding=pl.lit([0.0] * DIMS))
        _write(frame, layout, "vectors", shard)
        with pytest.raises(EmbeddingRunError, match="degenerate"):
            verify(layout)

    def test_verifying_before_embedding_is_refused(self, env, layout):
        """No vectors at all is not a complete run."""
        clear(layout)
        inventory(["bmw"], "2025-02", "2025-02", layout)
        with pytest.raises(EmbeddingRunError, match="run the embed command"):
            verify(layout)


class TestTheCommandLine:
    """That the flags reach the stages. A flag parsed and dropped is worse than no flag."""

    def test_dry_run_reaches_the_stage(self, env, layout, stub):
        """--dry-run must open no tap: it once parsed fine and embedded anyway."""
        clear(layout)
        inventory(["bmw"], "2025-02", "2025-02", layout)
        client = stub()
        assert main(["--model", layout.model, "--dimensions", str(DIMS), "embed", "--dry-run"]) == 0
        assert client.calls == [], "--dry-run must not reach the API"
        assert not existing("vectors", layout)

    def test_the_model_and_width_reach_the_layout(self, env, layout, stub):
        """Both name the output prefix, so a typo must not quietly write somewhere else."""
        clear(layout)
        assert main(["--model", layout.model, "--dimensions", str(DIMS), "inventory", "--entities", "bmw"]) == 0
        assert existing("inventory", layout)

    def test_an_incomplete_run_exits_nonzero(self, env, layout, stub):
        """The exit code is what a pipeline or a cron reads; it has to mean something."""
        clear(layout)
        inventory(["bmw"], "2025-02", "2025-02", layout)
        assert main(["--model", layout.model, "--dimensions", str(DIMS), "verify"]) == 1

    def test_a_complete_run_exits_zero(self, env, layout, stub):
        """And the happy path has to be reachable through the command line too."""
        clear(layout)
        inventory(["bmw"], "2025-02", "2025-02", layout)
        stub()
        embed(layout, workers=1)
        assert main(["--model", layout.model, "--dimensions", str(DIMS), "verify"]) == 0

    def test_a_month_span_is_parsed(self, env, layout, stub):
        """One month, a closed span and an open one are all spellings a user will try."""
        from novetly_score.data.embed import _months

        assert _months("2025-01") == ("2025-01", "2025-01")
        assert _months("2025-01:2025-03") == ("2025-01", "2025-03")
        assert _months("2025-01:") == ("2025-01", None)
        assert _months(None) == (None, None)


class TestTheLimiter:
    """The pacing that keeps the common case away from 429."""

    def test_it_lets_the_first_request_straight_through(self):
        """Pacing must not cost latency when there is headroom."""
        limiter = Limiter(600, 600_000)
        limiter.take(100)  # returns immediately or the test times out

    def test_it_holds_back_once_the_bucket_is_full(self):
        """A tight limit has to actually block, or the pacing is decoration."""
        now, slept = [0.0], []
        limiter = Limiter(
            2, 10_000_000, clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s))
        )
        for _ in range(3):
            limiter.take(1)
        assert slept, "the third request of a two-per-minute budget must wait"

    def test_it_paces_on_tokens_as_well_as_requests(self):
        """A few huge requests can exhaust a token budget while the request budget is untouched."""
        now, slept = [0.0], []
        limiter = Limiter(
            10_000, 100, clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s))
        )
        limiter.take(100)
        limiter.take(100)
        assert slept


def _read(layout, stage):
    """Read every shard of one stage back as a frame."""
    from novetly_score.data import OpenDuck

    with OpenDuck() as con:
        return pl.DataFrame(con.sql("SELECT * FROM read_parquet($path)", params={"path": layout.glob(stage)}))


def _write(frame, layout, stage, shard):
    """Overwrite one shard, to plant the damage a test is about."""
    from novetly_score.data.embed import upload

    upload(frame, layout.key(stage, shard), layout.bucket)
