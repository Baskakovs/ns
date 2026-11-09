"""Tests for OpenClient: where the credentials come from, and what counts as connected."""

from __future__ import annotations

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from moto.server import ThreadedMotoServer

from novetly_score.data import ConnectionFailedError, OpenClient
from novetly_score.data.s3 import ACCESS_KEY_VAR, ENDPOINT_VAR, SECRET_VAR

ENDPOINT = "https://yl-minio.example.com"


@pytest.fixture(scope="session")
def store():
    """A local S3-compatible store, as an endpoint URL.

    moto's in-process mock dispatches on AWS-shaped hostnames, so it cannot stand in for a
    custom endpoint. Its server mode can: it is an HTTP endpoint like MinIO's, which is what
    makes these tests exercise the real client configuration rather than a bypass of it.
    """
    server = ThreadedMotoServer(port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A clean environment holding valid S3 variables, with no .env within reach."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENDPOINT_VAR, ENDPOINT)
    monkeypatch.setenv(ACCESS_KEY_VAR, "key")
    monkeypatch.setenv(SECRET_VAR, "secret")
    return monkeypatch


def client_error(code):
    """A ClientError carrying *code*, shaped as botocore raises it."""
    return ClientError({"Error": {"Code": code, "Message": code}}, "ListBuckets")


class FakeClient:
    """A boto3 client stand-in answering list_buckets with whatever it was handed."""

    def __init__(self, answer):
        """Hold the scripted answer: a response dict, or an exception to raise."""
        self.answer = answer
        self.closed = False

    def list_buckets(self):
        """Return the scripted response, or raise the scripted exception."""
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def close(self):
        """Record that the block closed this client."""
        self.closed = True


@pytest.fixture
def fake_boto(monkeypatch):
    """Install a FakeClient in place of boto3's, so the check meets a scripted answer."""

    def install(answer):
        client = FakeClient(answer)
        monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)
        return client

    return install


class TestTheBlockIsTheLifetime:
    """Construction opens nothing; the with block opens and closes."""

    def test_construction_reaches_no_network(self, env, fake_boto):
        """Building an OpenClient does not call boto3 at all."""
        client = fake_boto(client_error("InvalidAccessKeyId"))
        OpenClient()  # would raise on entry, but nothing is entered
        assert not client.closed

    def test_the_block_closes_the_client(self, env, fake_boto):
        """Leaving the block releases the client's sockets."""
        client = fake_boto({"Buckets": []})
        with OpenClient() as s3:
            assert s3 is client
            assert not client.closed
        assert client.closed

    def test_the_block_closes_on_an_exception(self, env, fake_boto):
        """A raising block still closes the client, and the exception propagates."""
        client = fake_boto({"Buckets": []})
        with pytest.raises(ValueError, match="boom"), OpenClient():
            raise ValueError("boom")
        assert client.closed

    def test_one_instance_serves_two_blocks(self, env, store):
        """Re-entering an OpenClient opens a fresh client."""
        opener = OpenClient(store)
        with opener as first:
            first.create_bucket(Bucket="experiments")
        with opener as second:
            assert second is not first
            # Membership, not equality: moto's backend is global to the process, so the other
            # test modules' buckets are visible here too. What this proves is that the second
            # block sees what the first one wrote.
            assert "experiments" in [b["Name"] for b in second.list_buckets()["Buckets"]]


class TestCredentials:
    """Which of the argument, the environment and .env wins."""

    def test_arguments_need_no_environment(self, monkeypatch, tmp_path, store):
        """Passing all three reaches the store without reading anything."""
        monkeypatch.chdir(tmp_path)
        for var in (ENDPOINT_VAR, ACCESS_KEY_VAR, SECRET_VAR):
            monkeypatch.delenv(var, raising=False)
        with OpenClient(store, "key", "secret") as s3:
            assert s3.list_buckets()["ResponseMetadata"]["HTTPStatusCode"] == 200

    def test_the_environment_fills_what_is_not_passed(self, env, store):
        """An endpoint argument overrides S3_ENDPOINT; the key pair still comes from the environment."""
        env.setenv(ENDPOINT_VAR, "https://wrong.example.com")
        with OpenClient(store) as s3:
            assert s3.meta.endpoint_url == store

    def test_a_trailing_slash_is_stripped(self, env, store):
        """An endpoint written with a trailing slash is normalised."""
        with OpenClient(store + "/") as s3:
            assert s3.meta.endpoint_url == store

    def test_missing_credentials_are_all_named(self, env):
        """An absent and a blank variable are both reported, in one message."""
        env.delenv(ACCESS_KEY_VAR)
        env.setenv(SECRET_VAR, "   ")
        with pytest.raises(ConnectionFailedError) as exc, OpenClient():
            pass
        assert ACCESS_KEY_VAR in str(exc.value)
        assert SECRET_VAR in str(exc.value)
        assert ENDPOINT_VAR not in str(exc.value)

    def test_a_dotenv_file_fills_an_unset_variable(self, env, tmp_path, store):
        """What the environment lacks, a .env in the working directory supplies."""
        env.delenv(ACCESS_KEY_VAR)
        (tmp_path / ".env").write_text(f"{ACCESS_KEY_VAR}=from-file\n")
        with OpenClient(store) as s3:
            assert s3.meta.endpoint_url == store

    def test_the_environment_wins_over_the_file(self, env, tmp_path):
        """An exported variable is not overridden by the checked-out .env."""
        (tmp_path / ".env").write_text(f"{ENDPOINT_VAR}=https://from-file.example.com\n")
        with OpenClient(check=False) as s3:
            assert s3.meta.endpoint_url == ENDPOINT


class TestAgainstALocalStore:
    """The client driven against a local S3-compatible store."""

    def test_it_reads_and_writes(self, env, store):
        """A bucket created through the client is a bucket it can then list."""
        with OpenClient(store) as s3:
            s3.create_bucket(Bucket="reference-sets")
            assert "reference-sets" in [b["Name"] for b in s3.list_buckets()["Buckets"]]

    def test_path_addressing_is_the_default(self, env, store):
        """MinIO's buckets are not subdomains, so the default must be path-style."""
        with OpenClient(store) as s3:
            assert s3.meta.config.s3["addressing_style"] == "path"

    def test_the_defaults_are_overridable(self, env, store):
        """Every default is an argument, for a store that needs something else."""
        with OpenClient(store, region="eu-central-1", read_timeout=5, retries=1) as s3:
            assert s3.meta.region_name == "eu-central-1"
            assert s3.meta.config.read_timeout == 5
            assert s3.meta.config.retries["total_max_attempts"] == 2  # one retry after the first try


class TestWhatCountsAsConnected:
    """What entering the block accepts as a connection, and what it refuses."""

    def test_bad_credentials_fail_on_entry(self, env, fake_boto):
        """A rejected key pair raises on entry, not at the first read, and the client is closed."""
        client = fake_boto(client_error("InvalidAccessKeyId"))
        with pytest.raises(ConnectionFailedError, match="InvalidAccessKeyId"), OpenClient():
            pass
        assert client.closed, "a rejected client must not be left holding sockets"

    def test_an_unreachable_endpoint_fails_on_entry(self, env, fake_boto):
        """A host that never answers raises on entry, naming the endpoint."""
        fake_boto(EndpointConnectionError(endpoint_url=ENDPOINT))
        with pytest.raises(ConnectionFailedError, match=ENDPOINT), OpenClient():
            pass

    def test_a_scoped_key_that_cannot_list_is_still_open(self, env, fake_boto):
        """AccessDenied on ListBuckets means authenticated, so the block opens."""
        client = fake_boto(client_error("AccessDenied"))
        with OpenClient() as s3:
            assert s3 is client
            assert not client.closed

    def test_check_false_skips_the_round_trip(self, env, fake_boto):
        """A client can be opened against a store that is not up yet."""
        client = fake_boto(EndpointConnectionError(endpoint_url=ENDPOINT))
        with OpenClient(check=False) as s3:
            assert s3 is client


@pytest.mark.integration
def test_opens_a_client_on_the_real_endpoint():
    """Open and close a client on the configured MinIO instance.

    Skipped unless the three S3 variables are set, so the suite stays runnable without them.
    """
    try:
        with OpenClient() as s3:
            assert s3.list_buckets()["ResponseMetadata"]["HTTPStatusCode"] == 200
    except ConnectionFailedError as exc:
        if "not set" in str(exc):
            pytest.skip(str(exc))
        raise
