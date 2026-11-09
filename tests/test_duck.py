"""Tests for OpenDuck: how an endpoint URL becomes DuckDB settings, and what counts as connected."""

from __future__ import annotations

import duckdb
import pytest

from novetly_score.data import ConnectionFailedError, OpenDuck
from novetly_score.data.duck import BUCKET, CHECK_PREFIX, _check
from novetly_score.data.s3 import ACCESS_KEY_VAR, ENDPOINT_VAR, SECRET_VAR

ENDPOINT = "https://yl-minio.example.com"


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A clean environment holding valid S3 variables, with no .env within reach."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENDPOINT_VAR, ENDPOINT)
    monkeypatch.setenv(ACCESS_KEY_VAR, "key")
    monkeypatch.setenv(SECRET_VAR, "secret")
    return monkeypatch


class FakeConnection:
    """A DuckDB connection stand-in whose one query raises whatever it was handed."""

    def __init__(self, error):
        """Hold the scripted error, or None to answer the check normally."""
        self.error = error
        self.closed = False
        self.patterns = []

    def sql(self, _query, params=None):
        """Record the pattern asked for, then raise the scripted error or answer emptily."""
        self.patterns.append((params or {}).get("pattern"))
        if self.error is not None:
            raise self.error
        return self

    def fetchall(self):
        """Answer the check with no rows, as a listing of an unused prefix does."""
        return []

    def close(self):
        """Record that the block closed this connection."""
        self.closed = True


@pytest.fixture
def fake_duck(monkeypatch):
    """Install a FakeConnection in place of DuckDB's, so the check meets a scripted answer."""

    def install(error=None):
        connection = FakeConnection(error)
        monkeypatch.setattr(duckdb, "connect", lambda **_kwargs: connection)
        return connection

    return install


class TestTheBlockIsTheLifetime:
    """Construction opens nothing; the with block opens and closes."""

    def test_construction_reaches_no_network(self, env, fake_duck):
        """Building an OpenDuck must not connect, so a stray instance costs nothing."""
        connection = fake_duck()
        OpenDuck()
        assert connection.patterns == []

    def test_the_block_closes_the_connection(self, env, fake_duck):
        """Leaving the block releases the HTTP pool and the temp directory."""
        connection = fake_duck()
        with OpenDuck() as con:
            assert con is connection
        assert connection.closed

    def test_the_block_closes_on_an_exception(self, env, fake_duck):
        """An exception inside the block still releases the connection."""
        connection = fake_duck()
        with pytest.raises(ZeroDivisionError), OpenDuck():
            _ = 1 / 0
        assert connection.closed


class TestTheCheckIsOneCheapListing:
    """What the entry round-trip asks for."""

    def test_it_lists_an_unused_prefix(self, env, fake_duck):
        """The proof is a listing under a prefix no object sits under, not a walk of the bucket."""
        connection = fake_duck()
        with OpenDuck():
            pass
        assert connection.patterns == [f"s3://{BUCKET}/{CHECK_PREFIX}*"]

    def test_check_false_skips_the_round_trip(self, env, fake_duck):
        """A connection can be opened against a store that is not up yet."""
        connection = fake_duck(duckdb.Error("IO Error: Could not resolve hostname"))
        with OpenDuck(check=False) as con:
            assert con is connection
        assert connection.patterns == []


class TestTheEndpointBecomesSettings:
    """One URL in .env becomes DuckDB's host, TLS flag and addressing style."""

    def test_the_scheme_is_split_off(self, env):
        """DuckDB wants the host alone, and TLS as a separate flag."""
        settings = OpenDuck()._settings("https://store.example.com", "key", "secret")
        assert settings["s3_endpoint"] == "store.example.com"
        assert settings["s3_use_ssl"] is True

    def test_plain_http_turns_ssl_off(self, env):
        """A local store on http must not be addressed over TLS."""
        settings = OpenDuck()._settings("http://127.0.0.1:9000", "key", "secret")
        assert settings["s3_endpoint"] == "127.0.0.1:9000"
        assert settings["s3_use_ssl"] is False

    def test_a_bare_host_is_taken_as_the_host(self, env):
        """An endpoint written without a scheme is a host, not a scheme."""
        assert OpenDuck()._settings("store.example.com", "key", "secret")["s3_endpoint"] == "store.example.com"

    def test_path_addressing_is_the_default(self, env):
        """MinIO's buckets are not subdomains, so the default must be path-style."""
        assert OpenDuck()._settings(ENDPOINT, "key", "secret")["s3_url_style"] == "path"

    def test_the_credentials_do_not_pass_through_sql(self, env, fake_duck):
        """The key reaches DuckDB as config, never as a statement it could log."""
        connection = fake_duck()
        with OpenDuck():
            pass
        assert all("secret" not in (pattern or "") for pattern in connection.patterns)

    def test_the_limits_are_omitted_unless_asked_for(self, env):
        """Unset limits must leave DuckDB's own defaults in place rather than pin them."""
        bare = OpenDuck()._settings(ENDPOINT, "key", "secret")
        assert "threads" not in bare
        assert "memory_limit" not in bare
        asked = OpenDuck(threads=4, memory_limit="8GB")._settings(ENDPOINT, "key", "secret")
        assert asked["threads"] == 4
        assert asked["memory_limit"] == "8GB"


class TestWhatCountsAsConnected:
    """What entering the block accepts as a connection, and what it refuses."""

    def test_a_scoped_key_that_cannot_list_is_still_open(self, env, fake_duck):
        """AccessDenied means the signature verified, so the block opens."""
        connection = fake_duck(duckdb.Error("HTTP Error: HTTP GET error (HTTP 403)\nAccessDenied: not allowed"))
        with OpenDuck() as con:
            assert con is connection
        assert connection.closed  # closed by the block's end, not by the check

    def test_a_rejected_signature_fails_on_entry(self, env, fake_duck):
        """403 alone cannot be trusted: a wrong secret wears the same status as a scoped key."""
        connection = fake_duck(duckdb.Error("HTTP Error: (HTTP 403)\nSignatureDoesNotMatch: computed does not match"))
        with pytest.raises(ConnectionFailedError, match="SignatureDoesNotMatch"), OpenDuck():
            pass
        assert connection.closed, "a rejected connection must not be left holding sockets"

    def test_an_unreachable_endpoint_fails_on_entry(self, env, fake_duck):
        """A host that never answers raises on entry, naming the endpoint."""
        fake_duck(duckdb.Error("IO Error: Could not resolve hostname"))
        with pytest.raises(ConnectionFailedError, match=ENDPOINT), OpenDuck():
            pass

    def test_missing_credentials_are_named_before_connecting(self, env):
        """The same credential errors as OpenClient, raised before any engine work."""
        env.delenv(SECRET_VAR)
        with pytest.raises(ConnectionFailedError, match=SECRET_VAR), OpenDuck():
            pass

    def test_the_error_names_the_bucket_and_the_endpoint(self):
        """A failure says which path at which store failed, not merely that one did."""
        connection = FakeConnection(duckdb.Error("IO Error: nope"))
        with pytest.raises(ConnectionFailedError, match=f"s3://{BUCKET} at {ENDPOINT}"):
            _check(connection, BUCKET, ENDPOINT)


@pytest.mark.integration
def test_opens_a_connection_on_the_real_endpoint():
    """Open and close a DuckDB connection on the configured MinIO instance.

    Skipped unless the three S3 variables are set, so the suite stays runnable without them.
    """
    try:
        with OpenDuck() as con:
            assert con.sql("SELECT 42").fetchall() == [(42,)]
    except ConnectionFailedError as exc:
        if "not set" in str(exc):
            pytest.skip(str(exc))
        raise
