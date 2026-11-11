"""One context manager that opens a DuckDB connection onto the object store and closes it after.

DuckDB reads the parquet in place: it fetches a file's footer, then only the row groups and
columns a query asks for, so a filtered read costs a fraction of the object. Credentials are
the same three variables :mod:`novetly_score.data.s3` uses, read by the same function, so the
store is configured in exactly one place::

    with OpenDuck() as con:
        con.sql("SELECT count(*) FROM read_parquet($p)", params={"p": path})

The block is the lifetime, as with :class:`~novetly_score.data.s3.OpenClient`: the connection
holds an HTTP pool and a temp directory for spilled data, and leaving the block releases both.

**Opening is proved, not assumed.** ``duckdb.connect`` touches no network, so a connection that
exists is not a connection to anything: a wrong endpoint would surface at whichever query came
first. The ``with`` makes one listing before handing the connection over, against a prefix held
by no object, so the proof costs a round-trip and not a walk of the bucket. An *access denied*
answer still counts, on the reasoning :class:`~novetly_score.data.s3.OpenClient` gives and by
the same list of codes. No answer, or a rejected signature, raises
:class:`~novetly_score.data.s3.ConnectionFailedError`.

Status alone cannot tell those apart: MinIO answers both a scoped key and a wrong secret with
*403*, so the check reads the S3 error code DuckDB quotes in its message, not the status.

Credentials never pass through SQL. ``CREATE SECRET`` would interpolate the key into a statement
that DuckDB keeps in its query log; the settings go into the connection's config dict instead.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Final, TypeAlias

from novetly_score.data.s3 import DEFAULT_REGION, PERMITTED_DENIALS, ConnectionFailedError, _credentials

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

#: What ``duckdb.connect`` accepts as a config value. Spelled out rather than widened to
#: ``object`` because a dict's value type is invariant: a narrower mapping is not assignable
#: to the parameter, so the alias has to match the signature it is passed to.
Setting: TypeAlias = "str | bool | int | float | list[str]"

#: Bucket holding the experiment's data.
BUCKET: Final = "quant-experiments"

#: Prefix the entry check lists. No object is expected under it: a listing that answers with
#: nothing proves the endpoint and the signature just as well as one that answers with keys,
#: and costs one request instead of a walk of 20,000 of them. What it does not prove is the
#: layout — a listing under a bucket that does not exist comes back empty rather than 404 — so
#: a missing bucket or a moved prefix surfaces at the first read, naming the path it wanted.
CHECK_PREFIX: Final = ".connection-check"


class OpenDuck:
    """A verified DuckDB connection onto the store, open for the length of a ``with`` block.

        >>> with OpenDuck() as con:                   # doctest: +SKIP
        ...     con.sql("SELECT 1").fetchone()

    Nothing is opened by constructing one, so the network is reached inside the ``with``.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret: str | None = None,
        *,
        region: str = DEFAULT_REGION,
        bucket: str = BUCKET,
        threads: int | None = None,
        memory_limit: str | None = None,
        temp_directory: str | None = None,
        check: bool = True,
    ) -> None:
        """Record how to open the connection. Reaches no network.

        Args:
            endpoint: Endpoint URL, with its scheme. Left ``None`` it is read from ``S3_ENDPOINT``.
            access_key: Access key. Left ``None`` it is read from ``S3_ACCESS_KEY``.
            secret: Secret key. Left ``None`` it is read from ``S3_SECRET``.
            region: Region to sign over. MinIO ignores it; SigV4 requires one.
            bucket: Bucket the entry check lists.
            threads: Worker threads. Left ``None`` DuckDB uses one per core.
            memory_limit: Buffer-pool size, e.g. ``"8GB"``. Left ``None`` DuckDB takes 80% of RAM.
                Queries over the larger files spill to disk rather than fail when it is reached.
            temp_directory: Where to spill. Worth setting for an aggregation over the whole
                corpus, whose spill is larger than the default location may hold.
            check: List the bucket once on entry. Turn it off only to open against a store that
                is not up yet.
        """
        self._given = (endpoint, access_key, secret)
        self._region = region
        self._bucket = bucket
        self._threads = threads
        self._memory_limit = memory_limit
        self._temp_directory = temp_directory
        self._check = check
        self._connection: DuckDBPyConnection | None = None

    def __enter__(self) -> DuckDBPyConnection:
        """Open the connection, prove the store answers, and hand the connection over.

        Returns:
            A DuckDB connection configured for the store, live until the block ends.

        Raises:
            ConnectionFailedError: If a credential is missing, the endpoint is unreachable, or
                the key pair is rejected.
        """
        import duckdb  # deferred: loading the engine costs a fraction of a second

        endpoint, access_key, secret = _credentials(*self._given)
        settings = self._settings(endpoint, access_key, secret)
        try:
            connection = duckdb.connect(config=settings)
        except duckdb.InvalidInputException:
            # Setting s3_endpoint needs httpfs, which DuckDB autoloads by fetching it from its
            # own CDN the first time. That fetch fails intermittently on CI runners, and the
            # connection is refused outright. Install it deliberately, with retries, and try
            # again -- the download is cached per user, so this is paid once and only on failure.
            _install_httpfs()
            connection = duckdb.connect(config=settings)
        if self._check:
            try:
                _check(connection, self._bucket, endpoint)
            except ConnectionFailedError:
                connection.close()
                raise
        self._connection = connection
        return connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release the connection's HTTP pool and temporary files, however the block ended.

        Args:
            exc_type: Exception class, if the block raised.
            exc: The exception, if the block raised.
            tb: Its traceback, if the block raised.
        """
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def _settings(self, endpoint: str, access_key: str, secret: str) -> dict[str, Setting]:
        """Translate an endpoint URL and a key pair into DuckDB's S3 settings.

        DuckDB wants the host without its scheme and a separate flag for TLS, so the one URL in
        ``.env`` becomes two settings here. ``s3_url_style`` is ``path`` for the same reason
        :class:`~novetly_score.data.s3.OpenClient` uses path addressing: MinIO's buckets are not
        subdomains.

        Args:
            endpoint: Endpoint URL, with its scheme.
            access_key: Access key.
            secret: Secret key.

        Returns:
            A config dict for ``duckdb.connect``.
        """
        scheme, _, host = endpoint.partition("://")
        settings: dict[str, Setting] = {
            "s3_endpoint": host or scheme,  # no scheme given: the whole string is the host
            "s3_access_key_id": access_key,
            "s3_secret_access_key": secret,
            "s3_region": self._region,
            "s3_url_style": "path",
            "s3_use_ssl": scheme != "http",
        }
        if self._threads is not None:
            settings["threads"] = self._threads
        if self._memory_limit is not None:
            settings["memory_limit"] = self._memory_limit
        if self._temp_directory is not None:
            settings["temp_directory"] = self._temp_directory
        return settings


def _install_httpfs(attempts: int = 3, pause: float = 1.0) -> None:
    """Fetch and load the ``httpfs`` extension, retrying a failed download.

    DuckDB will not accept an ``s3_endpoint`` setting without this extension, and autoloads it
    from its CDN the first time one is asked for. That download is the single flakiest thing in
    this layer's CI -- it has refused a connection on Windows runners twice -- so it is worth
    asking for deliberately rather than leaving to an autoload that gets one attempt.

    Installing writes to DuckDB's own extension directory, so a later connection autoloads it
    from local disk and touches no network. Credentials are not involved: this runs on a plain
    in-memory connection, which is why the S3 settings can stay out of SQL entirely.

    Args:
        attempts: How many times to try the download before giving up.
        pause: Seconds to wait between attempts.

    Raises:
        ConnectionFailedError: If every attempt failed, naming the last error.
    """
    import time

    import duckdb

    last: Exception | None = None
    for attempt in range(attempts):
        probe = duckdb.connect()
        try:
            probe.install_extension("httpfs")
            probe.load_extension("httpfs")
        except duckdb.Error as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(pause)
        else:
            return
        finally:
            probe.close()
    msg = f"could not install DuckDB's httpfs extension after {attempts} attempts: {last}"
    raise ConnectionFailedError(msg)


def _check(connection: DuckDBPyConnection, bucket: str, endpoint: str) -> None:
    """Make one listing and decide whether it amounts to a connection.

    Args:
        connection: The freshly opened connection.
        bucket: Bucket to list, under :data:`CHECK_PREFIX`.
        endpoint: Endpoint URL, for the error message.

    Raises:
        ConnectionFailedError: If nothing answered, or what answered rejected the key pair.
    """
    import duckdb

    pattern = f"s3://{bucket}/{CHECK_PREFIX}*"
    try:
        connection.sql("SELECT 1 FROM glob($pattern)", params={"pattern": pattern}).fetchall()
    except duckdb.Error as exc:
        message = str(exc)
        if any(code in message for code in PERMITTED_DENIALS):
            return  # authenticated; the key simply may not list this bucket
        msg = f"could not read s3://{bucket} at {endpoint} with DuckDB: {message}"
        raise ConnectionFailedError(msg) from exc
