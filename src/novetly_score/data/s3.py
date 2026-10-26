"""One context manager that opens a client on the experiment's object store and closes it after.

Credentials come from ``S3_ENDPOINT``, ``S3_ACCESS_KEY`` and ``S3_SECRET`` in the repo's
gitignored ``.env``, so the common call takes no arguments::

    with OpenClient() as s3:
        print(s3.list_buckets())

Every default is an argument, so a changed endpoint, a second instance or a different timeout
needs no change here. The block is the lifetime: the client holds pooled sockets, and leaving
the block releases them however it ends.

**Opening is proved, not assumed.** ``boto3`` builds a client offline from any garbage, so a
client that merely exists is not a connection: a typo in ``.env`` would surface at whichever
read came first, far from the line that caused it. The ``with`` makes one ``ListBuckets``
round-trip before handing the client over. An *access denied* answer still counts — the
endpoint resolved, TLS completed and the server verified the signature, so the key is simply
scoped to its own buckets. No answer, or a rejected signature, raises
:class:`ConnectionFailedError`.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import TYPE_CHECKING, Final

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import find_dotenv, load_dotenv

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client  # a type-check-only stub; absent at runtime

#: Environment variable names, in the spelling ``.env`` uses.
ENDPOINT_VAR: Final = "S3_ENDPOINT"
ACCESS_KEY_VAR: Final = "S3_ACCESS_KEY"
SECRET_VAR: Final = "S3_SECRET"  # noqa: S105 — the variable's name, not its value

#: Region sent to MinIO. It has no regions, but SigV4 signs over one, so a value is required.
DEFAULT_REGION: Final = "us-east-1"

#: Error codes meaning the server answered and authenticated us, but scoped the key narrowly.
PERMITTED_DENIALS: Final = frozenset({"AccessDenied", "AllAccessDisabled"})


class ConnectionFailedError(RuntimeError):
    """The store is not configured, could not be reached, or refused the credentials."""


class OpenClient:
    """A verified S3 client, open for the length of a ``with`` block.

        >>> with OpenClient() as s3:                  # doctest: +SKIP
        ...     s3.list_buckets()

    Nothing is opened by constructing one, so the network is reached inside the ``with`` and
    never by a stray reference.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret: str | None = None,
        *,
        region: str = DEFAULT_REGION,
        addressing_style: str = "path",
        connect_timeout: int = 10,
        read_timeout: int = 60,
        retries: int = 3,
        check: bool = True,
    ) -> None:
        """Record how to open the client. Reaches no network.

        Args:
            endpoint: Endpoint URL. Left ``None`` it is read from ``S3_ENDPOINT``.
            access_key: Access key. Left ``None`` it is read from ``S3_ACCESS_KEY``.
            secret: Secret key. Left ``None`` it is read from ``S3_SECRET``.
            region: Region to sign over. MinIO ignores it; SigV4 requires one.
            addressing_style: ``"path"`` for MinIO, whose buckets are not subdomains;
                ``"virtual"`` for AWS proper.
            connect_timeout: Seconds to wait for the connection.
            read_timeout: Seconds to wait for a response.
            retries: Retries after the first attempt. botocore adds the initial try, so 3
                here means at most 4 attempts.
            check: Make one ``ListBuckets`` call on entry. Turn it off only to open against a
                store that is not up yet.
        """
        self._given = (endpoint, access_key, secret)
        self._region = region
        self._check = check
        self._config = Config(
            signature_version="s3v4",
            s3={"addressing_style": addressing_style},
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retries={"max_attempts": retries, "mode": "standard"},
        )
        self._client: S3Client | None = None

    def __enter__(self) -> S3Client:
        """Open the client, prove the store answers, and hand the client over.

        Returns:
            A ``boto3`` S3 client, live until the block ends.

        Raises:
            ConnectionFailedError: If a credential is missing, the endpoint is unreachable, or
                the key pair is rejected.
        """
        import boto3  # deferred: a few tenths of a second, paid only by callers that connect

        endpoint, access_key, secret = _credentials(*self._given)
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret,
            region_name=self._region,
            config=self._config,
        )
        if self._check:
            try:
                _check(client, endpoint)
            except ConnectionFailedError:
                client.close()
                raise
        self._client = client
        return client

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release the client's pooled sockets, whether the block ended normally or by exception.

        Args:
            exc_type: Exception class, if the block raised.
            exc: The exception, if the block raised.
            tb: Its traceback, if the block raised.
        """
        client, self._client = self._client, None
        if client is not None:
            client.close()


def _credentials(
    endpoint: str | None,
    access_key: str | None,
    secret: str | None,
) -> tuple[str, str, str]:
    """Fill the unset arguments from the environment, loading ``.env`` first.

    Variables already set in the environment win over the file, so a shell export or a CI
    secret overrides the checked-out ``.env`` without editing it. The dotenv search is anchored
    to the working directory because ``load_dotenv``'s own default walks up from *this file*,
    which is site-packages once the package is installed rather than editable.

    Args:
        endpoint: Endpoint URL, or ``None`` to read it.
        access_key: Access key, or ``None`` to read it.
        secret: Secret key, or ``None`` to read it.

    Returns:
        ``(endpoint, access_key, secret)``, stripped, the endpoint without a trailing slash.

    Raises:
        ConnectionFailedError: If any of the three is still blank, naming every one that needs
            setting rather than only the first.
    """
    load_dotenv(find_dotenv(usecwd=True), override=False)
    given = {ENDPOINT_VAR: endpoint, ACCESS_KEY_VAR: access_key, SECRET_VAR: secret}
    found = {var: (value if value is not None else os.environ.get(var, "")).strip() for var, value in given.items()}
    missing = [var for var, value in found.items() if not value]
    if missing:
        msg = f"S3 credentials not set: {', '.join(missing)}. Pass them, or set them in the environment or .env"
        raise ConnectionFailedError(msg)
    return found[ENDPOINT_VAR].rstrip("/"), found[ACCESS_KEY_VAR], found[SECRET_VAR]


def _check(client: S3Client, endpoint: str) -> None:
    """Make one round-trip and decide whether it amounts to a connection.

    Args:
        client: The freshly built client.
        endpoint: Endpoint URL, for the error message.

    Raises:
        ConnectionFailedError: If nothing answered, or what answered rejected the key pair.
    """
    try:
        client.list_buckets()
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in PERMITTED_DENIALS:
            return  # authenticated; the key simply may not enumerate buckets
        msg = f"{endpoint} rejected the credentials ({code or 'unknown error'})"
        raise ConnectionFailedError(msg) from exc
    except BotoCoreError as exc:
        msg = f"could not reach {endpoint}: {exc}"
        raise ConnectionFailedError(msg) from exc
