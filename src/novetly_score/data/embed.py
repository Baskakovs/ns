"""Embed the snippets with an OpenAI embedding model, resumably, and prove none was skipped.

Three stages, each a command, each safe to re-run::

    uv run python -m novetly_score.data.embed inventory --entities bmw,sap --months 2025-01:2025-03
    uv run python -m novetly_score.data.embed embed     --entities bmw,sap --months 2025-01:2025-03
    uv run python -m novetly_score.data.embed verify

``inventory`` decides *what* to embed, ``embed`` spends the money, ``verify`` proves the result
is complete. Run ``inventory`` first and read what it prints: it reports the exact input count,
token estimate and dollar cost before a single request is made.

**The unit is the distinct snippet text, not the snippet row.** The same text recurs across
articles, sources and companies — 2.78 rows per distinct text in a 720k-row sample — so
embedding rows instead of texts would pay for the same vector 2.8 times and store it 2.8 times.
``inventory`` deduplicates, and carries ``occurrences`` so the collapse is auditable.

**The key is the text's MD5.** The snippet files have no id column, and a positional one would
not survive a re-export. ``md5(snippet)`` is derivable from the data by anyone, in any engine,
for ever, which is what a join key between two separately written datasets has to be.

**Shards are the unit of work and of resume.** A snippet belongs to the shard named by the
first :data:`SHARD_CHARS` characters of its id, which puts the same text in the same shard on
every run. ``embed`` skips a shard that already has vectors and no failures, so a killed run
loses at most one shard, and re-running costs nothing for the shards already paid for.

**Nothing is assumed written.** A permanently failed batch lands in ``failed/`` rather than
vanishing, which forces its shard to be re-done; ``verify`` then anti-joins the inventory
against the vectors inside DuckDB and names the shortfall, so "complete" is a measurement and
not an absence of error messages.

The output sits beside the inputs, keyed by model and width::

    embeddings/snippets/text-embedding-3-small-d256/
        inventory/shard-XX.parquet    snippet_id, snippet, occurrences
        vectors/shard-XX.parquet      snippet_id, embedding
        failed/shard-XX.parquet       snippet_id, error

At 65M distinct snippets and 256 dimensions the vectors are about 67 GB, and the synchronous
API costs roughly $55-85 at $0.02/MTok. OpenAI's Batch API is half that and fits a job of this
shape; this script drives the synchronous API because it can be resumed and verified in minutes
rather than in 24-hour cycles.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import polars as pl

from novetly_score.data.duck import BUCKET, OpenDuck
from novetly_score.data.s3 import OpenClient
from novetly_score.data.snippets import MANIFEST, SNIPPETS_PREFIX, entities

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from duckdb import DuckDBPyConnection

#: The model this script is written for. ``text-embedding-3-large`` works unchanged.
MODEL: Final = "text-embedding-3-small"

#: Output width. The model is Matryoshka-trained, so 256 is a truncation of the same vector and
#: not a different model; it also matches the 256-dim title embeddings already in the bucket.
#: The full 1536 would be 401 GB rather than 67 GB for the same corpus.
DIMENSIONS: Final = 256

#: Characters of the id that name a shard. Two gives 256 shards, so a crash costs at most
#: 1/256th of the run.
SHARD_CHARS: Final = 2

#: Inputs per request. The API accepts 2048; a smaller batch fails smaller and retries cheaper.
BATCH_INPUTS: Final = 512

#: Estimated tokens per request, kept under the API's 300k so an estimate that runs 20% light
#: still fits.
BATCH_TOKENS: Final = 250_000

#: Characters kept per snippet. The limit is 8192 tokens; at worst one token per character this
#: cannot exceed it. Only a handful of snippets reach it — the sample's longest was 9,966.
MAX_CHARS: Final = 8_000

#: Characters per token, for estimating only. Latin text runs near 4, Chinese near 1.5; the
#: real figure comes back with every response and is what the cost report totals.
CHARS_PER_TOKEN: Final = 3.3

#: Dollars per million input tokens, for the estimate the stages print.
PRICE_PER_MTOK: Final = 0.02

#: Rows of inventory held in memory while a shard is embedded.
SHARD_ROWS_WARN: Final = 2_000_000

#: Retries for one batch before it is written off to ``failed/``.
MAX_ATTEMPTS: Final = 6

_SQL_FILE_LIST: Final = """
    SELECT entity, month FROM read_parquet($manifest)
    WHERE rows > 0 AND ($entities IS NULL OR list_contains($entities, entity))
      AND ($first IS NULL OR month >= $first) AND ($last IS NULL OR month <= $last)
    ORDER BY entity, month
"""

_SQL_INVENTORY: Final = """
    CREATE OR REPLACE TABLE inventory AS
    SELECT md5(snippet) AS snippet_id, any_value(snippet) AS snippet, count(*) AS occurrences
    FROM read_parquet($paths) WHERE snippet IS NOT NULL AND length(snippet) > 0
    GROUP BY md5(snippet)
"""

_SQL_INVENTORY_STATS: Final = """
    SELECT count(*) AS texts, sum(occurrences) AS rows,
           sum(length(snippet)) AS chars, max(length(snippet)) AS max_chars
    FROM inventory
"""

_SQL_SHARD: Final = """
    SELECT snippet_id, snippet, occurrences FROM inventory
    WHERE substr(snippet_id, 1, $width) = $shard ORDER BY snippet_id
"""

_SQL_SHARD_NAMES: Final = """
    SELECT DISTINCT substr(snippet_id, 1, $width) AS shard FROM inventory ORDER BY shard
"""

_SQL_MISSING: Final = """
    SELECT count(*) AS missing FROM read_parquet($inventory) AS i
    WHERE NOT EXISTS (SELECT 1 FROM read_parquet($vectors) AS v WHERE v.snippet_id = i.snippet_id)
"""

_SQL_TOTALS: Final = """
    SELECT (SELECT count(*) FROM read_parquet($inventory)) AS wanted,
           (SELECT count(*) FROM read_parquet($vectors)) AS written,
           (SELECT count(DISTINCT snippet_id) FROM read_parquet($vectors)) AS distinct_written
"""

_SQL_VECTOR_HEALTH: Final = """
    SELECT count(*) AS bad_dim FROM read_parquet($vectors) WHERE len(embedding) <> $dimensions
"""

_SQL_VECTOR_NORMS: Final = """
    SELECT count(*) AS degenerate FROM (
        SELECT list_reduce(list_transform(embedding, lambda x: x * x), lambda a, b: a + b) AS sq
        FROM read_parquet($vectors)
    ) WHERE sq IS NULL OR sq < 0.5 OR sq > 2.0
"""


class EmbeddingRunError(RuntimeError):
    """The run cannot proceed, or finished incomplete."""


@dataclass(frozen=True)
class Layout:
    """Where one (model, width) run's files live.

    Keyed by both, so a re-run at another width or on another model lands beside the first
    rather than on top of it.
    """

    model: str = MODEL
    dimensions: int = DIMENSIONS
    bucket: str = BUCKET

    @property
    def prefix(self) -> str:
        """The run's own prefix under the bucket.

        Returns:
            A key prefix, e.g. ``embeddings/snippets/text-embedding-3-small-d256``.
        """
        return f"embeddings/snippets/{self.model}-d{self.dimensions}"

    def key(self, stage: str, shard: str) -> str:
        """Name one shard's object key.

        Args:
            stage: ``"inventory"``, ``"vectors"`` or ``"failed"``.
            shard: The shard's hex name.

        Returns:
            The object key.
        """
        return f"{self.prefix}/{stage}/shard-{shard}.parquet"

    def glob(self, stage: str) -> str:
        """Name every shard of one stage, as DuckDB reads them.

        Args:
            stage: ``"inventory"``, ``"vectors"`` or ``"failed"``.

        Returns:
            An ``s3://`` glob.
        """
        return f"s3://{self.bucket}/{self.prefix}/{stage}/*.parquet"

    def url(self, stage: str, shard: str) -> str:
        """Name one shard as a URL.

        Args:
            stage: ``"inventory"``, ``"vectors"`` or ``"failed"``.
            shard: The shard's hex name.

        Returns:
            An ``s3://`` URL.
        """
        return f"s3://{self.bucket}/{self.key(stage, shard)}"


class Limiter:
    """A token bucket over requests and tokens per minute, shared by every worker.

    The API's own 429 is a last resort: it costs a round-trip and, under load, a retry storm.
    Pacing to the account's published limits means the common case never reaches it.
    """

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Fill the bucket.

        Args:
            requests_per_minute: Requests allowed per minute.
            tokens_per_minute: Input tokens allowed per minute.
            clock: Source of monotonic time. An argument so the pacing can be driven by a test
                in microseconds rather than in minutes.
            sleep: How to wait, for the same reason.
        """
        self._rpm = max(1, requests_per_minute)
        self._tpm = max(1, tokens_per_minute)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._requests = 0.0
        self._tokens = 0.0
        self._stamp = clock()

    def take(self, tokens: int) -> None:
        """Block until one request of *tokens* tokens is within both limits.

        Args:
            tokens: Estimated input tokens of the request.
        """
        while True:
            with self._lock:
                now = self._clock()
                elapsed = now - self._stamp
                self._stamp = now
                self._requests = max(0.0, self._requests - elapsed * self._rpm / 60.0)
                self._tokens = max(0.0, self._tokens - elapsed * self._tpm / 60.0)
                if self._requests < self._rpm and self._tokens < self._tpm:
                    self._requests += 1
                    self._tokens += tokens
                    return
                wait = max(
                    (self._requests - self._rpm + 1) * 60.0 / self._rpm,
                    (self._tokens - self._tpm + tokens) * 60.0 / self._tpm,
                )
            self._sleep(min(max(wait, 0.05), 5.0))


def tokens_of(text: str) -> int:
    """Estimate one input's tokens from its length.

    Args:
        text: The snippet.

    Returns:
        An estimate, never below one.
    """
    return max(1, int(len(text) / CHARS_PER_TOKEN) + 1)


def batches(frame: pl.DataFrame) -> Iterator[pl.DataFrame]:
    """Cut a shard into requests that respect both API limits.

    Args:
        frame: Inventory rows, with ``snippet_id`` and ``snippet``.

    Yields:
        Slices of *frame*, each within :data:`BATCH_INPUTS` and :data:`BATCH_TOKENS`.
    """
    start, tokens = 0, 0
    for index, text in enumerate(frame.get_column("snippet")):
        cost = tokens_of(text)
        if index > start and (index - start >= BATCH_INPUTS or tokens + cost > BATCH_TOKENS):
            yield frame[start:index]
            start, tokens = index, 0
        tokens += cost
    if start < frame.height:
        yield frame[start:]


def _client(api_key: str | None) -> Any:  # the SDK's type is only importable at runtime
    """Build an OpenAI client, retrying nothing the script retries itself.

    Args:
        api_key: The key, or ``None`` to read ``OPENAI_API_KEY``.

    Returns:
        An ``openai.OpenAI`` client.

    Raises:
        EmbeddingRunError: If the SDK is missing or no key is set.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover — declared in pyproject
        msg = "the openai package is not installed: uv sync"
        raise EmbeddingRunError(msg) from exc
    key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not key:
        msg = "OPENAI_API_KEY is not set. Put it in .env, or pass --api-key"
        raise EmbeddingRunError(msg)
    return OpenAI(api_key=key, max_retries=0, timeout=120.0)


def embed_batch(
    client: Any,  # an SDK type, see _client
    batch: pl.DataFrame,
    layout: Layout,
    limiter: Limiter,
) -> tuple[list[list[float]], int]:
    """Embed one batch, retrying on the failures that are worth retrying.

    Args:
        client: The OpenAI client.
        batch: Inventory rows for this request.
        layout: The run's model and width.
        limiter: The shared rate limiter.

    Returns:
        ``(vectors, tokens)``: one vector per row, in row order, and the tokens charged.

    Raises:
        EmbeddingRunError: If every attempt failed, or the API returned the wrong row count.
    """
    inputs = [text[:MAX_CHARS] for text in batch.get_column("snippet")]
    estimate = sum(tokens_of(text) for text in inputs)
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        limiter.take(estimate)
        try:
            answer = client.embeddings.create(model=layout.model, dimensions=layout.dimensions, input=inputs)
        except Exception as exc:  # noqa: BLE001 — the SDK raises a family; the retry is the same
            last = exc
            if not _retryable(exc) or attempt == MAX_ATTEMPTS - 1:
                break
            time.sleep(min(2.0**attempt, 60.0) * (0.5 + 0.5 * ((attempt * 7919) % 100) / 100.0))
            continue
        vectors = [item.embedding for item in sorted(answer.data, key=lambda item: item.index)]
        if len(vectors) != len(inputs):
            msg = f"asked for {len(inputs)} embeddings, got {len(vectors)}"
            raise EmbeddingRunError(msg)
        return vectors, answer.usage.total_tokens
    msg = f"batch of {len(inputs)} failed after {MAX_ATTEMPTS} attempts: {last}"
    raise EmbeddingRunError(msg)


def _retryable(exc: Exception) -> bool:
    """Say whether an API failure is worth another attempt.

    A 429 or a 5xx is weather; a 400 means the batch itself is wrong and will fail again.

    Args:
        exc: The exception the SDK raised.

    Returns:
        ``True`` if the batch should be retried.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        return True  # a timeout or a dropped connection carries no status
    return bool(status == 429 or status >= 500)


def file_list(con: DuckDBPyConnection, names: Sequence[str] | None, first: str | None, last: str | None) -> list[str]:
    """Name the snippet files a run covers, from the manifest rather than a listing.

    Args:
        con: An open connection.
        names: Snippet directory names, or ``None`` for every company in the store.
        first: Earliest month, ``"YYYY-MM"``, or ``None``.
        last: Latest month, ``"YYYY-MM"``, or ``None``.

    Returns:
        ``s3://`` paths, in entity then month order.

    Raises:
        EmbeddingRunError: If the selection names no file.
    """
    rows = con.sql(
        _SQL_FILE_LIST,
        params={"manifest": MANIFEST, "entities": list(names) if names else None, "first": first, "last": last},
    ).fetchall()
    if not rows:
        msg = "that selection of companies and months holds no snippet file"
        raise EmbeddingRunError(msg)
    return [f"s3://{BUCKET}/{SNIPPETS_PREFIX}/{entity}/{month}.parquet" for entity, month in rows]


def upload(frame: pl.DataFrame, key: str, bucket: str = BUCKET) -> int:
    """Write one frame to the store as parquet, whole or not at all.

    The parquet is built locally and then put in one call, so a shard's object never exists
    half-written — which is what lets its presence stand for "this shard is done".

    Args:
        frame: The rows to write.
        key: The object key to write them to.
        bucket: The bucket.

    Returns:
        The bytes written.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shard.parquet"
        frame.write_parquet(path, compression="zstd")
        size = path.stat().st_size
        with OpenClient() as s3:
            s3.upload_file(str(path), bucket, key)
    return size


def existing(stage: str, layout: Layout) -> set[str]:
    """Name the shards already written for one stage.

    Args:
        stage: ``"inventory"``, ``"vectors"`` or ``"failed"``.
        layout: The run's layout.

    Returns:
        Shard names, e.g. ``{"00", "01"}``.
    """
    prefix = f"{layout.prefix}/{stage}/shard-"
    found: set[str] = set()
    with OpenClient() as s3:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=layout.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                found.add(item["Key"].removeprefix(prefix).removesuffix(".parquet"))
    return found


def inventory(
    names: Sequence[str] | None,
    first: str | None,
    last: str | None,
    layout: Layout,
    *,
    memory_limit: str | None = None,
    temp_directory: str | None = None,
) -> dict[str, int]:
    """Decide what to embed: every distinct snippet text in the selection, with its id.

    Costs no API call. Read what it prints before running :func:`embed` — the token estimate
    and the dollar figure are the bill, and they are only known once the duplicates are gone.

    Args:
        names: Snippet directory names, or ``None`` for every company.
        first: Earliest month, or ``None``.
        last: Latest month, or ``None``.
        layout: Where to write.
        memory_limit: DuckDB buffer pool. The deduplication spills past it rather than failing.
        temp_directory: Where that spill goes.

    Returns:
        ``{"texts": …, "rows": …, "shards": …, "tokens": …}``.
    """
    with OpenDuck(memory_limit=memory_limit, temp_directory=temp_directory) as con:
        paths = file_list(con, names, first, last)
        print(f"reading {len(paths):,} snippet files", flush=True)
        started = time.monotonic()
        con.sql(_SQL_INVENTORY, params={"paths": paths}).fetchall()
        texts, rows, chars, longest = con.sql(_SQL_INVENTORY_STATS).fetchall()[0]
        estimate = int(chars / CHARS_PER_TOKEN) + texts
        print(
            f"{rows:,} snippet rows -> {texts:,} distinct texts "
            f"({rows / max(texts, 1):.2f}x), longest {longest:,} chars, "
            f"deduplicated in {time.monotonic() - started:.0f}s",
            flush=True,
        )
        print(
            f"~{estimate / 1e9:.2f}B tokens at {CHARS_PER_TOKEN} chars/token "
            f"-> ~${estimate / 1e6 * PRICE_PER_MTOK:,.2f} on {layout.model}; "
            f"vectors ~{texts * layout.dimensions * 4 / 1e9:,.0f} GB at {layout.dimensions} dims",
            flush=True,
        )
        shards = [row[0] for row in con.sql(_SQL_SHARD_NAMES, params={"width": SHARD_CHARS}).fetchall()]
        written = 0
        for index, shard in enumerate(shards, start=1):
            frame = pl.DataFrame(con.sql(_SQL_SHARD, params={"width": SHARD_CHARS, "shard": shard}))
            written += upload(frame, layout.key("inventory", shard), layout.bucket)
            print(f"  inventory shard {shard} ({index}/{len(shards)}): {frame.height:,} texts", flush=True)
    print(f"wrote {len(shards)} inventory shards, {written / 1e6:.0f} MB, to {layout.prefix}/inventory", flush=True)
    return {"texts": texts, "rows": rows, "shards": len(shards), "tokens": estimate}


def embed(
    layout: Layout,
    *,
    api_key: str | None = None,
    workers: int = 8,
    requests_per_minute: int = 3_000,
    tokens_per_minute: int = 1_000_000,
    shards: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Embed every inventory shard that is not already done, and write the vectors.

    A shard with vectors and no failures is skipped untouched, so this is the command to re-run
    after a crash, a rate-limit wall or a laptop lid. A shard that is partly done is finished
    rather than redone: its existing vectors are read back and only the rest is paid for.

    Args:
        layout: The run's layout, naming the model and the width.
        api_key: The OpenAI key, or ``None`` to read ``OPENAI_API_KEY``.
        workers: Concurrent requests.
        requests_per_minute: Pace to the account's request limit.
        tokens_per_minute: Pace to the account's token limit.
        shards: Only these shards, or ``None`` for all of them.
        dry_run: Report what would be embedded and stop before the first request.

    Returns:
        ``{"shards": …, "embedded": …, "skipped": …, "failed": …, "tokens": …}``.

    Raises:
        EmbeddingRunError: If no inventory has been written yet.
    """
    todo = sorted(existing("inventory", layout))
    if not todo:
        msg = f"no inventory at {layout.prefix}/inventory — run the inventory command first"
        raise EmbeddingRunError(msg)
    done, failed_before = existing("vectors", layout), existing("failed", layout)
    if shards is not None:
        todo = [shard for shard in todo if shard in set(shards)]
    pending = [shard for shard in todo if shard not in done or shard in failed_before]
    print(
        f"{len(todo)} shards selected: {len(todo) - len(pending)} already done, {len(pending)} to do",
        flush=True,
    )
    if dry_run or not pending:
        return {"shards": len(pending), "embedded": 0, "skipped": len(todo) - len(pending), "failed": 0, "tokens": 0}

    client = _client(api_key)
    limiter = Limiter(requests_per_minute, tokens_per_minute)
    totals = {"shards": 0, "embedded": 0, "skipped": len(todo) - len(pending), "failed": 0, "tokens": 0}
    for index, shard in enumerate(pending, start=1):
        counts = _embed_shard(client, limiter, layout, shard, resume=shard in done, workers=workers)
        for name, value in counts.items():
            totals[name] = totals.get(name, 0) + value
        totals["shards"] += 1
        spent = totals["tokens"] / 1e6 * PRICE_PER_MTOK
        print(
            f"shard {shard} ({index}/{len(pending)}): +{counts['embedded']:,} vectors, "
            f"{counts['failed']:,} failed, ${spent:,.2f} spent so far",
            flush=True,
        )
    return totals


def _embed_shard(
    client: Any,  # an SDK type, importable only at runtime; see _client
    limiter: Limiter,
    layout: Layout,
    shard: str,
    *,
    resume: bool,
    workers: int,
) -> dict[str, int]:
    """Embed one shard and write it, whole, with any permanent failures recorded beside it.

    Args:
        client: The OpenAI client.
        limiter: The shared rate limiter.
        layout: The run's layout.
        shard: The shard's name.
        resume: Whether vectors for part of this shard already exist.
        workers: Concurrent requests.

    Returns:
        ``{"embedded": …, "failed": …, "tokens": …}``.
    """
    with OpenDuck() as con:
        wanted = pl.DataFrame(
            con.sql("SELECT * FROM read_parquet($path)", params={"path": layout.url("inventory", shard)})
        )
        kept = (
            pl.DataFrame(con.sql("SELECT * FROM read_parquet($path)", params={"path": layout.url("vectors", shard)}))
            if resume
            else pl.DataFrame(schema={"snippet_id": pl.String, "embedding": pl.List(pl.Float32)})
        )
    missing = wanted.join(kept.select("snippet_id"), on="snippet_id", how="anti")
    if missing.height > SHARD_ROWS_WARN:
        print(f"  shard {shard} holds {missing.height:,} texts; consider a larger SHARD_CHARS", flush=True)
    if missing.is_empty():
        return {"embedded": 0, "failed": 0, "tokens": 0}

    vectors: list[pl.DataFrame] = [kept] if kept.height else []
    failures: list[dict[str, str]] = []
    tokens = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        work = [(batch, pool.submit(embed_batch, client, batch, layout, limiter)) for batch in batches(missing)]
        for batch, future in work:
            try:
                answer, charged = future.result()
            except EmbeddingRunError as exc:
                failures.extend({"snippet_id": key, "error": str(exc)[:500]} for key in batch.get_column("snippet_id"))
                continue
            tokens += charged
            vectors.append(
                pl.DataFrame(
                    {"snippet_id": batch.get_column("snippet_id"), "embedding": answer},
                    schema={"snippet_id": pl.String, "embedding": pl.List(pl.Float32)},
                )
            )
    if vectors:
        upload(pl.concat(vectors, how="vertical"), layout.key("vectors", shard), layout.bucket)
    if failures:
        upload(pl.DataFrame(failures), layout.key("failed", shard), layout.bucket)
    elif resume or shard in existing("failed", layout):
        _forget(layout.key("failed", shard), layout.bucket)
    embedded = sum(frame.height for frame in vectors) - kept.height
    return {"embedded": embedded, "failed": len(failures), "tokens": tokens}


def _forget(key: str, bucket: str) -> None:
    """Drop a failure record that a later run made untrue.

    Args:
        key: The object key.
        bucket: The bucket.
    """
    with OpenClient() as s3:
        s3.delete_object(Bucket=bucket, Key=key)


def verify(layout: Layout) -> dict[str, int]:
    """Prove every inventoried snippet has exactly one sound vector, or name the shortfall.

    The check that matters is the anti-join, which runs inside DuckDB over the parquet rather
    than in a set of 65 million strings in Python. The vector health checks catch the other way
    a run can look finished and be wrong: a truncated width, or an all-zero row.

    Args:
        layout: The run's layout.

    Returns:
        ``{"wanted": …, "written": …, "missing": …, "duplicated": …, "bad_dim": …,
        "degenerate": …, "failed": …}``.

    Raises:
        EmbeddingRunError: If no inventory or no vectors have been written.
    """
    if not existing("inventory", layout):
        msg = f"no inventory at {layout.prefix}/inventory"
        raise EmbeddingRunError(msg)
    if not existing("vectors", layout):
        msg = f"no vectors at {layout.prefix}/vectors — run the embed command"
        raise EmbeddingRunError(msg)
    paths = {"inventory": layout.glob("inventory"), "vectors": layout.glob("vectors")}
    with OpenDuck() as con:
        wanted, written, distinct = con.sql(_SQL_TOTALS, params=paths).fetchall()[0]
        (missing,) = con.sql(_SQL_MISSING, params=paths).fetchall()[0]
        vectors = {"vectors": paths["vectors"]}
        (bad_dim,) = con.sql(_SQL_VECTOR_HEALTH, params={**vectors, "dimensions": layout.dimensions}).fetchall()[0]
        (degenerate,) = con.sql(_SQL_VECTOR_NORMS, params=vectors).fetchall()[0]
    failed = len(existing("failed", layout))
    report = {
        "wanted": wanted,
        "written": written,
        "missing": missing,
        "duplicated": written - distinct,
        "bad_dim": bad_dim,
        "degenerate": degenerate,
        "failed": failed,
    }
    for name, value in report.items():
        print(f"  {name:<12} {value:,}", flush=True)
    if missing or report["duplicated"] or bad_dim or degenerate or failed:
        msg = (
            f"incomplete: {missing:,} snippets without a vector, {report['duplicated']:,} duplicated, "
            f"{bad_dim:,} of the wrong width, {degenerate:,} degenerate, {failed:,} shards with failures. "
            f"Re-run the embed command; it will redo only what is missing"
        )
        raise EmbeddingRunError(msg)
    print(f"complete: {written:,} vectors, one per distinct snippet, all {layout.dimensions} dims", flush=True)
    return report


def _months(span: str | None) -> tuple[str | None, str | None]:
    """Split a ``FIRST:LAST`` month span.

    Args:
        span: ``"2025-01:2025-03"``, ``"2025-01"`` for one month, ``"2025-01:"`` for open-ended,
            or ``None`` for every month.

    Returns:
        ``(first, last)``.
    """
    if not span:
        return None, None
    first, _, last = span.partition(":")
    return (first.strip() or None), ((last.strip() or None) if _ else first.strip() or None)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one stage from the command line.

    Args:
        argv: Arguments, or ``None`` for ``sys.argv``.

    Returns:
        ``0`` if the stage succeeded, ``1`` if it reported the run incomplete.
    """
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)

    parser = argparse.ArgumentParser(prog="novetly_score.data.embed", description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL, help=f"embedding model (default {MODEL})")
    parser.add_argument("--dimensions", type=int, default=DIMENSIONS, help=f"output width (default {DIMENSIONS})")
    stages = parser.add_subparsers(dest="stage", required=True)

    inv = stages.add_parser("inventory", help="decide what to embed; costs no API call")
    inv.add_argument("--entities", help="comma-separated company names (default: every company)")
    inv.add_argument("--months", help="FIRST:LAST, e.g. 2025-01:2025-03 (default: every month)")
    inv.add_argument("--memory-limit", default="8GB", help="DuckDB buffer pool (default 8GB)")
    inv.add_argument("--temp-directory", default=None, help="where DuckDB spills the deduplication")

    # embed takes no company or month argument: the inventory it reads *is* the selection,
    # so a second one here could disagree with what was costed and written.
    run = stages.add_parser("embed", help="embed every inventory shard not already done")
    run.add_argument("--api-key", default=None, help="OpenAI key (default: OPENAI_API_KEY)")
    run.add_argument("--workers", type=int, default=8, help="concurrent requests (default 8)")
    run.add_argument("--rpm", type=int, default=3_000, help="requests per minute to pace to (default 3000)")
    run.add_argument("--tpm", type=int, default=1_000_000, help="tokens per minute to pace to (default 1000000)")
    run.add_argument("--shards", default=None, help="comma-separated shard names, for one slice")
    run.add_argument("--dry-run", action="store_true", help="report what is pending and stop")

    stages.add_parser("verify", help="prove every inventoried snippet has one sound vector")

    args = parser.parse_args(argv)
    layout = Layout(model=args.model, dimensions=args.dimensions)
    try:
        if args.stage == "inventory":
            first, last = _months(args.months)
            inventory(
                _companies(args.entities),
                first,
                last,
                layout,
                memory_limit=args.memory_limit,
                temp_directory=args.temp_directory,
            )
        elif args.stage == "embed":
            embed(
                layout,
                api_key=args.api_key,
                workers=args.workers,
                requests_per_minute=args.rpm,
                tokens_per_minute=args.tpm,
                shards=_split(args.shards),
                dry_run=args.dry_run,
            )
        else:
            verify(layout)
    except EmbeddingRunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _split(value: str | None) -> list[str] | None:
    """Split a comma-separated argument.

    Args:
        value: The argument, or ``None``.

    Returns:
        The items, or ``None`` if there were none.
    """
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _companies(value: str | None) -> list[str] | None:
    """Resolve a comma-separated company argument to snippet directory names.

    Goes through the snippets layer, so ``company:bmw``, ``bmw`` and a YUKKA id mix freely.

    Args:
        value: The argument, or ``None`` for every company.

    Returns:
        Directory names, or ``None``.
    """
    items = _split(value)
    return list(entities(items)) if items else None


if __name__ == "__main__":  # pragma: no cover — the command line
    raise SystemExit(main())
