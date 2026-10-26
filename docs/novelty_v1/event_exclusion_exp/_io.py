"""Token lookup, universe batching and atomic shard writes for this experiment's sweeps.

Deliberately self-contained. The equivalents in ``novelty_score_exp/sentiment_sweep.py`` are the
originals and this experiment's sweep is modelled on them, but importing across experiments would
mean an edit to that sweep silently changing this one's batch partition — and the partition is
exactly what makes the two dumps joinable. A few duplicated helpers are cheaper than that coupling.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import httpx
import polars as pl

#: NLP version our dumps, embeddings and production counts are all built on. Never send anything else.
#: The API default is 9-4-0, which is a different source universe rather than a newer model, and its
#: ``news_id`` values would not join to the sentiment dump.
NLP_VERSION = "9-3-5"

#: The API's timestamp format — fixed-width, no zone marker, UTC by convention.
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Retry schedule in seconds for transport errors and 5xx. A 401 is never retried.
BACKOFF = (2, 8, 32)


class AuthExpiredError(RuntimeError):
    """Raised on a 401 so the whole sweep aborts instead of writing empty shards."""


def read_token(env_path: str = ".env") -> str:
    """Return the Yukka bearer token from the environment, falling back to a `.env` file.

    The ``.env`` fallback is what lets these modules run as scripts: invoked as
    ``uv run src/.../events_sweep.py`` there is no ``honey`` import, so no ``load_dotenv``.
    """
    tok = os.environ.get("YUKKA_TOKEN")
    if tok:
        return tok.strip()
    for line in Path(env_path).read_text().splitlines():
        if line.startswith("YUKKA_TOKEN"):
            return line.split("=", 1)[1].strip().strip("\"'")
    msg = "YUKKA_TOKEN not found in the environment or .env"
    raise RuntimeError(msg)


def auth_headers() -> dict[str, str]:
    """Bearer header for either Yukka API."""
    return {"Authorization": f"Bearer {read_token()}"}


def load_universe(path: str) -> list[str]:
    """Return the sorted distinct `yukka_id` universe. Sorted so batch membership is reproducible."""
    ids = pl.read_parquet(path)["yukka_id"].drop_nulls().unique().sort().to_list()
    return [str(i) for i in ids]


def batches(universe: list[str], size: int) -> list[list[str]]:
    """Chunk the universe into fixed-size entity batches."""
    return [universe[i : i + size] for i in range(0, len(universe), size)]


def check_response(r: httpx.Response) -> None:
    """Turn an HTTP status into the right exception: 401 aborts the sweep, 5xx retries."""
    if r.status_code == 401:
        msg = "401 from the API — token expired or invalid; aborting rather than writing empty shards"
        raise AuthExpiredError(msg)
    if r.status_code >= 500:
        msg = f"{r.status_code} from the API: {r.text[:200]}"
        raise httpx.HTTPError(msg)
    r.raise_for_status()


def validate(df: pl.DataFrame, lo: str, hi: str) -> None:
    """Fail a shard whose contents contradict the request, rather than persisting it.

    Raises:
        ValueError: If any ``news_id`` is null, or a ``publish_time`` falls outside ``[lo, hi)``.
    """
    if df.is_empty():
        return
    if df["news_id"].null_count():
        msg = f"{df['news_id'].null_count()} rows with a null news_id"
        raise ValueError(msg)
    a, b = datetime.strptime(lo, "%Y-%m-%d"), datetime.strptime(hi, "%Y-%m-%d")
    lo_seen, hi_seen = df["publish_time"].min(), df["publish_time"].max()
    if lo_seen < a or hi_seen >= b:
        msg = f"publish_time {lo_seen} .. {hi_seen} outside the requested [{lo}, {hi})"
        raise ValueError(msg)


def write_atomic(df: pl.DataFrame, path: Path) -> int:
    """Write a shard via tmp + rename so an interrupted run never leaves a file resume would skip."""
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)
    return path.stat().st_size
