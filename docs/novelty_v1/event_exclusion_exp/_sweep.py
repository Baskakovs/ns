"""Resumable month-sharded dump sweep, shared by the events and sentiment pulls.

A ``DumpSpec`` supplies the endpoint and its column casting; everything else — the month grid,
resume, atomic writes, the manifest, the 401 abort and the date-bisection retry — is here, so a
second dump added later cannot quietly drift from how the first was pulled. That matters because the
experiment joins the dumps on ``(news_id, yukka_id)``.

Why the grid is monthly: ``events_dump`` **rejects** ``entities x days > 1000`` outright, with a 400
and an explicit message, so :func:`check_complexity` asserts the batch size up front. (The sentiment
dump has no such limit and is pulled by year elsewhere; ``verify.spot_check_completeness`` re-checks
a sample of any dump's shards against single-entity re-pulls, which is how a genuine shortfall would
be caught.)

Resume is by file *existence*: an entity batch with no rows in a month is a legitimate empty
result, so "empty means retry" would re-fetch it forever. Writes are atomic, so a killed process
never leaves a partial file that resume would skip. A 401 aborts the whole sweep rather than
writing hundreds of empty shards that look real.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from honey.event_exclusion_exp._io import (
    BACKOFF,
    NLP_VERSION,
    AuthExpiredError,
    auth_headers,
    batches,
    check_response,
    load_universe,
    validate,
    write_atomic,
)

#: Server-side cap on ``entities * days`` per request, stated by ``events_dump``'s own 400 body.
MAX_COMPLEXITY = 1000

#: Longest month, for the worst-case complexity check.
MAX_DAYS_PER_MONTH = 31

#: How many times a failing window may be halved before the shard is given up on. 5 takes a
#: 31-day window down to a single day, which is the finest split that still fills one shard.
MAX_SPLIT_DEPTH = 5

#: Retry schedule used while a failing request can still be narrowed — deliberately *empty*. The two
#: failures that reach here are an oversized response and a poisoned ``(entity, day)`` record, and
#: neither is cured by waiting: the remedy is a smaller request. Isolating one poisoned record takes
#: ~10 narrowing steps, so every retry-with-sleep on the way down is paid ten times over. The full
#: :data:`BACKOFF` still applies at the finest granularity, where narrowing is no longer an option
#: and patience is the only remaining move.
SPLIT_BACKOFF: tuple[int, ...] = ()

#: Seconds to pace between requests, per worker. The events endpoint tolerates serial requests
#: comfortably (a month of 20 entities in 7s) but returns 500 for nearly everything under
#: concurrent load — measured 93 failures to 3 successes at 4 workers, against 3/3 successes on the
#: *same* windows issued serially 8s apart. Pacing is what makes the difference, not the window.
DEFAULT_PACE = 0.0


@dataclass(frozen=True)
class DumpSpec:
    """One dump endpoint: where to call it, what columns to keep, how to cast them."""

    name: str
    url: str
    schema: dict[str, Any]
    normalise: Callable[[list[dict[str, Any]]], pl.DataFrame]
    extra_params: dict[str, str] = field(default_factory=dict)


def check_complexity(batch: int) -> None:
    """Fail fast if ``batch`` would breach the server's ``entities * days <= 1000`` cap.

    Raises:
        SystemExit: With the largest workable batch size, so the fix is obvious.
    """
    worst = batch * MAX_DAYS_PER_MONTH
    if worst > MAX_COMPLEXITY:
        msg = (
            f"--batch {batch} x {MAX_DAYS_PER_MONTH} days = {worst} exceeds the API's complexity "
            f"limit of {MAX_COMPLEXITY}. Use --batch {MAX_COMPLEXITY // MAX_DAYS_PER_MONTH} or less."
        )
        raise SystemExit(msg)


def months(lo: date, hi: date) -> list[str]:
    """The ``YYYY-MM`` periods whose request window overlaps ``[lo, hi)``, in order."""
    out: list[str] = []
    y, m = lo.year, lo.month
    while date(y, m, 1) < hi:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_window(period: str, lo: date, hi: date) -> tuple[str, str] | None:
    """The ``[from, to)`` request window for a ``YYYY-MM`` period, clipped to the sweep bounds."""
    y, m = (int(x) for x in period.split("-"))
    start = date(y, m, 1)
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    a, b = max(start, lo), min(nxt, hi)
    return (a.isoformat(), b.isoformat()) if a < b else None


def shard_path(out: Path, batch_ix: int, period: str) -> Path:
    """Path of one shard. The key is in the filename, which is what makes resume trivial."""
    return out / f"dump-b{batch_ix:03d}-{period}.parquet"


def fetch(
    spec: DumpSpec,
    client: httpx.Client,
    entities: list[str],
    lo: str,
    hi: str,
    backoff: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """POST one (entities, window) request, retrying transport errors and 5xx. Raises on 401.

    ``deduplicate=false`` on both dumps keeps the raw grain: near-duplicate suppression would change
    what "an article tagged event a" means on one side of the join but not the other.

    Args:
        spec: Which dump to call.
        client: Shared HTTP client.
        entities: The batch's entity ids.
        lo: Inclusive ISO start date.
        hi: Exclusive ISO end date.
        backoff: Retry schedule in seconds; defaults to :data:`BACKOFF`. :func:`fetch_window` passes
            a short one while a failing window can still be split, since waiting 42s to re-ask for
            a response the server has already refused to build only adds load.
    """
    last: Exception | None = None
    params = {
        "datetime_from": lo,
        "datetime_to": hi,
        "version": NLP_VERSION,
        "deduplicate": "false",
        **spec.extra_params,
    }
    schedule = BACKOFF if backoff is None else backoff
    for pause in (*schedule, None):
        try:
            r = client.post(spec.url, json=entities, params=params)
            check_response(r)
            return list(r.json().get("entries", []))
        except AuthExpiredError:
            raise
        except (httpx.HTTPError, ValueError) as exc:  # ValueError covers a truncated JSON body
            last = exc
            if pause is None:
                break
            time.sleep(pause)
    msg = f"giving up after {len(schedule) + 1} attempts: {last}"
    raise RuntimeError(msg)


def fetch_window(
    spec: DumpSpec,
    client: httpx.Client,
    entities: list[str],
    lo: str,
    hi: str,
    depth: int = 0,
    max_days: int | None = None,
    max_entities: int | None = None,
    pace: float = DEFAULT_PACE,
    gaps: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Fetch one window, chunking it proactively and bisecting further if a request still fails.

    Three distinct failure modes, handled in order of how cheap the remedy is:

    * **Oversized response.** The heaviest entity batches return responses the gateway will not
      finish — a 502 after ~11 minutes, or a 500 — with no usable error. Retrying is futile.
    * **Recent periods.** ``events_dump`` returns 500 for a *whole month* of 2025-12 onwards while
      the same batch's week (13,003 rows) and single day (4,999 rows) succeed immediately.
      ``max_days`` chunks up front so this is never provoked.
    * **Poisoned individual records.** Some ``(entity, day)`` combinations return 500 no matter how
      the request is framed, and they are genuinely per-record rather than per-date — measured:
      ``b001`` fails on 2025-12-18 while ``b002`` and ``b029`` succeed on that day, and ``b002``
      fails on 2025-12-19 while ``b001`` succeeds. No date split can fix this, so once a window is
      down to a single day the **entity list** is halved instead, isolating the bad record. A single
      entity on a single day that still fails is recorded in ``gaps`` and skipped, because losing
      nine months of data to a handful of broken records would be the worse outcome — but the gap is
      returned to the caller and logged rather than swallowed.

    Sub-results are concatenated, so a shard is identical to one a single call could have returned,
    minus anything listed in ``gaps``.

    Args:
        spec: Which dump to call.
        client: Shared HTTP client.
        entities: The batch's entity ids.
        lo: Inclusive ISO start date.
        hi: Exclusive ISO end date.
        depth: Recursion depth, bounded by :data:`MAX_SPLIT_DEPTH`.
        max_days: Split any window longer than this before requesting it. ``None`` requests the
            whole window and only splits on failure.
        max_entities: Request at most this many entities at a time. ``None`` sends the whole batch
            and narrows only on failure. Set it to 1 for **low-volume, high-poison** periods: the
            pre-2020 archive is tiny (a batch-month is ~700 rows) yet riddled with poisoned records,
            and recovering from failure there costs far more than avoiding it — one shard of 629
            rows took 80 minutes via date-then-entity narrowing, against seconds for 20 single-entity
            requests.
        pace: Seconds to wait before each request, to keep concurrent workers from bursting.
        gaps: Mutable list collecting ``{"entity", "date"}`` for single entity-days the API refuses
            to serve. When ``None`` such a failure raises instead, which is the right behaviour for
            a caller that would rather have no shard than an incomplete one.

    Returns:
        The window's entries, in whatever order the sub-windows returned them.

    Raises:
        AuthExpiredError: Propagated immediately so the sweep aborts.
        RuntimeError: If a single entity-day cannot be fetched and ``gaps`` is None.
    """
    a, b = date.fromisoformat(lo), date.fromisoformat(hi)

    # Proactive entity chunking, tried before date chunking: when poison is common but volume is
    # low, isolating by entity up front confines any failure to a single entity's window, instead of
    # failing the whole batch and rediscovering the culprit by bisection.
    if max_entities is not None and len(entities) > max_entities:
        out: list[dict[str, Any]] = []
        for i in range(0, len(entities), max_entities):
            out += fetch_window(
                spec,
                client,
                entities[i : i + max_entities],
                lo,
                hi,
                depth,
                max_days=max_days,
                max_entities=None,
                pace=pace,
                gaps=gaps,
            )
        return out

    # Proactive chunking: ask in pieces the endpoint is known to serve, rather than provoking a
    # failure and recovering from it.
    if max_days is not None and (b - a).days > max_days:
        out: list[dict[str, Any]] = []
        start = a
        while start < b:
            end = min(start + timedelta(days=max_days), b)
            out += fetch_window(
                spec,
                client,
                entities,
                start.isoformat(),
                end.isoformat(),
                depth,
                max_days=None,
                max_entities=None,
                pace=pace,
                gaps=gaps,
            )
            start = end
        return out

    if pace:
        time.sleep(pace)
    splittable = depth < MAX_SPLIT_DEPTH and (b - a).days > 1
    # While the window can still be split, retry only briefly: a 5xx here means the server would not
    # build a response this size, and the full ladder just spends 42s adding load before the split
    # that actually fixes it. At the finest granularity there is nothing left to try but patience.
    can_split_entities = len(entities) > 1
    short = SPLIT_BACKOFF if (splittable or can_split_entities) else None
    try:
        return fetch(spec, client, entities, lo, hi, backoff=short)
    except AuthExpiredError:
        raise
    except (RuntimeError, httpx.HTTPError):
        if splittable:
            mid = a + (b - a) / 2
            print(f"    splitting [{lo}, {hi}) -> [{lo}, {mid}) + [{mid}, {hi})", flush=True)
            return fetch_window(
                spec, client, entities, lo, mid.isoformat(), depth + 1, pace=pace, gaps=gaps
            ) + fetch_window(spec, client, entities, mid.isoformat(), hi, depth + 1, pace=pace, gaps=gaps)
        if can_split_entities:
            half = len(entities) // 2
            print(f"    splitting {len(entities)} entities on {lo} -> {half} + {len(entities) - half}", flush=True)
            return fetch_window(spec, client, entities[:half], lo, hi, depth, pace=pace, gaps=gaps) + fetch_window(
                spec, client, entities[half:], lo, hi, depth, pace=pace, gaps=gaps
            )
        if gaps is None:
            raise
        print(f"    UNFETCHABLE: {entities[0]} on {lo} — recorded as a gap and skipped", flush=True)
        gaps.append({"entity": entities[0], "date": lo})
        return []


def run_shard(
    spec: DumpSpec,
    client: httpx.Client,
    out: Path,
    batch_ix: int,
    entities: list[str],
    period: str,
    lo: date,
    hi: date,
    abort: threading.Event,
    record: Callable[[dict[str, Any]], None],
    max_days: int | None = None,
    max_entities: int | None = None,
    pace: float = DEFAULT_PACE,
) -> None:
    """Fetch, validate and persist one shard, then append its manifest line. Skips existing shards."""
    path = shard_path(out, batch_ix, period)
    if path.exists() or abort.is_set():
        return
    win = month_window(period, lo, hi)
    if win is None:
        return
    t0 = time.time()
    try:
        gaps: list[dict[str, str]] = []
        entries = fetch_window(
            spec, client, entities, *win, max_days=max_days, max_entities=max_entities, pace=pace, gaps=gaps
        )
        df = spec.normalise(entries)
        validate(df, *win)
        size = write_atomic(df, path)
    except AuthExpiredError:
        abort.set()
        raise
    except Exception as exc:  # noqa: BLE001 — one bad shard must not kill the sweep
        record({"batch": batch_ix, "period": period, "error": f"{type(exc).__name__}: {exc}"})
        return
    row: dict[str, Any] = {
        "batch": batch_ix,
        "period": period,
        "entities": len(entities),
        "rows": df.height,
        "bytes": size,
        "elapsed_s": round(time.time() - t0, 2),
    }
    if gaps:
        # Surfaced, never swallowed: the shard is real but incomplete, and only the manifest
        # records which entity-days the API refused so the shortfall can be quantified later.
        row["gap_entity_days"] = len(gaps)
        row["gaps"] = gaps
    record(row)


def sweep(
    spec: DumpSpec,
    out: Path,
    universe_path: str,
    batch: int,
    workers: int,
    lo: date,
    hi: date,
    max_days: int | None = None,
    max_entities: int | None = None,
    pace: float = DEFAULT_PACE,
) -> None:
    """Run the whole grid: every (entity batch, month) shard not already on disk."""
    check_complexity(batch)
    out.mkdir(parents=True, exist_ok=True)
    universe = load_universe(universe_path)
    grid = batches(universe, batch)
    periods = [p for p in months(lo, hi) if month_window(p, lo, hi) is not None]
    (out / "_plan.json").write_text(
        json.dumps(
            {
                "dump": spec.name,
                "endpoint": spec.url,
                "entities": len(universe),
                "batch_size": batch,
                "batches": len(grid),
                # The universe is recorded verbatim, not just its path: the batch partition is
                # derived from it, so a later edit to the universe file would silently reassign
                # entities to different batches and mismatch the two dumps at join time.
                "universe": universe,
                "periods": periods,
                "total_shards": len(grid) * len(periods),
                "date_from": lo.isoformat(),
                "date_to": hi.isoformat(),
                "version": NLP_VERSION,
                "deduplicate": False,
                "max_days": max_days,
                "max_entities": max_entities,
                "pace": pace,
                "workers": workers,
                "started": time.time(),
            },
            indent=2,
        )
    )
    free = shutil.disk_usage(out).free / 1e9
    print(
        f"{spec.name}: {len(universe)} entities, {len(grid)} batches x {len(periods)} months = "
        f"{len(grid) * len(periods)} shards | {free:.0f} GB free",
        flush=True,
    )

    lock, abort = threading.Lock(), threading.Event()
    manifest = out / "_manifest.jsonl"

    def record(row: dict[str, Any]) -> None:
        """Append one manifest line under a lock, and echo it."""
        row["ts"] = time.time()
        with lock, manifest.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        tag = row.get("error") or f"{row.get('rows', 0):>9,} rows  {row.get('elapsed_s', 0):>6.1f}s"
        print(f"  b{row['batch']:03d} {row['period']}  {tag}", flush=True)

    todo = [(bi, ents, p) for p in periods for bi, ents in enumerate(grid) if not shard_path(out, bi, p).exists()]
    print(f"{len(todo)} shards to fetch ({len(grid) * len(periods) - len(todo)} already on disk)", flush=True)

    limits = httpx.Limits(max_connections=workers + 2, max_keepalive_connections=workers)
    with (
        httpx.Client(headers=auth_headers(), timeout=1800, limits=limits) as client,
        ThreadPoolExecutor(workers) as ex,
    ):
        list(
            ex.map(
                lambda t: run_shard(
                    spec,
                    client,
                    out,
                    t[0],
                    t[1],
                    t[2],
                    lo,
                    hi,
                    abort,
                    record,
                    max_days=max_days,
                    max_entities=max_entities,
                    pace=pace,
                ),
                todo,
            )
        )
    if abort.is_set():
        msg = "sweep aborted: the API returned 401. Refresh YUKKA_TOKEN and re-run — finished shards are kept."
        raise SystemExit(msg)
    print("sweep complete", flush=True)


def build_parser(default_out: str, doc: str | None) -> argparse.ArgumentParser:
    """The CLI shared by both sweeps."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=default_out, help="shard directory")
    ap.add_argument("--universe", default="data/stoxx600_ever_members.parquet", help="parquet with a yukka_id column")
    ap.add_argument(
        "--batch",
        type=int,
        default=20,
        help="entities per request; must match the other dump's sweep, and batch x 31 must be <= 1000",
    )
    ap.add_argument("--workers", type=int, default=14, help="concurrent requests")
    ap.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="split each period into windows of at most this many days before requesting "
        "(events_dump 500s on a whole month of 2025-12 onwards; --max-days 7 avoids it)",
    )
    ap.add_argument(
        "--max-entities",
        type=int,
        default=None,
        help="request at most this many entities at a time; use 1 for pre-2020 periods, where "
        "poisoned records are common and volume is low so avoiding failure beats recovering from it",
    )
    ap.add_argument(
        "--pace",
        type=float,
        default=DEFAULT_PACE,
        help="seconds to wait before each request, per worker; throttles bursts that provoke 500s",
    )
    ap.add_argument("--from", dest="lo", default="2024-01-01", help="inclusive start date")
    ap.add_argument("--to", dest="hi", default=None, help="exclusive end date (default: today)")
    return ap


def main(spec: DumpSpec, default_out: str, doc: str | None = None) -> None:
    """CLI entry point for one dump."""
    a = build_parser(default_out, doc).parse_args()
    hi = date.fromisoformat(a.hi) if a.hi else datetime.now(tz=UTC).date()
    sweep(
        spec,
        Path(a.out),
        a.universe,
        a.batch,
        a.workers,
        date.fromisoformat(a.lo),
        hi,
        max_days=a.max_days,
        max_entities=a.max_entities,
        pace=a.pace,
    )
