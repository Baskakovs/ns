"""Live status for a month-sharded dump sweep: progress, throughput, ETA, gaps.

    uv run src/honey/event_exclusion_exp/sweep_status.py --watch 20

Read-only — it never touches the shards, so it is safe to run against a sweep in flight, from as
many terminals as you like. Everything comes from the sweep's own ``_plan.json`` and
``_manifest.jsonl`` plus a directory listing, so it also works after the fact on a finished run.

Three things it reports that a bare progress bar would not:

* **ETA from recent throughput, not the run average.** These sweeps are wildly non-uniform — early
  years carry a fraction of the news volume of recent ones, and the periods needing per-record
  isolation run ~10x slower. An average-based ETA is misleading for exactly the stretch you are
  watching, so the estimate uses a trailing window and the window is stated.
* **Gap entity-days.** ``events_dump`` returns 500 for individual poisoned ``(entity, day)``
  records; the sweep isolates and skips them rather than losing whole shards. That is the right
  trade but it leaves the panel marginally incomplete, so the running total is surfaced here rather
  than buried in the manifest.
* **Per-period completion.** A sweep is only usable up to its last *complete* month — a month with
  47 of 48 batches is missing 20 entities, which silently narrows the cross-section — so the
  furthest complete month is called out separately from the raw shard count.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

_HTML = """<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="15">
<title>{title}</title><style>
body{{background:#14110f;color:#e1e0d9;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;
padding:24px 28px;margin:0}}
h1{{font-size:15px;font-weight:600;margin:0 0 14px;color:#11D48E;letter-spacing:.02em}}
pre{{margin:0;white-space:pre-wrap}}
.bar{{background:#2a2622;height:10px;border-radius:5px;overflow:hidden;margin:10px 0 16px;max-width:640px}}
.bar>div{{background:#11D48E;height:100%}}
.warn{{color:#E8843A}} .bad{{color:#E5484D}} .dim{{color:#898781}}
</style></head><body><h1>{title}</h1>
<div class="bar"><div style="width:{pct:.1f}%"></div></div><pre>{body}</pre>
<p class="dim">refreshes every 15s &middot; generated {stamp}</p></body></html>
"""


def _hms(sec: float | None) -> str:
    """Seconds as ``2h14m``, or ``-`` when unknown."""
    if sec is None or sec != sec or sec < 0:
        return "-"
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec / 60:.0f}m"
    return f"{sec / 3600:.1f}h"


def _size(nbytes: float) -> str:
    """Bytes as a human-readable size."""
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if nbytes >= div:
            return f"{nbytes / div:.1f} {unit}"
    return f"{nbytes:.0f} B"


def _period_of(name: str) -> str:
    """The period key from a shard filename, for either month or year granularity."""
    stem = name.removesuffix(".parquet")
    parts = stem.split("-")
    return "-".join(parts[2:]) if len(parts) > 2 else parts[-1]


def read_state(out: Path, window: int = 40) -> dict[str, Any]:
    """Gather everything the dashboard shows.

    Args:
        out: The sweep's shard directory.
        window: How many recent successful shards to base throughput and ETA on.

    Returns:
        A dict of plain values, so the text and HTML renderers stay dumb.
    """
    plan_path = out / "_plan.json"
    plan: dict[str, Any] = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    shards = sorted(p for p in out.glob("dump-b*.parquet"))
    done = len(shards)
    total = int(plan.get("total_shards") or 0)

    records: list[dict[str, Any]] = []
    manifest = out / "_manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # a torn final line while the sweep is writing
                continue

    ok = [r for r in records if not r.get("error")]
    # The manifest is append-only across re-runs, so a shard that failed once and succeeded later
    # still has its error line. Counting those would make a finished sweep look broken, so only
    # errors on shards that are *still absent* are reported as outstanding.
    present = {p.name for p in shards}
    errs = [
        r
        for r in records
        if r.get("error") and f"dump-b{int(r['batch']):03d}-{r.get('period', r.get('year'))}.parquet" not in present
    ]
    stale_errs = sum(1 for r in records if r.get("error")) - len(errs)
    gaps = sum(int(r.get("gap_entity_days") or 0) for r in ok)

    # Throughput from a trailing window of completions, which is what makes the ETA honest on a
    # sweep whose per-shard cost varies by an order of magnitude across periods.
    recent = ok[-window:]
    rate = float("nan")
    if len(recent) >= 2:
        span = recent[-1]["ts"] - recent[0]["ts"]
        if span > 0:
            rate = (len(recent) - 1) / span * 60.0

    remaining = max(total - done, 0)
    eta = remaining / rate * 60.0 if rate and rate == rate and rate > 0 else None

    rows = sum(int(r.get("rows") or 0) for r in ok)
    nbytes = sum(int(r.get("bytes") or 0) for r in ok)
    on_disk = sum(p.stat().st_size for p in shards) if shards else 0
    projected = on_disk / done * total if done and total else 0.0

    per_period = Counter(_period_of(p.name) for p in shards)
    batches = int(plan.get("batches") or 0)
    complete = sorted(k for k, v in per_period.items() if batches and v >= batches)
    partial = sorted((k, v) for k, v in per_period.items() if batches and v < batches)

    # The furthest *contiguous* complete run, which is the window the panels can actually use.
    contiguous = 0
    if complete:
        expected = sorted(str(x) for x in (plan.get("periods") or complete))
        for period in expected:
            if period in set(complete):
                contiguous += 1
            else:
                break

    last_ts = max((r["ts"] for r in records), default=None)
    return {
        "dump": plan.get("dump") or out.name,
        "out": str(out),
        "done": done,
        "total": total,
        "pct": (100.0 * done / total) if total else 0.0,
        "rate": rate,
        "window": len(recent),
        "eta": eta,
        "errors": len(errs),
        "stale_errors": stale_errs,
        "last_error": (errs[-1].get("error") or "")[:120] if errs else "",
        "gaps": gaps,
        "rows": rows,
        "bytes": nbytes,
        "on_disk": on_disk,
        "projected": projected,
        "complete_periods": len(complete),
        "contiguous_periods": contiguous,
        "n_periods": len(plan.get("periods") or []) or len(per_period),
        "furthest_complete": complete[-1] if complete else "-",
        "partial": partial[:6],
        "idle_s": (time.time() - last_ts) if last_ts else None,
        "started": plan.get("started"),
        "date_from": plan.get("date_from", "?"),
        "date_to": plan.get("date_to", "?"),
        "workers": plan.get("workers"),
        "batch_size": plan.get("batch_size"),
    }


def render_text(s: dict[str, Any]) -> str:
    """The dashboard as a block of text."""
    elapsed = time.time() - s["started"] if s.get("started") else None
    eta = "-" if (s["total"] and s["done"] >= s["total"]) else _hms(s["eta"])
    lines = [
        f"{s['dump']}  {s['date_from']} .. {s['date_to']}   ({s['out']})",
        f"  batch size {s['batch_size']} entities  |  workers {s['workers']}",
        "",
        f"  shards        {s['done']:,} / {s['total']:,}   ({s['pct']:.1f}%)",
        f"  months        {s['complete_periods']} of {s['n_periods']} complete"
        f"  |  contiguous from start: {s['contiguous_periods']}"
        f"  |  furthest complete: {s['furthest_complete']}",
        f"  throughput    {s['rate']:.1f} shards/min   (trailing {s['window']})",
        f"  ETA           {eta}   |  elapsed {_hms(elapsed)}",
        "",
        f"  rows pulled   {s['rows']:,}",
        f"  on disk       {_size(s['on_disk'])}   |  projected total {_size(s['projected'])}",
        f"  gap entity-days {s['gaps']:,}   (poisoned records isolated and skipped)",
        f"  outstanding   {s['errors']:,} shard(s) failed and still missing"
        + (f"   ({s['stale_errors']:,} earlier failures since re-fetched)" if s["stale_errors"] else ""),
    ]
    if s["last_error"]:
        lines.append(f"    last: {s['last_error']}")
    if s["partial"]:
        lines.append("  partial months: " + ", ".join(f"{k} ({v})" for k, v in s["partial"]))
    finished = s["total"] and s["done"] >= s["total"]
    if finished:
        lines.append("  COMPLETE — every shard on disk")
    elif s["idle_s"] is not None and s["idle_s"] > 300:
        lines.append(f"  !! no manifest activity for {_hms(s['idle_s'])} — the sweep may have stopped")
    return "\n".join(lines)


def write_html(s: dict[str, Any], out: Path) -> Path:
    """Write a self-refreshing status page next to the shards."""
    path = out / "sweep_status.html"
    body = render_text(s)
    for needle, cls in (("!!", "bad"), ("COMPLETE", "dim")):
        body = body.replace(needle, f'<span class="{cls}">{needle}</span>', 1)
    path.write_text(
        _HTML.format(
            title=f"{s['dump']} — {s['done']:,}/{s['total']:,} shards ({s['pct']:.1f}%)",
            pct=s["pct"],
            body=body,
            stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
    )
    return path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/events_dump", help="shard directory to watch")
    ap.add_argument("--html", action="store_true", help="also write sweep_status.html in that directory")
    ap.add_argument("--watch", type=float, default=0.0, help="reprint every N seconds (0 = once)")
    ap.add_argument("--window", type=int, default=40, help="trailing shards used for throughput and ETA")
    a = ap.parse_args()

    out = Path(a.out)
    while True:
        state = read_state(out, window=a.window)
        if a.watch:
            print("\033[2J\033[H", end="")  # clear, so a watched view does not scroll
        print(render_text(state), flush=True)
        if a.html:
            print(f"\n  -> {write_html(state, out)}", flush=True)
        if not a.watch:
            return
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
