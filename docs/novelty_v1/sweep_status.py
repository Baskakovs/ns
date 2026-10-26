"""Status view for a `sentiment_sweep` run — terminal summary, or a live HTML page.

    uv run src/honey/novelty_score_exp/sweep_status.py            # print once
    uv run src/honey/novelty_score_exp/sweep_status.py --html     # + sweep_status.html, refreshes itself

Reads only `_plan.json` and `_manifest.jsonl`, so it never touches the sweep's process and can be run
as often as you like while the sweep is running, or afterwards as a report. Mirrors the meta-refresh
approach in `status_dashboard.py` so a long run can be watched in a browser tab.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

_HTML = """<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>sentiment sweep status</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;background:#f6f5f2;color:#1c1e21;margin:0;padding:40px}}
@media(prefers-color-scheme:dark){{body{{background:#17181a;color:#e9e8e4}}}}
.wrap{{max-width:720px;margin:0 auto}} h1{{font-size:22px;margin:0 0 4px}}
.sub{{color:#8a8f96;font-size:13px;margin:0 0 20px}}
.bar{{background:#d9d6cf;border-radius:8px;height:26px;overflow:hidden}}
@media(prefers-color-scheme:dark){{.bar{{background:#2e3135}}}}
.fill{{background:#0ea5a4;height:100%;width:{pct}%;transition:width .4s}}
.big{{font-family:ui-monospace,monospace;font-size:20px;margin:14px 0}}
.grid{{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}}
.card{{background:#fff;border:1px solid #e3e1db;border-radius:10px;padding:12px 16px;min-width:120px}}
@media(prefers-color-scheme:dark){{.card{{background:#1f2124;border-color:#2e3135}}}}
.card .v{{font-family:ui-monospace,monospace;font-size:19px;font-weight:600}}
.card .l{{font-size:11px;color:#8a8f96;margin-top:2px}}
.err{{color:#c0392b;font-family:ui-monospace,monospace;font-size:12px;margin-top:18px;white-space:pre-wrap}}
.ts{{color:#8a8f96;font-size:12px;margin-top:18px}}
</style></head><body><div class="wrap">
<h1>sentiment sweep &mdash; {state}</h1><p class="sub">{sub} &middot; auto-refreshes every 5s</p>
<div class="bar"><div class="fill"></div></div>
<div class="big">{done} / {total} shards &nbsp;({pct}%)</div>
<div class="grid">
<div class="card"><div class="v">{rows}</div><div class="l">rows fetched</div></div>
<div class="card"><div class="v">{gb}</div><div class="l">on disk</div></div>
<div class="card"><div class="v">{rate}</div><div class="l">rows/s</div></div>
<div class="card"><div class="v">{eta}</div><div class="l">ETA</div></div>
<div class="card"><div class="v">{el}</div><div class="l">elapsed</div></div>
<div class="card"><div class="v">{errs}</div><div class="l">outstanding shards</div></div>
</div>{errblock}<p class="ts">updated {upd}</p></div></body></html>"""


def _hms(sec: float | None) -> str:
    """Format seconds as `1h 04m`, or an em dash when unknown."""
    if sec is None or sec != sec or sec < 0:
        return "—"
    return f"{int(sec) // 3600}h {int(sec) % 3600 // 60:02d}m"


def _size(nbytes: int) -> str:
    """Human-readable size — MB below a gigabyte, GB above, so a small run is not all zeroes."""
    return f"{nbytes / 1e9:.2f} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.1f} MB"


def _brief(msg: str, width: int = 110) -> str:
    """One line of an error — a 502 arrives as ten lines of nginx HTML, which drowns the summary."""
    flat = " ".join(str(msg).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _shard_name(rec: dict[str, Any]) -> str:
    """The shard filename a manifest record refers to."""
    return f"dump-b{int(rec['batch']):03d}-{rec['year']}.parquet"


def _year_of(name: str) -> str:
    """The year key in a shard filename."""
    return name.removesuffix(".parquet").split("-")[-1]


def read_state(out: Path, window: int = 40) -> dict[str, Any]:
    """Summarise a sweep from its plan and manifest, tolerating a run that has not started yet.

    The shard directory outlives any single run: `_plan.json` is rewritten each time, but shards
    and manifest lines from earlier runs stay. So progress is measured against the *plan's own*
    grid (its batches x years) rather than a raw directory count — otherwise a 2024-2026 re-run
    over a directory already holding 2013-2023 reports 464%.

    Args:
        out: The sweep's shard directory.
        window: How many recent completions to base throughput and ETA on.

    Returns:
        A dict of plain values, so the text and HTML renderers stay dumb.
    """
    plan_path, man_path = out / "_plan.json", out / "_manifest.jsonl"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    records: list[dict[str, Any]] = []
    if man_path.exists():
        for line in man_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # a torn final line while the sweep is writing
                continue

    shards = sorted(out.glob("dump-b*.parquet"))
    present = {p.name for p in shards}
    years = [int(y) for y in plan.get("years", [])]
    batches = int(plan.get("batches") or 0)
    expected = {f"dump-b{b:03d}-{y}.parquet" for y in years for b in range(batches)}
    in_plan = sorted(expected & present) if expected else sorted(present)
    done, total = len(in_plan), int(plan.get("total_shards", 0) or len(expected))
    extra = sorted(present - expected) if expected else []

    # The manifest is append-only across re-runs, so a shard that failed once and was re-fetched
    # later still carries its error line. Only errors on shards *still absent* are outstanding.
    # One line per still-missing shard, latest attempt: a shard retried three times is one problem.
    failed: dict[str, dict[str, Any]] = {}
    for rec in records:
        if "error" in rec and _shard_name(rec) not in present:
            failed[_shard_name(rec)] = rec
    errors = list(failed.values())
    stale_errors = len({_shard_name(r) for r in records if "error" in r}) - len(errors)
    # Last word per shard, so a re-fetch replaces the earlier attempt rather than double-counting.
    latest: dict[str, dict[str, Any]] = {}
    for rec in records:
        if "error" not in rec:
            latest[_shard_name(rec)] = rec
    ok = [r for name, r in latest.items() if name in present]

    rows = sum(int(r.get("rows", 0)) for r in ok)
    disk = sum(p.stat().st_size for p in shards)

    # Throughput from a trailing window of completions rather than the run average: shard cost
    # varies by an order of magnitude across years, and the average is stale the moment a run ends.
    recent = sorted((r for r in ok if r.get("ts")), key=lambda r: r["ts"])[-window:]
    rate = 0.0
    if len(recent) >= 2:
        span = recent[-1]["ts"] - recent[0]["ts"]
        if span > 0:
            rate = sum(int(r.get("rows", 0)) for r in recent[1:]) / span

    last_ts = max((r["ts"] for r in records if r.get("ts")), default=None)
    idle = (time.time() - last_ts) if last_ts else None
    started = plan.get("started")
    # Wall-clock to the last sign of life, not to now — a finished run must stop accruing elapsed.
    elapsed = (last_ts - started) if (started and last_ts and last_ts > started) else None
    eta = None
    if rate and done < total:
        remaining_rows = rows / done * (total - done) if done else 0.0
        eta = remaining_rows / rate
    return {
        "plan": plan,
        "done": done,
        "total": total,
        "extra": len(extra),
        "extra_span": (min(map(_year_of, extra)), max(map(_year_of, extra))) if extra else None,
        "rows": rows,
        "disk": disk,
        "rate": rate,
        "elapsed": elapsed,
        "idle": idle,
        "eta": eta,
        "errors": errors,
        "stale_errors": stale_errors,
        "empty": sum(1 for r in ok if int(r.get("rows", 0)) == 0),
    }


def render_text(s: dict[str, Any]) -> str:
    """One-screen terminal summary."""
    pct = s["done"] / s["total"] * 100 if s["total"] else 0.0
    p, done = s["plan"], s["total"] and s["done"] >= s["total"]
    lines = [
        f"sentiment sweep  {p.get('date_from', '?')} .. {p.get('date_to', '?')}  version {p.get('version', '?')}",
        f"  shards      {s['done']:,} / {s['total']:,}  ({pct:.1f}%)",
    ]
    if s["extra"]:
        lo, hi = s["extra_span"]
        span = lo if lo == hi else f"{lo}..{hi}"
        lines.append(f"  also here   {s['extra']:,} shards outside this plan ({span}) — earlier runs")
    lines += [
        f"  rows        {s['rows']:,}",
        f"  on disk     {_size(s['disk'])}",
        f"  throughput  {s['rate']:,.0f} rows/s  (trailing)",
        f"  elapsed     {_hms(s['elapsed'])}    ETA {'—' if done else _hms(s['eta'])}",
        f"  empty       {s['empty']:,} shards (legitimately zero rows)",
        f"  outstanding {len(s['errors']):,} shard(s) failed and still missing"
        + (f"   ({s['stale_errors']:,} earlier failures since re-fetched)" if s["stale_errors"] else ""),
    ]
    lines.extend(f"    b{e['batch']:03d} {e['year']}: {_brief(e['error'])}" for e in s["errors"][:10])
    if len(s["errors"]) > 10:
        lines.append(f"    ... and {len(s['errors']) - 10} more")
    if done:
        lines.append("  COMPLETE — every shard in this plan is on disk")
    elif s["idle"] is not None and s["idle"] > 300:
        lines.append(f"  !! no manifest activity for {_hms(s['idle'])} — the sweep may have stopped")
    return "\n".join(lines)


def write_html(s: dict[str, Any], out: Path) -> Path:
    """Write `sweep_status.html` next to the shards and return its path."""
    pct = s["done"] / s["total"] * 100 if s["total"] else 0.0
    p = s["plan"]
    state = "complete" if s["total"] and s["done"] >= s["total"] else "running"
    errs = s["errors"]
    errblock = ""
    if errs:
        shown = "\n".join(f"b{e['batch']:03d} {e['year']}: {_brief(e['error'])}" for e in errs[:12])
        errblock = f'<div class="err">{shown}</div>'
    path = out / "sweep_status.html"
    path.write_text(
        _HTML.format(
            state=state,
            sub=f"{p.get('entities', '?')} entities · batch {p.get('batch_size', '?')} · "
            f"{p.get('date_from', '?')} to {p.get('date_to', '?')} · version {p.get('version', '?')}",
            pct=f"{pct:.1f}",
            done=f"{s['done']:,}",
            total=f"{s['total']:,}",
            rows=f"{s['rows']:,}",
            gb=_size(s["disk"]),
            rate=f"{s['rate']:,.0f}",
            eta=_hms(s["eta"]),
            el=_hms(s["elapsed"]),
            errs=f"{len(errs):,}",
            errblock=errblock,
            upd=time.strftime("%H:%M:%S"),
        )
    )
    return path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/sentiment_dump", help="shard directory")
    ap.add_argument("--html", action="store_true", help="also write sweep_status.html")
    ap.add_argument("--watch", type=float, default=0.0, help="reprint every N seconds")
    a = ap.parse_args()
    out = Path(a.out)
    while True:
        s = read_state(out)
        print(render_text(s), flush=True)
        if a.html:
            print(f"  html        {write_html(s, out)}", flush=True)
        if not a.watch:
            return
        time.sleep(a.watch)
        print("", flush=True)


if __name__ == "__main__":
    main()
