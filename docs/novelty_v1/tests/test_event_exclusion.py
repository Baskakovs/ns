"""Count-algebra tests for the event-exclusion experiment.

The package is excluded from the coverage gate (research code), but the count algebra in
``panel.py`` and ``statistic.py`` is what the whole result rests on, and a join fan-out would
produce a plausible-looking panel with silently wrong numbers. These tests pin the grain.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import pytest

from honey.event_exclusion_exp import _sweep, panel, run_ic, statistic, sweep_status, verify

# One hand-computed fixture, reused across tests.
#
# ACME on 2024-01-02 has three articles:
#   n1  two matches (+1, -1)   tagged Merger, Lawsuit
#   n2  one match  (+1)        tagged Merger
#   n3  one match  ( 0)        untagged
# so pos=2 neu=1 neg=1 tot=4, and for Merger pos_a=2 neg_a=1 tot_a=3 (n1's two matches + n2's one).
#
# BETA on 2024-01-02 has one article, one match (-1), untagged.
_MATCHES = pl.DataFrame(
    {
        "news_id": ["n1", "n1", "n2", "n3", "n4"],
        "yukka_id": ["company:acme"] * 4 + ["company:beta"],
        "date": [date(2024, 1, 2)] * 5,
        "sentiment": [1, -1, 1, 0, -1],
    },
    schema_overrides={"sentiment": pl.Int8},
)

_TAGS = pl.DataFrame(
    {
        "news_id": ["n1", "n1", "n2"],
        "yukka_id": ["company:acme"] * 3,
        "event_name": ["Merger", "Lawsuit", "Merger"],
    }
)


def _baseline() -> pl.DataFrame:
    """The fixture's baseline counts."""
    return panel.baseline_counts(_MATCHES.lazy()).collect().sort("yukka_id")


def _events() -> pl.DataFrame:
    """The fixture's event counts."""
    return panel.event_counts(_MATCHES.lazy(), _TAGS.lazy()).collect().sort("yukka_id", "event_name")


def test_baseline_counts_match_hand_computation() -> None:
    """Baseline pos/neu/neg/tot are the hand-computed values, and tot is their sum."""
    got = _baseline()
    acme = got.filter(pl.col("yukka_id") == "company:acme").to_dicts()[0]
    assert (acme["pos"], acme["neu"], acme["neg"], acme["tot"]) == (2, 1, 1, 4)
    beta = got.filter(pl.col("yukka_id") == "company:beta").to_dicts()[0]
    assert (beta["pos"], beta["neu"], beta["neg"], beta["tot"]) == (0, 0, 1, 1)
    assert (got["pos"] + got["neu"] + got["neg"] == got["tot"]).all()


def test_event_counts_match_hand_computation() -> None:
    """Merger picks up both of n1's matches plus n2's; Lawsuit picks up only n1's two."""
    got = _events()
    by_event = {r["event_name"]: r for r in got.to_dicts()}
    assert (by_event["Merger"]["pos_a"], by_event["Merger"]["neg_a"], by_event["Merger"]["tot_a"]) == (2, 1, 3)
    assert (by_event["Lawsuit"]["pos_a"], by_event["Lawsuit"]["neg_a"], by_event["Lawsuit"]["tot_a"]) == (1, 1, 2)
    # BETA has no tagged article, so it contributes no event rows at all.
    assert "company:beta" not in set(got["yukka_id"])


def test_duplicate_event_rows_do_not_fan_out_the_join() -> None:
    """Repeating an event row (different role/factuality) must not inflate the counts.

    This is the failure mode the ``unique`` in ``event_tags`` exists to prevent: without it the
    same article's sentiment would be counted once per event mention.
    """
    duplicated = pl.concat([_TAGS, _TAGS, _TAGS]).unique()
    got = panel.event_counts(_MATCHES.lazy(), duplicated.lazy()).collect()
    assert got.sort("yukka_id", "event_name").equals(_events())


def test_event_tags_dedupes_multi_row_events(tmp_path) -> None:
    """`event_tags` collapses the dump's (role, factuality, temporality) grain to one row per event."""
    shard = tmp_path / "dump-b000-2024-01.parquet"
    pl.DataFrame(
        {
            "news_id": ["n1"] * 3,
            "yukka_id": ["company:acme"] * 3,
            "publish_time": [datetime(2024, 1, 2, 9, 0)] * 3,
            "event_name": ["Merger", "Merger", "Merger"],
            "factuality": ["fact", "possible", "fact"],
            "event_is_negated": [0, 0, 1],
        }
    ).with_columns(
        pl.col("publish_time").cast(pl.Datetime("us")), pl.col("event_is_negated").cast(pl.Int8)
    ).write_parquet(shard)

    tags = panel.event_tags([shard], "2024-01-01", "2024-02-01").collect()
    assert tags.height == 1
    facts = panel.event_tags([shard], "2024-01-01", "2024-02-01", facts_only=True).collect()
    assert facts.height == 1


def test_subset_property_holds_cellwise() -> None:
    """pos_a <= pos, neg_a <= neg, tot_a <= tot for every cell — the subtraction can never go negative."""
    joined = _events().join(_baseline(), on=["date", "yukka_id"], how="inner")
    assert joined.height == _events().height
    assert (joined["pos_a"] <= joined["pos"]).all()
    assert (joined["neu_a"] <= joined["neu"]).all()
    assert (joined["neg_a"] <= joined["neg"]).all()
    assert (joined["tot_a"] <= joined["tot"]).all()


def test_baseline_statistic_matches_formula() -> None:
    """(pos - neg) / tot, with tot the pos+neu+neg denominator."""
    got = statistic.baseline_statistic(_baseline()).sort("yukka_id")
    acme = got.filter(pl.col("yukka_id") == "company:acme")["signal"][0]
    assert acme == pytest.approx((2 - 1) / 4)
    beta = got.filter(pl.col("yukka_id") == "company:beta")["signal"][0]
    assert beta == pytest.approx((0 - 1) / 1)


def test_variant_statistic_matches_formula() -> None:
    """((pos - pos_a) - (neg - neg_a)) / (tot - tot_a) for the named event."""
    got = statistic.variant_statistic(_baseline(), _events(), "Lawsuit")
    acme = got.filter(pl.col("yukka_id") == "company:acme")["signal"][0]
    # ACME minus Lawsuit: pos 2-1=1, neg 1-1=0, tot 4-2=2
    assert acme == pytest.approx((1 - 0) / 2)


def test_untagged_company_is_unchanged_by_any_variant() -> None:
    """A company with no tagged articles keeps its baseline value in every variant."""
    base = statistic.baseline_statistic(_baseline())
    for event in ("Merger", "Lawsuit"):
        got = statistic.variant_statistic(_baseline(), _events(), event)
        beta_base = base.filter(pl.col("yukka_id") == "company:beta")["signal"][0]
        beta_var = got.filter(pl.col("yukka_id") == "company:beta")["signal"][0]
        assert beta_var == pytest.approx(beta_base)


def test_absent_event_reproduces_baseline_exactly() -> None:
    """A tag matching zero articles must reproduce the baseline statistic bit-for-bit.

    The null-event sanity check: if this fails, the variant path is contaminating the baseline.
    """
    base = statistic.baseline_statistic(_baseline()).sort("date", "yukka_id")
    got = statistic.variant_statistic(_baseline(), _events(), "NoSuchEvent").sort("date", "yukka_id")
    assert got.equals(base)


def test_fully_excluded_cell_is_null_not_inf() -> None:
    """When an event accounts for every match, tot - tot_a == 0 must yield null, not inf or a crash."""
    matches = pl.DataFrame(
        {
            "news_id": ["n1"],
            "yukka_id": ["company:acme"],
            "date": [date(2024, 1, 2)],
            "sentiment": [1],
        },
        schema_overrides={"sentiment": pl.Int8},
    )
    tags = pl.DataFrame({"news_id": ["n1"], "yukka_id": ["company:acme"], "event_name": ["Merger"]})
    base = panel.baseline_counts(matches.lazy()).collect()
    events = panel.event_counts(matches.lazy(), tags.lazy()).collect()
    got = statistic.variant_statistic(base, events, "Merger")
    assert got.height == 1
    assert got["signal"][0] is None


def test_to_wide_uses_the_bridge_and_drops_unmapped_entities() -> None:
    """Wide panels are RIC-keyed; an entity with no RIC in the bridge is dropped, not renamed."""
    long = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 2)],
            "yukka_id": ["company:acme", "company:ghost"],
            "signal": [0.25, -1.0],
        }
    )
    bridge = pl.DataFrame({"yukka_id": ["company:acme"], "ric": ["ACME.DE"]})
    wide = statistic.to_wide(long, bridge)
    assert wide.columns == ["date", "ACME.DE"]
    assert wide["ACME.DE"][0] == pytest.approx(0.25)


def test_newey_west_reduces_to_plain_sem_at_lag_zero() -> None:
    """At lag 0 the estimator is exactly the population standard error of the mean."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    assert run_ic.newey_west_se(x, 0) == pytest.approx(x.std() / np.sqrt(len(x)))


def test_newey_west_inflates_under_autocorrelation() -> None:
    """On an AR(1) series the NW error must exceed the naive one, or every t-stat is overstated.

    This is why the horizon-h arms use lag h: consecutive dates share h-1 days of forward return.
    """
    rng = np.random.default_rng(1)
    ar = np.zeros(4000)
    for i in range(1, len(ar)):
        ar[i] = 0.8 * ar[i - 1] + rng.normal()
    assert run_ic.newey_west_se(ar, 20) > 2 * ar.std() / np.sqrt(len(ar))


def test_newey_west_needs_two_observations() -> None:
    """A single observation has no estimable standard error."""
    assert np.isnan(run_ic.newey_west_se(np.array([1.0]), 1))


def test_benjamini_hochberg_matches_hand_computation() -> None:
    """q_i = min over k >= i of p_k * n / k, monotone and capped at 1."""
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.5])
    assert run_ic.benjamini_hochberg(p) == pytest.approx([0.005, 0.025, 0.04 * 5 / 3, 0.25, 0.5])


def test_benjamini_hochberg_preserves_order_and_skips_nan() -> None:
    """Input order is preserved and NaN neither receives nor dilutes a correction."""
    p = np.array([0.5, np.nan, 0.001])
    q = run_ic.benjamini_hochberg(p)
    assert np.isnan(q[1])
    # family size is 2, so the smallest of two p-values gets p * 2 / 1
    assert q[2] == pytest.approx(0.002)
    assert q[0] == pytest.approx(0.5)


def test_to_trading_day_rolls_forward_and_preserves_identity() -> None:
    """Weekend counts land on the next trading day, and tot stays pos+neu+neg."""
    counts = pl.DataFrame(
        {
            "date": [date(2024, 1, 5), date(2024, 1, 6), date(2024, 1, 7), date(2024, 1, 8)],
            "yukka_id": ["company:acme"] * 4,
            "pos": [1, 2, 3, 4],
            "neu": [0, 0, 0, 0],
            "neg": [1, 0, 0, 1],
            "tot": [2, 2, 3, 5],
        }
    )
    spine = pl.DataFrame({"date": [date(2024, 1, 5), date(2024, 1, 8)]})  # Fri, Mon
    got = statistic.to_trading_day(counts, spine).sort("date")
    assert got["date"].to_list() == [date(2024, 1, 5), date(2024, 1, 8)]
    # Sat(2)+Sun(3)+Mon(4) pos roll into Monday
    monday = got.filter(pl.col("date") == date(2024, 1, 8)).to_dicts()[0]
    assert (monday["pos"], monday["neg"], monday["tot"]) == (9, 1, 10)
    assert (got["pos"] + got["neu"] + got["neg"] == got["tot"]).all()


def test_to_trading_day_preserves_subset_property() -> None:
    """Rolling both panels keeps pos_a <= pos, so the subtraction stays safe."""
    spine = pl.DataFrame({"date": [date(2024, 1, 8)]})
    base = statistic.to_trading_day(_baseline(), spine)
    events = statistic.to_trading_day(_events(), spine, group_extra=("event_name",))
    joined = events.join(base, on=["date", "yukka_id"], how="inner")
    assert joined.height == events.height
    assert (joined["pos_a"] <= joined["pos"]).all()
    assert (joined["tot_a"] <= joined["tot"]).all()


def test_to_trading_day_drops_counts_after_the_last_trading_day() -> None:
    """News with no trading day left to roll into is dropped, not silently attached to the past."""
    counts = pl.DataFrame(
        {"date": [date(2024, 1, 5), date(2024, 1, 20)], "yukka_id": ["company:acme"] * 2, "tot": [2, 9]}
    )
    got = statistic.to_trading_day(counts, pl.DataFrame({"date": [date(2024, 1, 5)]}))
    assert got["date"].to_list() == [date(2024, 1, 5)]
    assert got["tot"].to_list() == [2]


def _write_plan(directory: Path, batch_size: int) -> None:
    """Write the minimal `_plan.json` the partition reconstruction needs."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "_plan.json").write_text(json.dumps({"batch_size": batch_size}))


def _write_universe(path: Path, ids: list[str]) -> None:
    """Write a universe parquet with a yukka_id column."""
    pl.DataFrame({"yukka_id": ids}).write_parquet(path)


def test_entity_partition_reconstructs_the_sweep_partition(tmp_path: Path) -> None:
    """The partition is determined by the sorted universe plus the plan's batch_size."""
    universe = tmp_path / "universe.parquet"
    _write_universe(universe, ["company:c", "company:a", "company:b", "company:d", "company:e"])
    d = tmp_path / "dump"
    _write_plan(d, 2)
    got = panel.entity_partition(d, str(universe))
    assert got == [["company:a", "company:b"], ["company:c", "company:d"], ["company:e"]]


def test_entity_partition_requires_a_plan(tmp_path: Path) -> None:
    """Without `_plan.json` there is no way to know which entities a shard covers."""
    universe = tmp_path / "universe.parquet"
    _write_universe(universe, ["company:a"])
    with pytest.raises(FileNotFoundError):
        panel.entity_partition(tmp_path / "no_plan", str(universe))


def test_shards_for_entities_spans_a_finer_partition(tmp_path: Path) -> None:
    """A wide events group maps onto every finer sentiment batch that holds one of its entities.

    This is what lets the two dumps be pulled at different batch sizes — which they must be, since
    the two endpoints reject and truncate at different thresholds.
    """
    sent = tmp_path / "sent"
    sent.mkdir()
    # sentiment partitioned in batches of 2: b000 = a,b · b001 = c,d · b002 = e,f
    partition = [["company:a", "company:b"], ["company:c", "company:d"], ["company:e", "company:f"]]
    for bi in range(3):
        (sent / f"dump-b{bi:03d}-2024-01.parquet").touch()
        (sent / f"dump-b{bi:03d}-2024-02.parquet").touch()

    # an events group of 4 entities spans the first two sentiment batches, not the third
    got = panel.shards_for_entities(sent, partition, {"company:a", "company:c", "company:d"})
    assert [p.name for p in got] == [
        "dump-b000-2024-01.parquet",
        "dump-b000-2024-02.parquet",
        "dump-b001-2024-01.parquet",
        "dump-b001-2024-02.parquet",
    ]
    assert panel.shards_for_entities(sent, partition, {"company:zzz"}) == []


def test_readers_restrict_to_the_requested_entities(tmp_path: Path) -> None:
    """A shard wider than the group must not leak other entities into the group's counts.

    Without the filter, an entity sharing a sentiment shard with the group would be aggregated once
    per group that touches its shard — double counting straight into the baseline.
    """
    shard = tmp_path / "dump-b000-2024-01.parquet"
    pl.DataFrame(
        {
            "news_id": ["n1", "n2"],
            "yukka_id": ["company:acme", "company:other"],
            "publish_time": [datetime(2024, 1, 2, 9, 0)] * 2,
            "sentiment": [1, -1],
        },
        schema_overrides={"sentiment": pl.Int8},
    ).with_columns(pl.col("publish_time").cast(pl.Datetime("us"))).write_parquet(shard)

    got = panel.sentiment_matches([shard], "2024-01-01", "2024-02-01", entities={"company:acme"}).collect()
    assert got["yukka_id"].to_list() == ["company:acme"]
    both = panel.sentiment_matches([shard], "2024-01-01", "2024-02-01").collect()
    assert both.height == 2


def test_entity_partition_prefers_the_plans_own_universe(tmp_path: Path) -> None:
    """A recorded universe wins over the file, so a later edit to the file cannot reassign batches.

    This is the silent-corruption path the recorded universe exists to close: reconstructing the
    partition from a changed file would mismatch the two dumps with nothing to show for it.
    """
    d = tmp_path / "dump"
    d.mkdir()
    (d / "_plan.json").write_text(json.dumps({"batch_size": 2, "universe": ["company:a", "company:b", "company:c"]}))
    stale = tmp_path / "stale.parquet"
    _write_universe(stale, ["company:x", "company:y", "company:z"])

    got = panel.entity_partition(d, str(stale))
    assert got == [["company:a", "company:b"], ["company:c"]]


def test_entity_partition_falls_back_to_the_file_for_older_dumps(tmp_path: Path) -> None:
    """A plan without a recorded universe still works when a universe file is supplied."""
    d = tmp_path / "dump"
    _write_plan(d, 2)
    universe = tmp_path / "universe.parquet"
    _write_universe(universe, ["company:a", "company:b", "company:c"])
    assert panel.entity_partition(d, str(universe)) == [["company:a", "company:b"], ["company:c"]]


def test_entity_partition_without_universe_anywhere_is_an_error(tmp_path: Path) -> None:
    """Guessing the partition would silently mismatch entities, so it refuses instead."""
    d = tmp_path / "dump"
    _write_plan(d, 2)
    with pytest.raises(ValueError, match="records no universe"):
        panel.entity_partition(d)


def test_split_by_event_matches_filtering() -> None:
    """The pre-split form must give byte-identical results to filtering the whole panel."""
    events = _events()
    split = statistic.split_by_event(events)
    assert set(split) == set(events["event_name"])
    for name in (*split, "NoSuchEvent"):
        via_dict = statistic.variant_statistic(_baseline(), split, name).sort("date", "yukka_id")
        via_frame = statistic.variant_statistic(_baseline(), events, name).sort("date", "yukka_id")
        assert via_dict.equals(via_frame), name


def test_to_wide_can_key_by_entity_id_without_a_bridge() -> None:
    """The local-currency panel is yukka_id-keyed, so `to_wide` must skip the bridge for it.

    Getting this wrong is silent: `Signal.ic` intersects column names and would report NaN rather
    than raise, so the robustness arm would look like it ran and produce nothing.
    """
    long = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 2)],
            "yukka_id": ["company:acme", "company:ghost"],
            "signal": [0.25, -1.0],
        }
    )
    wide = statistic.to_wide(long, pl.DataFrame({"yukka_id": [], "ric": []}), key="yukka_id")
    assert set(wide.columns) == {"date", "company:acme", "company:ghost"}
    assert wide["company:ghost"][0] == pytest.approx(-1.0)


class _FakeClient:
    """Stands in for httpx.Client, failing any window longer than `max_days`."""

    def __init__(self, max_days: int) -> None:
        self.max_days = max_days
        self.calls: list[tuple[str, str]] = []

    def post(self, url: str, json: object, params: dict[str, str]) -> object:
        """Record the window and 502 if it spans more than `max_days`."""
        lo, hi = params["datetime_from"], params["datetime_to"]
        self.calls.append((lo, hi))
        span = (date.fromisoformat(hi) - date.fromisoformat(lo)).days
        if span > self.max_days:
            msg = f"502 for a {span}-day window"
            raise httpx.HTTPError(msg)
        return _FakeResponse([{"window": f"{lo}/{hi}"}])


class _FakeResponse:
    """Minimal httpx.Response stand-in."""

    status_code = 200

    def __init__(self, entries: list[dict[str, object]]) -> None:
        self._entries = entries

    def json(self) -> dict[str, object]:
        """The dump payload shape."""
        return {"entries": self._entries}

    def raise_for_status(self) -> None:
        """No-op: this fake only returns 200s."""


_SPEC = _sweep.DumpSpec(name="fake", url="http://x", schema={}, normalise=lambda e: pl.DataFrame(e))


def test_fetch_window_bisects_a_too_large_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A window the gateway cannot serve is halved until each half succeeds, losing no days.

    This is the 502-on-the-heaviest-batches failure: retrying the same request never works, and
    those batches hold the highest-volume entities, so giving up would drop exactly the data that
    matters most.
    """
    monkeypatch.setattr(_sweep, "BACKOFF", ())  # no sleeping in tests
    client = _FakeClient(max_days=8)
    got = _sweep.fetch_window(_SPEC, client, ["company:ubs"], "2024-01-01", "2024-02-01")

    # every returned sub-window is short enough to have succeeded, and they tile the whole month
    windows = sorted(
        (date.fromisoformat(w.split("/")[0]), date.fromisoformat(w.split("/")[1])) for w in (e["window"] for e in got)
    )
    assert windows[0][0] == date(2024, 1, 1)
    assert windows[-1][1] == date(2024, 2, 1)
    for (_, end), (nxt, _) in pairwise(windows):
        assert end == nxt, "sub-windows must tile the range with no gap or overlap"
    assert all((b - a).days <= 8 for a, b in windows)


def test_fetch_window_gives_up_on_a_single_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single day that still fails is a real error, not something to split further."""
    monkeypatch.setattr(_sweep, "BACKOFF", ())
    client = _FakeClient(max_days=0)  # nothing succeeds
    with pytest.raises((RuntimeError, httpx.HTTPError)):
        _sweep.fetch_window(_SPEC, client, ["company:ubs"], "2024-01-01", "2024-01-02")


def test_fetch_window_does_not_split_when_the_request_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path must stay a single request."""
    monkeypatch.setattr(_sweep, "BACKOFF", ())
    client = _FakeClient(max_days=31)
    got = _sweep.fetch_window(_SPEC, client, ["company:acme"], "2024-01-01", "2024-02-01")
    assert len(client.calls) == 1
    assert got == [{"window": "2024-01-01/2024-02-01"}]


@pytest.mark.parametrize(
    ("name", "period", "window"),
    [
        ("dump-b013-2024.parquet", "2024", (date(2024, 1, 1), date(2025, 1, 1))),
        ("dump-b043-2024-01.parquet", "2024-01", (date(2024, 1, 1), date(2024, 2, 1))),
        ("dump-b000-2026-12.parquet", "2026-12", (date(2026, 12, 1), date(2027, 1, 1))),
    ],
)
def test_shard_period_handles_both_granularities(name: str, period: str, window: tuple[date, date]) -> None:
    """Events are month-sharded and sentiment year-sharded, so both shapes must parse."""
    assert verify.shard_period(Path(name)) == period
    assert verify.period_window(period) == window


def test_period_window_rejects_an_unrecognised_key() -> None:
    """Guessing a window from a malformed key would silently check the wrong dates."""
    with pytest.raises(ValueError, match="unrecognised shard period"):
        verify.period_window("2024-01-15")


def test_fetch_window_chunks_proactively_without_failing_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """`max_days` splits before requesting, so the 500-provoking wide request is never sent.

    Bisection recovers from a failure; chunking avoids causing it. That matters here because the
    endpoint answers a burst of failures with more failures, so provoking one is not free.
    """
    monkeypatch.setattr(_sweep, "BACKOFF", ())
    client = _FakeClient(max_days=31)  # everything would succeed; proving no wide call is made
    got = _sweep.fetch_window(_SPEC, client, ["company:ubs"], "2024-01-01", "2024-02-01", max_days=7)

    spans = [(date.fromisoformat(lo), date.fromisoformat(hi)) for lo, hi in client.calls]
    assert all((b - a).days <= 7 for a, b in spans), "no request may exceed max_days"
    assert spans[0][0] == date(2024, 1, 1)
    assert spans[-1][1] == date(2024, 2, 1)
    for (_, end), (nxt, _) in pairwise(spans):
        assert end == nxt, "chunks must tile the range with no gap or overlap"
    assert len(got) == len(spans), "every chunk contributes its entries"


def test_fetch_window_chunking_still_bisects_a_failing_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunking and bisection compose: a chunk too big for the server is still split further."""
    monkeypatch.setattr(_sweep, "BACKOFF", ())
    client = _FakeClient(max_days=3)  # 7-day chunks will fail and must bisect
    got = _sweep.fetch_window(_SPEC, client, ["company:ubs"], "2024-01-01", "2024-01-15", max_days=7)
    served = [
        (date.fromisoformat(e["window"].split("/")[0]), date.fromisoformat(e["window"].split("/")[1])) for e in got
    ]
    served.sort()
    assert served[0][0] == date(2024, 1, 1)
    assert served[-1][1] == date(2024, 1, 15)
    assert all((b - a).days <= 3 for a, b in served)


class _PoisonClient(_FakeClient):
    """A server with specific `(entity, day)` records that always 500, however small the request."""

    def __init__(self, poisoned: set[tuple[str, str]]) -> None:
        super().__init__(max_days=10_000)
        self.poisoned = poisoned

    def post(self, url: str, json: object, params: dict[str, str]) -> object:
        """500 if any requested entity-day is poisoned, mimicking the real endpoint."""
        lo, hi = params["datetime_from"], params["datetime_to"]
        self.calls.append((lo, hi))
        start, end = date.fromisoformat(lo), date.fromisoformat(hi)
        days = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days)]
        for ent in json:  # type: ignore[union-attr]
            if any((ent, d) in self.poisoned for d in days):
                msg = "500 poisoned record"
                raise httpx.HTTPError(msg)
        return _FakeResponse([{"window": f"{lo}/{hi}", "n": len(json)}])  # type: ignore[arg-type]


def test_fetch_window_isolates_a_poisoned_entity_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """One bad record must cost only that record, not the whole shard.

    The real endpoint 500s on specific (entity, day) pairs — b001 fails on 2025-12-18 while b002
    and b029 succeed that day — so no date split can help. Halving the entity list isolates it.
    """
    monkeypatch.setattr(_sweep, "BACKOFF", ())
    monkeypatch.setattr(_sweep, "SPLIT_BACKOFF", ())
    client = _PoisonClient({("company:bad", "2024-01-18")})
    entities = [f"company:e{i}" for i in range(9)] + ["company:bad"]
    gaps: list[dict[str, str]] = []

    got = _sweep.fetch_window(_SPEC, client, entities, "2024-01-01", "2024-02-01", max_days=7, gaps=gaps)

    assert gaps == [{"entity": "company:bad", "date": "2024-01-18"}]
    # every other entity's data for that day still came back
    served = [e for e in got if e["window"].startswith("2024-01-18")]
    assert sum(e["n"] for e in served) == 9, "the nine healthy entities must still be fetched"


def test_fetch_window_raises_on_a_poisoned_record_when_gaps_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a gaps list the caller wants all-or-nothing, so an unfetchable record must raise."""
    monkeypatch.setattr(_sweep, "BACKOFF", ())
    monkeypatch.setattr(_sweep, "SPLIT_BACKOFF", ())
    client = _PoisonClient({("company:bad", "2024-01-18")})
    with pytest.raises((RuntimeError, httpx.HTTPError)):
        _sweep.fetch_window(_SPEC, client, ["company:bad"], "2024-01-01", "2024-02-01", max_days=7)


def test_fetch_window_healthy_entities_need_no_entity_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing poisoned, chunking alone suffices — no entity splitting is attempted."""
    monkeypatch.setattr(_sweep, "BACKOFF", ())
    monkeypatch.setattr(_sweep, "SPLIT_BACKOFF", ())
    client = _PoisonClient(set())
    gaps: list[dict[str, str]] = []
    got = _sweep.fetch_window(
        _SPEC, client, ["company:a", "company:b"], "2024-01-01", "2024-02-01", max_days=7, gaps=gaps
    )
    assert gaps == []
    assert all(e["n"] == 2 for e in got), "entities stay together when nothing fails"


def test_gap_report_reads_recorded_gaps(tmp_path: Path) -> None:
    """Gaps are recovered from the manifest and deduplicated across re-runs."""
    d = tmp_path / "dump"
    d.mkdir()
    lines = [
        {"batch": 1, "period": "2025-12", "rows": 10},
        {
            "batch": 1,
            "period": "2025-12",
            "rows": 10,
            "gap_entity_days": 2,
            "gaps": [{"entity": "company:a", "date": "2025-12-18"}, {"entity": "company:b", "date": "2025-12-19"}],
        },
        # a re-run records the same gap again; it must not be double counted
        {
            "batch": 1,
            "period": "2025-12",
            "rows": 10,
            "gap_entity_days": 1,
            "gaps": [{"entity": "company:a", "date": "2025-12-18"}],
        },
    ]
    (d / "_manifest.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    got = verify.gap_report(d)
    assert got.height == 2
    assert set(got["entity"]) == {"company:a", "company:b"}


def test_gap_report_on_a_clean_sweep_is_empty(tmp_path: Path) -> None:
    """No gaps recorded means an empty frame with the right schema, not a crash."""
    d = tmp_path / "dump"
    d.mkdir()
    (d / "_manifest.jsonl").write_text(json.dumps({"batch": 0, "period": "2024-01", "rows": 5}))
    got = verify.gap_report(d)
    assert got.is_empty()
    assert set(got.columns) == {"entity", "date", "batch", "period"}


def test_month_end_dates_picks_the_last_trading_day_per_month() -> None:
    """Month-end means the last date actually present, not the calendar last day."""
    dates = pl.Series(
        "date",
        [date(2024, 1, 29), date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 29), date(2024, 3, 28)],
    )
    got = run_ic.month_end_dates(dates).to_list()
    assert got == [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 28)]


def test_restrict_daily_is_a_passthrough_and_month_end_filters() -> None:
    """Restriction happens after the IC is computed, so both schemes score identical signals."""
    ic = pl.DataFrame({"date": [date(2024, 1, 30), date(2024, 1, 31), date(2024, 2, 29)], "ic": [0.1, 0.2, 0.3]})
    assert run_ic.restrict(ic, "daily").equals(ic)
    got = run_ic.restrict(ic, "month-end")
    assert got["date"].to_list() == [date(2024, 1, 31), date(2024, 2, 29)]
    assert got["ic"].to_list() == [0.2, 0.3]


def test_mask_membership_nulls_non_member_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    """A price outside its membership window must be nulled, not silently ranked.

    Without this the cross-section includes companies before they joined the index and after they
    left — 5% of cells over two years, and far more over the ten-year history.
    """
    ranks = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "AAA.DE": [10, 10, None],  # leaves the index on the 4th
            "BBB.L": [None, 300, 300],  # joins on the 3rd
        }
    )
    monkeypatch.setattr("yukka.data.Index", type("I", (), {"STOXX600": type("S", (), {"membership_intervals": ranks})}))
    from honey.event_exclusion_exp import prices as prices_mod

    wide = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "AAA.DE": [1.0, 2.0, 3.0],
            "BBB.L": [4.0, 5.0, 6.0],
        }
    )
    got = prices_mod.mask_membership(wide).sort("date")
    assert got["AAA.DE"].to_list() == [1.0, 2.0, None]
    assert got["BBB.L"].to_list() == [None, 5.0, 6.0]


def test_mask_membership_respects_a_rank_band(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank band keeps only members inside it, for reproducing the production top-N universe."""
    ranks = pl.DataFrame({"date": [date(2024, 1, 2)], "AAA.DE": [10], "BBB.L": [300]})
    monkeypatch.setattr("yukka.data.Index", type("I", (), {"STOXX600": type("S", (), {"membership_intervals": ranks})}))
    from honey.event_exclusion_exp import prices as prices_mod

    wide = pl.DataFrame({"date": [date(2024, 1, 2)], "AAA.DE": [1.0], "BBB.L": [4.0]})
    got = prices_mod.mask_membership(wide, (1, 200))
    assert got["AAA.DE"][0] == 1.0
    assert got["BBB.L"][0] is None


def test_mask_membership_leaves_unknown_columns_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A column the membership panel has never heard of is kept, not silently zeroed out."""
    ranks = pl.DataFrame({"date": [date(2024, 1, 2)], "AAA.DE": [10]})
    monkeypatch.setattr("yukka.data.Index", type("I", (), {"STOXX600": type("S", (), {"membership_intervals": ranks})}))
    from honey.event_exclusion_exp import prices as prices_mod

    wide = pl.DataFrame({"date": [date(2024, 1, 2)], "AAA.DE": [1.0], "MYSTERY.XX": [9.0]})
    got = prices_mod.mask_membership(wide)
    assert got["MYSTERY.XX"][0] == 9.0


def _write_sweep(tmp: Path, *, total: int, shards: list[str], records: list[dict[str, object]]) -> Path:
    """A fake sweep directory: plan, shard files and manifest lines."""
    d = tmp / "dump"
    d.mkdir(parents=True, exist_ok=True)
    (d / "_plan.json").write_text(
        json.dumps({"dump": "events_dump", "total_shards": total, "batches": 2, "periods": ["2024-01", "2024-02"]})
    )
    for name in shards:
        (d / name).write_bytes(b"x" * 100)
    (d / "_manifest.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    return d


def test_sweep_status_does_not_count_refetched_failures_as_outstanding(tmp_path: Path) -> None:
    """A shard that failed once and succeeded later must not make a finished sweep look broken."""
    now = time.time()
    d = _write_sweep(
        tmp_path,
        total=2,
        shards=["dump-b000-2024-01.parquet", "dump-b001-2024-01.parquet"],
        records=[
            {"batch": 0, "period": "2024-01", "error": "500", "ts": now - 500},
            {"batch": 0, "period": "2024-01", "rows": 5, "bytes": 100, "ts": now - 200},
            {"batch": 1, "period": "2024-01", "rows": 7, "bytes": 100, "ts": now - 100},
        ],
    )
    s = sweep_status.read_state(d)
    assert s["errors"] == 0, "the failure was later re-fetched"
    assert s["stale_errors"] == 1
    assert "COMPLETE" in sweep_status.render_text(s)


def test_sweep_status_reports_a_still_missing_shard(tmp_path: Path) -> None:
    """A failure whose shard is absent is a real outstanding problem and must be surfaced."""
    now = time.time()
    d = _write_sweep(
        tmp_path,
        total=2,
        shards=["dump-b000-2024-01.parquet"],
        records=[
            {"batch": 0, "period": "2024-01", "rows": 5, "bytes": 100, "ts": now - 200},
            {"batch": 1, "period": "2024-01", "error": "500 boom", "ts": now - 100},
        ],
    )
    s = sweep_status.read_state(d)
    assert s["errors"] == 1
    assert "boom" in s["last_error"]
    assert "COMPLETE" not in sweep_status.render_text(s)


def test_sweep_status_counts_complete_periods_and_gaps(tmp_path: Path) -> None:
    """A month short of a batch is not complete — that silently narrows the cross-section."""
    now = time.time()
    d = _write_sweep(
        tmp_path,
        total=4,
        shards=["dump-b000-2024-01.parquet", "dump-b001-2024-01.parquet", "dump-b000-2024-02.parquet"],
        records=[
            {"batch": 0, "period": "2024-01", "rows": 5, "bytes": 100, "ts": now - 300},
            {"batch": 1, "period": "2024-01", "rows": 5, "bytes": 100, "gap_entity_days": 3, "ts": now - 200},
            {"batch": 0, "period": "2024-02", "rows": 5, "bytes": 100, "ts": now - 100},
        ],
    )
    s = sweep_status.read_state(d)
    assert s["complete_periods"] == 1
    assert s["furthest_complete"] == "2024-01"
    assert s["partial"] == [("2024-02", 1)]
    assert s["gaps"] == 3
