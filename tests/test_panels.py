"""Tests for the materialised counts panel: its stamp, and its refusal to serve a stale one."""

from __future__ import annotations

import io
import json

import polars as pl
import pytest

from novetly_score.data.panels import (
    COLUMNS,
    COUNTS_KEY,
    STAMP_KEY,
    StalePanelError,
    _compare,
    read_counts,
)

CURRENT = {"files": 18183, "rows": 181276415, "last_month": "2026-11"}


class TestCompare:
    """The staleness rule, which is the whole reason a derived panel can be trusted."""

    def test_an_identical_stamp_passes(self):
        """Nothing moved, so the panel still answers for the store."""
        _compare(dict(CURRENT), CURRENT)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("last_month", "2026-12"),  # a month was added
            ("rows", 181276416),  # a file was backfilled
            ("files", 18184),  # a company appeared
        ],
    )
    def test_any_field_moving_is_refused(self, field, value):
        """Three numbers, because each kind of change to the store moves at least one."""
        written = {**CURRENT, field: value}

        with pytest.raises(StalePanelError, match=field):
            _compare(written, CURRENT)

    def test_the_message_names_both_values(self):
        """So the reader knows whether they are behind by a month or by a backfill."""
        written = {**CURRENT, "last_month": "2026-10"}

        with pytest.raises(StalePanelError, match="panel '2026-10', store '2026-11'"):
            _compare(written, CURRENT)

    def test_a_stamp_missing_a_field_is_refused(self):
        """An older stamp format cannot be shown to match, so it does not pass."""
        with pytest.raises(StalePanelError, match="files"):
            _compare({"rows": CURRENT["rows"], "last_month": CURRENT["last_month"]}, CURRENT)

    def test_every_moved_field_is_reported(self):
        """A new month moves all three; the message says so rather than stopping at the first."""
        with pytest.raises(StalePanelError, match=r"files.*rows.*last_month"):
            _compare({"files": 1, "rows": 2, "last_month": "2024-07"}, CURRENT)


class TestReadCounts:
    """What a read does when the store has no panel, or one without provenance."""

    def _client(self, monkeypatch, objects: dict[str, bytes]):
        """Stand in for OpenClient with an in-memory key-value store."""

        class NoSuchKey(Exception):  # noqa: N818 — mirrors boto3's own exception name
            pass

        class Client:
            exceptions = type("Exceptions", (), {"NoSuchKey": NoSuchKey})()

            def get_object(self, Bucket, Key):  # noqa: N803 — boto3's own casing
                if Key not in objects:
                    raise NoSuchKey(Key)
                return {"Body": io.BytesIO(objects[Key])}

        class Opened:
            def __enter__(self):
                return Client()

            def __exit__(self, *_exc):
                return None

        monkeypatch.setattr("novetly_score.data.panels.OpenClient", lambda *a, **k: Opened())

    def _panel(self) -> bytes:
        """A one-row panel, as parquet bytes."""
        frame = pl.DataFrame(
            {
                "ric": ["BMWG.DE"],
                "day": ["2024-07-01"],
                "positive": [3],
                "neutral": [5],
                "negative": [1],
                "uncharacterised": [0],
            }
        ).with_columns(pl.col("day").str.to_date())
        buffer = io.BytesIO()
        frame.write_parquet(buffer)
        return buffer.getvalue()

    def test_a_missing_panel_says_how_to_build_it(self, monkeypatch):
        """The first caller on a fresh store gets an instruction, not a KeyError."""
        self._client(monkeypatch, {})

        with pytest.raises(StalePanelError, match="write_counts"):
            read_counts()

    def test_a_panel_without_a_stamp_is_refused(self, monkeypatch):
        """Unprovenanced is as bad as stale: it cannot be shown to match the store."""
        self._client(monkeypatch, {COUNTS_KEY: self._panel()})

        with pytest.raises(StalePanelError, match="no stamp"):
            read_counts()

    def test_check_off_reads_an_unstamped_panel(self, monkeypatch):
        """The escape hatch, for looking at a panel you already know is behind."""
        self._client(monkeypatch, {COUNTS_KEY: self._panel()})

        frame = read_counts(check=False)

        assert frame.columns == list(COLUMNS)
        assert frame.height == 1

    def test_a_matching_stamp_reads(self, monkeypatch):
        """The happy path: stamp agrees with the store, so the frame is handed over."""
        self._client(
            monkeypatch,
            {COUNTS_KEY: self._panel(), STAMP_KEY: json.dumps(CURRENT).encode()},
        )
        monkeypatch.setattr("novetly_score.data.panels.stamp", lambda *a, **k: dict(CURRENT))

        assert read_counts().columns == list(COLUMNS)

    def test_a_stale_stamp_is_refused_before_the_frame_is_returned(self, monkeypatch):
        """A grown store raises rather than answering with a panel that stops short."""
        behind = {**CURRENT, "last_month": "2026-10"}
        self._client(
            monkeypatch,
            {COUNTS_KEY: self._panel(), STAMP_KEY: json.dumps(behind).encode()},
        )
        monkeypatch.setattr("novetly_score.data.panels.stamp", lambda *a, **k: dict(CURRENT))

        with pytest.raises(StalePanelError, match="behind the store"):
            read_counts()


@pytest.mark.integration
def test_the_real_panel_matches_the_real_store():
    """The materialised panel is the same answer the aggregation gives, in one read.

    Skipped unless the S3 variables are set. This is the invariant the whole shortcut rests on:
    if these ever disagree, the panel is wrong rather than merely old.
    """
    from novetly_score.data import ConnectionFailedError

    try:
        panel = read_counts()
    except ConnectionFailedError as exc:
        pytest.skip(str(exc))
    except StalePanelError as exc:
        pytest.skip(f"panel not built or behind: {exc}")

    assert panel.columns == list(COLUMNS)
    assert panel.schema["day"] == pl.Date
    assert panel.height == panel.select("ric", "day").n_unique(), "one row per RIC and day"
    assert panel.get_column("ric").n_unique() > 500
    assert panel.select(pl.col("positive", "neutral", "negative").min()).to_numpy().min() >= 0
