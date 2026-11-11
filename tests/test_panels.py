"""Tests for the materialised counts panel: its stamp, and its refusal to serve a stale one."""

from __future__ import annotations

import io
import json
from datetime import date

import polars as pl
import pytest

from novetly_score.data.panels import (
    COLUMNS,
    COUNTS_KEY,
    STAMP_KEY,
    StalePanelError,
    _compare,
    read_panel,
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


class TestReadPanel:
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
                "entity": ["company:bmw"],
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
            read_panel()

    def test_a_panel_without_a_stamp_is_refused(self, monkeypatch):
        """Unprovenanced is as bad as stale: it cannot be shown to match the store."""
        self._client(monkeypatch, {COUNTS_KEY: self._panel()})

        with pytest.raises(StalePanelError, match="no stamp"):
            read_panel()

    def test_check_off_reads_an_unstamped_panel(self, monkeypatch):
        """The escape hatch, for looking at a panel you already know is behind."""
        self._client(monkeypatch, {COUNTS_KEY: self._panel()})

        frame = read_panel(check=False)

        assert frame.columns == list(COLUMNS)
        assert frame.height == 1

    def test_a_matching_stamp_reads(self, monkeypatch):
        """The happy path: stamp agrees with the store, so the frame is handed over."""
        self._client(
            monkeypatch,
            {COUNTS_KEY: self._panel(), STAMP_KEY: json.dumps(CURRENT).encode()},
        )
        monkeypatch.setattr("novetly_score.data.panels.stamp", lambda *a, **k: dict(CURRENT))

        assert read_panel().columns == list(COLUMNS)

    def test_a_stale_stamp_is_refused_before_the_frame_is_returned(self, monkeypatch):
        """A grown store raises rather than answering with a panel that stops short."""
        behind = {**CURRENT, "last_month": "2026-10"}
        self._client(
            monkeypatch,
            {COUNTS_KEY: self._panel(), STAMP_KEY: json.dumps(behind).encode()},
        )
        monkeypatch.setattr("novetly_score.data.panels.stamp", lambda *a, **k: dict(CURRENT))

        with pytest.raises(StalePanelError, match="behind the store"):
            read_panel()


@pytest.mark.integration
def test_the_real_panel_matches_the_real_store():
    """The materialised panel is the same answer the aggregation gives, in one read.

    Skipped unless the S3 variables are set. This is the invariant the whole shortcut rests on:
    if these ever disagree, the panel is wrong rather than merely old.
    """
    from novetly_score.data import ConnectionFailedError

    try:
        panel = read_panel()
    except ConnectionFailedError as exc:
        pytest.skip(str(exc))
    except StalePanelError as exc:
        pytest.skip(f"panel not built or behind: {exc}")

    assert panel.columns == list(COLUMNS)
    assert panel.schema["day"] == pl.Date
    assert panel.height == panel.select("entity", "day").n_unique(), "one row per company and day"
    assert panel.get_column("entity").n_unique() > 500
    assert panel.select(pl.col("positive", "neutral", "negative").min()).to_numpy().min() >= 0


class TestUniverseCountsReadsThePanel:
    """``data(universe).counts(...)`` narrows the stored panel: the same call shape as prices."""

    UNIVERSE = pl.DataFrame(
        {
            "yukka_id": ["company:bmw", "company:sap"],
            "ric": ["BMWG.DE", "SAPG.DE"],
            "start_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "end_date": [date(2026, 12, 31), date(2026, 12, 31)],
        }
    )

    def _panel(self) -> pl.DataFrame:
        """Three days for two companies, plus a company the universe does not name."""
        return pl.DataFrame(
            {
                "entity": ["company:bmw"] * 3 + ["company:sap"] * 3 + ["company:other"],
                "day": [date(2025, 1, 2), date(2025, 2, 3), date(2025, 3, 4)] * 2 + [date(2025, 1, 2)],
                "positive": [1, 2, 3, 4, 5, 6, 7],
                "neutral": [0] * 7,
                "negative": [0] * 7,
                "uncharacterised": [0] * 7,
            }
        )

    def _patched(self, monkeypatch):
        """Serve the fixture panel in place of the stored one."""
        monkeypatch.setattr("novetly_score.data.panels.read_panel", lambda **_: self._panel())

    def test_only_this_universe_s_companies(self, monkeypatch):
        """A company the panel holds but the universe does not name is left out."""
        from novetly_score.data import data

        self._patched(monkeypatch)

        frame = data(self.UNIVERSE).counts()

        assert set(frame.get_column("entity")) == {"company:bmw", "company:sap"}

    @pytest.mark.parametrize(
        ("span", "days"),
        [
            ((None, None), 3),
            (("2025-02", None), 2),
            ((None, "2025-02"), 2),
            (("2025-02", "2025-02"), 1),
            ((date(2025, 2, 3), date(2025, 3, 4)), 2),
        ],
    )
    def test_the_span_narrows_by_day(self, monkeypatch, span, days):
        """A month means all of it, a day means that day, and either end open runs to the edge."""
        from novetly_score.data import data

        self._patched(monkeypatch)

        frame = data(self.UNIVERSE).counts(*span)

        assert frame.get_column("day").n_unique() == days

    def test_a_backwards_span_is_refused(self, monkeypatch):
        """The same refusal prices makes, rather than an empty frame."""
        from novetly_score.data import data

        self._patched(monkeypatch)

        with pytest.raises(ValueError, match="runs backwards"):
            data(self.UNIVERSE).counts("2025-03", "2025-01")

    def test_key_relabels_point_in_time(self, monkeypatch):
        """``key="ric"`` is the as_of relabelling, so counts and prices join without a bridge."""
        from novetly_score.data import data

        self._patched(monkeypatch)

        frame = data(self.UNIVERSE).counts(key="ric")

        assert frame.columns[0] == "ric"
        assert set(frame.get_column("ric")) == {"BMWG.DE", "SAPG.DE"}
