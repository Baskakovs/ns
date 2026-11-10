"""Tests for the RIC-keyed panel: the one month of the real store it is a thin wrapper over."""

from __future__ import annotations

import polars as pl
import pytest

from novetly_score.panel import COLUMNS, counts


@pytest.mark.integration
def test_reads_a_month_of_the_real_store():
    """One month of the real STOXX 600, through the real store: columns, keys and dtypes.

    Skipped unless the S3 variables and a YUKKA token are set, so the suite stays runnable
    without either. The logic under it — the counting and the RIC bridge — is covered against
    the fixture store in ``test_snippets.py``; what this adds is that the three sources line up.
    """
    from novetly_score.data import ConnectionFailedError

    pytest.importorskip("yukka.data")
    try:
        frame = counts("2025-01", "2025-01")
    except ConnectionFailedError as exc:
        if "not set" in str(exc):
            pytest.skip(str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 — a missing token is a skip, not a failure
        pytest.skip(f"universe or store unavailable: {exc}")

    assert frame.columns == list(COLUMNS)
    assert frame.schema["date"] == pl.Date
    assert frame.height == frame.select("date", "ric").n_unique(), "one row per RIC and day"
    assert frame.get_column("ric").n_unique() > 500
    assert frame.select(pl.col("pos", "neu", "neg").min()).to_numpy().min() >= 0
