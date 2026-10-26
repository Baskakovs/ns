"""Tests for the exclusion-invariance measurement."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from honey.novelty_score_exp import invariance as inv


@pytest.fixture
def scores():
    """A synthetic panel: enough dates to clear the EW warm-up, enough names for a cross-section."""
    rng = np.random.default_rng(7)
    dates = pl.date_range(pl.date(2024, 7, 1), pl.date(2024, 12, 31), eager=True)
    rows = []
    for d in dates:
        for e in [f"E{i:02d}" for i in range(30)]:
            for _ in range(rng.integers(3, 9)):
                rows.append(
                    {
                        "date": d,
                        "entity": e,
                        "sentiment": int(rng.choice([-1, 0, 1])),
                        "novelty": float(rng.random()),
                    }
                )
    return pl.DataFrame(rows)


class TestArticleMass:
    """Counting what survives a rule."""

    def test_mass_is_the_three_columns_summed(self, scores):
        """Baseline mass equals the article count, since every article lands in exactly one bucket."""
        assert inv.article_mass(inv.baseline_long(scores)) == scores.height

    def test_a_threshold_keeps_less_than_everything(self, scores):
        """Filtering on novelty strictly reduces the counted mass."""
        from honey.novelty_score_exp import novelty_knn_model as nk

        assert inv.article_mass(nk.threshold_long(scores, 0.5)) < inv.article_mass(inv.baseline_long(scores))


class TestCompare:
    """The displacement measures."""

    def test_a_panel_against_itself_is_a_perfect_match(self, scores):
        """Identity is the calibration point: r and rho are 1 and the baskets coincide."""
        sig, cols = inv.production_signal(inv.baseline_long(scores))
        d = inv.compare("self", 1.0, sig, cols, sig, cols)
        assert d.pooled_r == pytest.approx(1.0)
        assert d.rank_rho == pytest.approx(1.0)
        assert d.basket_overlap == pytest.approx(1.0)

    def test_uniform_thinning_barely_moves_the_signal(self, scores):
        """The invariance itself: the ratio is a proportion, so a sentiment-neutral drop mostly cancels."""
        base, bcols = inv.production_signal(inv.baseline_long(scores))
        thin, tcols = inv.production_signal(inv.random_drop(scores, 0.5, seed=0))
        assert inv.compare("thin", 0.5, base, bcols, thin, tcols).pooled_r > 0.5


class TestWithinCellTilt:
    """The weighting path's alpha budget."""

    def test_no_tilt_when_novelty_is_independent_of_sentiment(self, scores):
        """Random novelty gives a tilt indistinguishable from zero next to the within-cell spread."""
        t = inv.within_cell_tilt(scores).row(0, named=True)
        assert abs(t["tilt_mean"]) < t["within_cell_sd_median"]

    def test_tilt_is_detected_when_it_is_constructed(self):
        """A cell where positives are novel and negatives are not shows up with the right sign."""
        rows = [
            {"date": dt.date(2024, 7, 1), "entity": "E", "sentiment": s, "novelty": nov}
            for s, nov in [(1, 0.9), (1, 0.8), (-1, 0.1), (-1, 0.2)]
        ]
        t = inv.within_cell_tilt(pl.DataFrame(rows)).row(0, named=True)
        assert t["tilt_mean"] == pytest.approx(0.7)

    def test_cells_below_the_minimum_are_excluded(self):
        """A two-article cell is too thin to carry a tilt estimate."""
        rows = [
            {"date": dt.date(2024, 7, 1), "entity": "E", "sentiment": 1, "novelty": 0.9},
            {"date": dt.date(2024, 7, 1), "entity": "E", "sentiment": -1, "novelty": 0.1},
        ]
        assert inv.within_cell_tilt(pl.DataFrame(rows), min_articles=4).row(0, named=True)["cells"] == 0


class TestLoadScores:
    """Locating a built scores parquet."""

    def test_a_missing_window_says_what_is_built(self, tmp_path):
        """The error names the available windows rather than just failing to find a path."""
        with pytest.raises(FileNotFoundError, match="no scores parquet for window"):
            inv.load_scores("999h", 5, tmp_path)


class TestProductionParity:
    """The chain under test must be the production one."""

    def test_parameters_come_from_prod_params(self):
        """Hardcoding these would let the measurement drift away from the book it describes."""
        from honey.novelty_score_exp import novelty_knn_model as nk

        assert (
            nk.PROD_PARAMS["span"],
            nk.PROD_PARAMS["clip_c"],
            nk.PROD_PARAMS["threshold"],
        ) == (inv.SPAN, inv.CLIP_C, inv.THRESHOLD)
