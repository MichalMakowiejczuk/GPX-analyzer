import pandas as pd
import pytest

from scripts.profile.slope_analyzer import SlopeAnalyzer


@pytest.fixture
def simple_track():
    """Krótki track 3 punktowy z podjazdem i zjazdem"""
    return pd.DataFrame(
        {
            "km": [0.0, 1.0, 2.0],
            "elev_smooth": [100, 150, 120],
            "segment": [1, 1, 1],
        }
    )


@pytest.fixture
def flat_track():
    return pd.DataFrame(
        {
            "km": [0.0, 1.0, 2.0],
            "elev_smooth": [100, 100, 100],
            "segment": [1, 1, 1],
        }
    )


def test_delta_computation(simple_track):
    analyzer = SlopeAnalyzer(simple_track)
    df = analyzer.df
    # delta_elev
    assert df["delta_elev"].iloc[1] == 50
    assert df["delta_elev"].iloc[2] == -30
    # delta_km
    assert df["delta_km"].iloc[1] == 1.0
    # slope_pct
    assert df["slope_pct"].iloc[1] == pytest.approx(50 / 1000 * 100, rel=1e-2)
    assert df["slope_pct"].iloc[2] == pytest.approx(-30 / 1000 * 100, rel=1e-2)


def test_total_ascent_and_descent(simple_track):
    analyzer = SlopeAnalyzer(simple_track)
    assert analyzer.get_total_ascent() == 50
    assert analyzer.get_total_descent() == 30


def test_highest_and_lowest_point(simple_track):
    analyzer = SlopeAnalyzer(simple_track)
    assert analyzer.get_highest_point() == 150
    assert analyzer.get_lowest_point() == 100


def test_flat_track(flat_track):
    analyzer = SlopeAnalyzer(flat_track)
    assert analyzer.get_total_ascent() == 0
    assert analyzer.get_total_descent() == 0
    assert analyzer.get_highest_point() == 100
    assert analyzer.get_lowest_point() == 100


def test_compute_slope_lengths(simple_track):
    analyzer = SlopeAnalyzer(simple_track)
    result = analyzer.compute_slope_lengths(slope_thresholds=(0, 5, 10))
    # powinno powstać kilka grup nachylenia
    assert "length_km" in result.columns
    assert result["length_km"].sum() == pytest.approx(2.0, rel=1e-2)
    # procenty sumują się ~100%
    assert result["% of total"].sum() == pytest.approx(100.0, rel=1e-2)
