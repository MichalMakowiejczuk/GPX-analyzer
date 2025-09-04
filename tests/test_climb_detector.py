import pandas as pd
import pytest

from scripts.profile.climb_detector import ClimbDetector


@pytest.fixture
def flat_track():
    """3 km route with no elevation change."""
    return pd.DataFrame(
        {
            "km": [0.0, 1.0, 2.0, 3.0],
            "elev_smooth": [100, 100, 100, 100],
            "latitude": [50.0, 50.001, 50.002, 50.003],
            "longitude": [19.9, 19.91, 19.92, 19.93],
        }
    )


@pytest.fixture
def simple_climb():
    """3 km route with elevation gain of 150m."""
    return pd.DataFrame(
        {
            "km": [0.0, 1.0, 2.0, 3.0],
            "elev_smooth": [100, 150, 200, 250],
            "latitude": [50.0, 50.001, 50.002, 50.003],
            "longitude": [19.9, 19.91, 19.92, 19.93],
            "datapoint_slope": [0.0, 5.0, 5.0, 5.0],
        }
    )


def test_empty_dataframe_returns_empty():
    df = pd.DataFrame(columns=["km", "elev_smooth", "latitude", "longitude"])
    detector = ClimbDetector(df)
    result = detector.detect()
    assert result.empty


def test_flat_track_no_climbs(flat_track):
    detector = ClimbDetector(flat_track)
    result = detector.detect()
    assert result.empty


def test_simple_climb_detected(simple_climb):
    detector = ClimbDetector(simple_climb)
    result = detector.detect(min_length_m=500, min_avg_slope=2.0)
    assert len(result) == 1

    climb = result.iloc[0]
    assert climb["length_m"] == pytest.approx(2000.0, rel=1e-2)
    assert climb["gain_m"] == pytest.approx(100.0, rel=1e-2)
    assert climb["avg_grade_pct"] > 4.0


def test_too_short_climb_is_filtered(simple_climb):
    detector = ClimbDetector(simple_climb.iloc[:2])  # only first 1 km
    result = detector.detect(min_length_m=1500)
    assert result.empty


def test_merge_segments():
    """Test route with two climbs separated by a short descent."""
    df = pd.DataFrame(
        {
            "km": [0.0, 0.5, 1.0, 1.5, 2, 2.5],
            "elev_smooth": [100, 130, 160, 100, 120, 140],
            "latitude": [50.000, 50.001, 50.002, 50.003, 50.004, 50.005],
            "longitude": [19.900, 19.910, 19.920, 19.930, 19.940, 19.950],
        }
    )
    detector = ClimbDetector(df)

    result = detector.detect(min_length_m=100, merge_gap_m=1000)

    assert len(result) == 1, "Expected 1 merged climb segment"
