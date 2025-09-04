import pandas as pd
import pytest

from scripts.profile import ElevationProfile


@pytest.fixture
def simple_track_df():
    return pd.DataFrame(
        {
            "km": [0.0, 1.0, 2.0, 3.0],
            "latitude": [50.0, 50.001, 50.002, 50.003],
            "longitude": [19.9, 19.91, 19.92, 19.93],
            "elevation": [200, 210, 220, 230],
        }
    )


def test_summary_contains_expected_keys(simple_track_df):
    profile = ElevationProfile(simple_track_df)
    summary = profile.summary()

    expected_keys = {
        "distance_km",
        "total_ascent_m",
        "total_descent_m",
        "highest_point_m",
        "lowest_point_m",
        "AVG slope (%)",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["distance_km"] == pytest.approx(3.0, rel=1e-3)
    assert summary["highest_point_m"] == 220


def test_total_ascent_and_descent(simple_track_df):
    profile = ElevationProfile(simple_track_df)
    ascent = profile.get_total_ascent()
    descent = profile.get_total_descent()

    assert ascent > 0
    assert descent == 0


def test_get_route_data_returns_copy(simple_track_df):
    profile = ElevationProfile(simple_track_df)
    df1 = profile.get_route_data()
    df1["elevation"] = 999
    df2 = profile.get_route_data()
    assert (df2["elevation"] != 999).all()


def test_plot_profile_returns_matplotlib_objects(simple_track_df):
    profile = ElevationProfile(simple_track_df)
    fig, ax = profile.plot_profile()
    import matplotlib

    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
