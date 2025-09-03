from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from scripts.profile import ElevationProfile


def plot_elevation_profile(
    profile: ElevationProfile, slope_thresholds: Tuple
) -> plt.Figure:
    profile_stats = profile.summary()

    total_distance_km = profile_stats["distance_km"]
    if profile_stats["distance_km"] < 10:
        background_shift_km = 0.05
        background_shift_elev = 2
    elif profile_stats["distance_km"] < 25:
        background_shift_km = 0.15
        background_shift_elev = 5
    elif profile_stats["distance_km"] < 100:
        background_shift_km = 0.3
        background_shift_elev = 10
    else:
        background_shift_km = 0.45
        background_shift_elev = 15

    fig, ax = profile.plot_profile(
        show_labels=False,
        show_background=True,
        slope_thresholds=slope_thresholds,
        background_shift_elev=background_shift_elev,
        background_shift_km=background_shift_km,
    )

    if total_distance_km < 25:
        ax.set_xticks(np.arange(0, total_distance_km, 1))

    elif total_distance_km < 125:
        ax.set_xticks(np.arange(0, total_distance_km, 5))
    else:
        ax.set_xticks(np.arange(0, total_distance_km, 10))
    return fig
