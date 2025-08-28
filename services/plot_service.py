from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from scripts.profile import ElevationProfile


def plot_main_profile(profile: ElevationProfile, slope_thresholds: Tuple) -> plt.Figure:
    fig, ax = profile.plot_profile(
        show_labels=False, show_background=True, slope_thresholds=slope_thresholds
    )
    total_distance_km = float(profile.get_route_data()["km"].max())
    if total_distance_km < 10:
        ax.set_xticks(np.arange(0, total_distance_km, 1))
    elif total_distance_km < 50:
        ax.set_xticks(np.arange(0, total_distance_km, 5))
    else:
        ax.set_xticks(np.arange(0, total_distance_km, 10))
    return fig
