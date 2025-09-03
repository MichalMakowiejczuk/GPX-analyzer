from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.profile import ElevationProfile
from services.plot_service import plot_elevation_profile


def render_segment_profile(
    track_df: pd.DataFrame, smooth_window: int, slope_thresholds: Tuple
) -> None:
    with st.expander("Selected fragment of the route profile", expanded=False):
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        min_km = float(track_df["km"].min())
        max_km = float(track_df["km"].max())
        with col1:
            selected_range = st.slider(
                "Select a distance range [km]:",
                min_value=min_km,
                max_value=max_km,
                value=(min_km, max_km),
                step=0.1,
            )

        if selected_range[0] >= selected_range[1]:
            st.warning(
                "Invalid range. Make sure the starting value is less than the ending value."
            )
            return

        segment_df = track_df[
            (track_df["km"] >= selected_range[0])
            & (track_df["km"] <= selected_range[1])
        ]
        if len(segment_df) < 3:
            st.info("Too few points for a meaningful graph.")
            return

        segment_profile = ElevationProfile(
            segment_df, seg_unit_km=0.5, smooth_window=smooth_window
        )
        segment_stats = segment_profile.summary()
        with col2:
            st.metric("Distance", f"{segment_stats['distance_km']:.1f} km")
        with col3:
            st.metric("Elevation Gain", f"{segment_stats['total_ascent_m']:.0f} m")
        with col4:
            st.metric("Elevation Loss", f"{segment_stats['total_descent_m']:.0f} m")
        with col5:
            st.metric("AVG slope:", f"{segment_stats['AVG slope (%)']:.2f} %")

        fig = plot_elevation_profile(segment_profile, slope_thresholds)

        # if segment_stats["distance_km"] < 10:
        #     background_shift_km = 0.05
        #     background_shift_elev = 2
        # elif segment_stats["distance_km"] < 25:
        #     background_shift_km = 0.15
        #     background_shift_elev = 5
        # elif segment_stats["distance_km"] < 100:
        #     background_shift_km = 0.3
        #     background_shift_elev = 10
        # else:
        #     background_shift_km = 0.45
        #     background_shift_elev = 15

        # fig_s, _ = segment_profile.plot_profile(
        #     show_labels=False,
        #     show_background=True,
        #     slope_thresholds=slope_thresholds,
        #     slope_type="segment",
        #     background_shift_km=background_shift_km,
        #     background_shift_elev=background_shift_elev,
        # )

        # set y-axis limits with some margin
        elev_min = segment_df["elevation"].min()
        elev_max = segment_df["elevation"].max()
        elev_margin = (elev_max - elev_min) * 0.1
        plt.ylim(elev_min - elev_margin, elev_max + elev_margin)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def render_main_profile(
    profile: ElevationProfile, slope_thresholds: Tuple
) -> plt.Figure:
    with st.expander("Main elevation profile", expanded=True):
        fig = plot_elevation_profile(profile, slope_thresholds)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
