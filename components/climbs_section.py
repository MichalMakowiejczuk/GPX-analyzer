from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from scripts.climb_classification import classify_climb_difficulty
from scripts.profile import ElevationProfile


def render_climbs_section(
    climbs_df: pd.DataFrame,
    track_df: pd.DataFrame,
    slope_thresholds: Tuple,
    smooth_window: int,
) -> None:
    if climbs_df.empty:
        st.warning("No climbs detected for the given parameters.")
        return

    with st.expander("Detected climbs - Table", expanded=False):
        climbs_table = climbs_df[
            [
                "start-end km",
                "length_m",
                "gain_m",
                "avg_grade_pct",
                "Difficulty category",
            ]
        ]
        climbs_table = climbs_table.rename(
            columns={
                "start-end km": "Climb (start-end km)",
                "length_m": "Length (m)",
                "gain_m": "Elevation Gain (m)",
                "avg_grade_pct": "AVG. Slope (%)",
                "Difficulty category": "Climb Category",
            }
        )
        st.dataframe(climbs_table, use_container_width=True)

    with st.expander("Elevation profiles of detected climbs"):
        tab_titles = [f"Climb {i+1}" for i in range(len(climbs_df))]
        tabs = st.tabs(tab_titles)

        for i, row in enumerate(climbs_df.itertuples()):
            with tabs[i]:
                climb_df = track_df.iloc[
                    track_df["km"]
                    .sub(row.start_km)
                    .abs()
                    .idxmin() : track_df["km"]
                    .sub(row.end_km)
                    .abs()
                    .idxmin()
                    + 1
                ].reset_index(drop=True)

                if len(climb_df) < 3:
                    st.info("Too short climb segment to display profile.")
                    continue

                climb_df["km"] -= climb_df["km"].iloc[0]
                seg_unit_km = 0.25 if row.length_m < 3000 else 0.5

                climb_profile = ElevationProfile(
                    climb_df, seg_unit_km=seg_unit_km, smooth_window=smooth_window
                )
                fig_c, ax_c = climb_profile.plot_profile(
                    show_labels=False,
                    show_background=False,
                    slope_thresholds=slope_thresholds,
                    slope_type="segment",
                )
                ax_c.set_ylim(climb_df["elevation"].min(), climb_df["elevation"].max())

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Length", f"{row.length_m:.0f} m")
                    st.caption(
                        f"**from** {row.start_km:.2f} km **to** {row.end_km:.2f} km of route."
                    )
                    st.metric("Elevation gain", f"{row.gain_m:.0f} m")
                    st.metric("AVG. slope", f"{row.avg_grade_pct} %")
                    st.metric(
                        "Climb category",
                        classify_climb_difficulty(row.length_m, row.avg_grade_pct)[0],
                    )
                with col2:
                    st.pyplot(fig_c, use_container_width=True)
                plt.close(fig_c)
