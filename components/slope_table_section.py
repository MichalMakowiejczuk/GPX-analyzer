from typing import Tuple

import plotly.express as px
import streamlit as st

from scripts.profile import ElevationProfile


def render_slope_table_and_plot(
    profile: ElevationProfile, slope_thresholds: Tuple
) -> None:
    slope_df = profile.compute_slope_lengths(slope_thresholds=slope_thresholds)

    slope_df = slope_df.rename(
        columns={
            "slope_range": "Slope Range",
            "length_km": "Length (km)",
        }
    )

    uphill_downhill = profile.compute_slope_lengths(slope_thresholds=(-2, 2))
    uphill_downhill["slope_range"] = uphill_downhill["slope_range"].astype(str)
    uphill_downhill.loc[:, "slope_range"] = [
        "Downhill (< -2%)",
        "Flat",
        "Uphill (> 2%)",
    ]

    fig = px.pie(
        names=uphill_downhill["slope_range"],
        values=uphill_downhill["length_km"],
        width=300,
        height=300,
    )
    fig.update_traces(
        textinfo="label+percent",
        textfont_size=15,
        showlegend=False,
        marker=dict(
            colors=["#02BCF5", "lightgreen", "orangered"],
            line=dict(color="#000000", width=2),
        ),
    )

    with st.expander("Table of lengths by slope ranges", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(slope_df, use_container_width=True, hide_index=True)
        with col2:
            st.plotly_chart(fig, use_container_width=True)
