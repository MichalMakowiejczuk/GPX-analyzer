import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from scripts.profile import ElevationProfile

def render_segment_profile(track_df, smooth_window, slope_thresholds):
    with st.expander("Selected fragment of the route profile", expanded=False):
        col1, col2 = st.columns([1, 1])
        min_km = float(track_df["km"].min())
        max_km = float(track_df["km"].max())
        with col1:
            selected_range = st.slider("Select a distance range [km]:",
                                       min_value=min_km,
                                       max_value=max_km,
                                       value=(min_km, max_km),
                                       step=0.1)
         
        if selected_range[0] >= selected_range[1]:
            st.warning("Invalid range. Make sure the starting value is less than the ending value.")
            return

        segment_df = track_df[(track_df["km"] >= selected_range[0]) & (track_df["km"] <= selected_range[1])]
        if len(segment_df) < 3:
            st.info("Too few points for a meaningful graph.")
            return

        segment_profile = ElevationProfile(segment_df, seg_unit_km=0.5, smooth_window=smooth_window)
        segment_stats = segment_profile.summary()
        with col2: 
            st.write(segment_stats)

        fig_s, _ = segment_profile.plot_profile(show_labels=False,
                                                show_background=True,
                                                slope_thresholds=slope_thresholds,
                                                slope_type="segment")
        
        # set y-axis limits with some margin
        elev_min = segment_df["elevation"].min() 
        elev_max = segment_df["elevation"].max()
        elev_margin = (elev_max - elev_min) * 0.1
        plt.ylim(elev_min - elev_margin, elev_max + elev_margin)

        st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)


def render_slope_table(profile, slope_thresholds):
    slope_df = profile.compute_slope_lengths(slope_thresholds=slope_thresholds)
    uphill_downhill = profile.compute_slope_lengths(slope_thresholds=(-2, 2))
    uphill_downhill.loc[:, 'slope_range'] = ['Downhill (< -2%)', 'Flat', 'Uphill (> 2%)']

    fig = px.pie(names=uphill_downhill['slope_range'],
                 values=uphill_downhill['length_km'],
                 width=300,
                 height=300)
    fig.update_traces(textinfo='label+percent',
                      textfont_size=15,
                      showlegend=False,
                      marker=dict(colors=["#02BCF5", 'lightgreen', 'orangered'], line=dict(color='#000000', width=2)))

    with st.expander("Table of lengths by slope ranges", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(slope_df, use_container_width=True, hide_index=True)
        with col2:
            st.plotly_chart(fig, use_container_width=True)