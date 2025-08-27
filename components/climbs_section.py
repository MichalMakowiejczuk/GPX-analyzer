import streamlit as st
import matplotlib.pyplot as plt
from scripts.profile import ElevationProfile
from scripts.climb_classification import classify_climb_difficulty

def render_climbs_section(climbs_df, track_df, slope_thresholds, smooth_window):
    if climbs_df.empty:
        st.warning("No climbs detected for the given parameters.")
        return

    with st.expander("Detected climbs", expanded=False):
        st.dataframe(climbs_df[['start-end km', 'length_m', 'gain_m', 'avg_grade_pct', 'Difficulty category']],
                     use_container_width=True)

    st.subheader("Elevation profiles of detected climbs")
    tab_titles = [f"Climb {i+1}" for i in range(len(climbs_df))]
    tabs = st.tabs(tab_titles)

    for i, row in enumerate(climbs_df.itertuples()):
        with tabs[i]:
            climb_df = track_df.iloc[
                track_df['km'].sub(row.start_km).abs().idxmin():
                track_df['km'].sub(row.end_km).abs().idxmin() + 1
            ].reset_index(drop=True)

            if len(climb_df) < 3:
                st.info("Too short climb segment to display profile.")
                continue

            climb_df["km"] -= climb_df["km"].iloc[0]
            seg_unit_km = 0.25 if row.length_m < 3000 else 0.5

            climb_profile = ElevationProfile(climb_df, seg_unit_km=seg_unit_km, smooth_window=smooth_window)
            fig_c, ax_c = climb_profile.plot_profile(show_labels=False,
                                                     show_background=False,
                                                     slope_thresholds=slope_thresholds,
                                                     slope_type="segment")
            ax_c.set_ylim(climb_df["elevation"].min(), climb_df["elevation"].max())

            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Length", f"{row.length_m} m")
                st.caption(f"**from** {row.start_km:.2f} km **to** {row.end_km:.2f} km of track.")
                st.metric("Ascent", f"{row.gain_m} m")
                sub_col1, sub_col2 = st.columns([1, 1])
                with sub_col1:
                    st.metric("AVG. slope", f"{row.avg_grade_pct} %")
                with sub_col2:
                    st.metric("Max. slope", f"{row.max_grade_pct} %")
                st.metric("Climb category", classify_climb_difficulty(row.length_m, row.avg_grade_pct)[0])
            with col2:
                st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)