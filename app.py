import numpy as np
import streamlit as st

from components.climbs_section import render_climbs_section
from components.profile_section import render_segment_profile, render_slope_table
from components.sidebar import render_sidebar
from components.stats_section import render_main_stats_and_map
from scripts.climb_classification import classify_climb_difficulty
from services.analysis_service import compute_route_stats, detect_climbs
from services.gpx_service import load_gpx
from services.map_service import generate_map
from services.plot_service import plot_main_profile


def main() -> None:
    st.set_page_config(page_title="GPX profile analyzer", layout="wide", page_icon="🚴")
    st.title("GPX profile analyzer")
    (
        uploaded_file,
        min_length,
        min_avg_slope,
        merge_gap_m,
        smooth_window,
        slope_thresholds,
    ) = render_sidebar()

    if uploaded_file is None:
        st.info("Upload a GPX file to analyze its elevation profile and detect climbs.")
        st.stop()

    # Wczytanie danych
    track_df, profile = load_gpx(uploaded_file, smooth_window)
    stats = compute_route_stats(profile)

    # Wykres główny
    main_profile_plot = plot_main_profile(profile, slope_thresholds)

    # Wykrywanie podjazdów
    climbs_df = detect_climbs(profile, min_length, min_avg_slope, merge_gap_m)
    if not climbs_df.empty:
        climbs_df.index = np.arange(1, len(climbs_df) + 1)
        climbs_df["start_km"] = climbs_df["start_km"].round(2)
        climbs_df["end_km"] = climbs_df["end_km"].round(2)
        climbs_df["start-end km"] = (
            climbs_df["start_km"]
            .astype(str)
            .str.cat(climbs_df["end_km"].astype(str), sep=" - ")
            + " km"
        )
        climbs_df["Difficulty score"] = climbs_df.apply(
            lambda row: classify_climb_difficulty(row.length_m, row.avg_grade_pct)[1],
            axis=1,
        )
        climbs_df["Difficulty category"] = climbs_df.apply(
            lambda row: classify_climb_difficulty(row.length_m, row.avg_grade_pct)[0],
            axis=1,
        )

    # Mapa
    base_map = generate_map(profile.get_route_data(), climbs_df)

    # Sekcje UI
    render_main_stats_and_map(stats, base_map)

    with st.expander("Main elevation profile", expanded=True):
        st.pyplot(main_profile_plot, use_container_width=True)

    render_segment_profile(track_df, smooth_window, slope_thresholds)
    render_slope_table(profile, slope_thresholds)
    render_climbs_section(climbs_df, track_df, slope_thresholds, smooth_window)

    st.markdown("---")
    st.caption(
        "© 2025 Michał Makowiejczuk | GPX Profile Analyzer | hobby project | GitHub: https://github.com/MichalMakowiejczuk/GPX-analyzer"
    )


if __name__ == "__main__":
    main()
