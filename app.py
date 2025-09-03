import streamlit as st

from components.climbs_section import render_climbs_section
from components.profile_section import render_main_profile, render_segment_profile
from components.sidebar import render_sidebar
from components.slope_table_section import render_slope_table_and_plot
from components.stats_section import render_main_stats_and_map
from services.analysis_service import compute_route_stats
from services.climb_service import prepare_climbs
from services.gpx_service import load_gpx
from services.map_service import generate_map


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
        st.info(
            "Upload a GPX file (or select an example) to analyze its elevation profile and detect climbs."
        )
        st.stop()

    # Load and process GPX data
    track_df, profile = load_gpx(uploaded_file, smooth_window)
    stats = compute_route_stats(profile)

    # Climb detection
    climbs_df = prepare_climbs(profile, min_length, min_avg_slope, merge_gap_m)

    # Map
    base_map = generate_map(profile.get_route_data(), climbs_df)

    # UI
    render_main_stats_and_map(stats, base_map)
    render_main_profile(profile, slope_thresholds)
    render_segment_profile(track_df, smooth_window, slope_thresholds)
    render_slope_table_and_plot(profile, slope_thresholds)
    render_climbs_section(climbs_df, track_df, slope_thresholds, smooth_window)

    st.markdown("---")
    st.caption(
        "© 2025 Michał Makowiejczuk | GPX Profile Analyzer | hobby project | GitHub: https://github.com/MichalMakowiejczuk/GPX-analyzer"
    )


if __name__ == "__main__":
    main()
