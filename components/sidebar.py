import os

import streamlit as st


def parse_thresholds(threshold_str: str):
    try:
        return tuple(
            float(x.strip()) for x in threshold_str.split(",") if x.strip() != ""
        )
    except:
        st.warning("Incorrect threshold format - I'm using the default ones: 2,4,6,8")
        return (2, 4, 6, 8)


def render_sidebar():
    st.sidebar.header("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload GPX file:", type="gpx")

    # Pliki przykładowe
    data_dir = "sample_data"
    example_files = (
        [f for f in os.listdir(data_dir) if f.endswith(".gpx")]
        if os.path.isdir(data_dir)
        else []
    )
    options = ["---"] + example_files
    selected_example = st.sidebar.selectbox("Or choose an example:", options)

    if not uploaded_file and selected_example != "---":
        uploaded_file = open(os.path.join(data_dir, selected_example), "rb")

    st.sidebar.subheader("Climb detection settings")
    min_length = st.sidebar.number_input(
        "Minimal climb length [m]", min_value=100, max_value=30000, value=500, step=250
    )
    min_avg_slope = st.sidebar.number_input(
        "Minimal average grade (%)", min_value=2.0, max_value=16.0, value=3.0, step=0.5
    )
    merge_gap_m = st.sidebar.number_input(
        "Maximum descent/platau length during an ascent [m]",
        min_value=0,
        max_value=2000,
        value=100,
        step=50,
    )
    smooth_window = st.sidebar.number_input(
        "Profile smoothing window (rolling mean)",
        min_value=1,
        max_value=20,
        value=5,
        step=2,
    )

    st.sidebar.subheader("Slope ranges for profile coloring")
    slope_thresholds_str = st.sidebar.text_input(
        "Thresholds (in %), separated by commas", value="2,4,6,8"
    )
    slope_thresholds = parse_thresholds(slope_thresholds_str)

    return (
        uploaded_file,
        min_length,
        min_avg_slope,
        merge_gap_m,
        smooth_window,
        slope_thresholds,
    )
