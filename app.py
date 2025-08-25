import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px

from scripts.gpx_parser import GPXParser
from scripts.elevation_profile import ElevationProfile
from scripts.climb_classification import classify_climb_difficulty
from scripts.map_builder import build_base_map_with_detected_climbs

# =====================
# Page config
# =====================
st.set_page_config(page_title="GPX profile analyzer", layout="wide", page_icon="🚴")
st.title("GPX profile analyzer")

# =====================
# Sidebar: Upload & Settings
# =====================
with st.sidebar:
    uploaded_file = st.file_uploader("Upload GPX file:", type="gpx")
    
    data_dir = "sample_data"
    example_files = [f for f in os.listdir(data_dir) if f.endswith(".gpx")] if os.path.isdir(data_dir) else []
    options = ["---"] + example_files
    selected_example = st.selectbox("Or choose an example:", options)
    uploaded_file = uploaded_file or (open(os.path.join(data_dir, selected_example), "rb") if selected_example != "---" else None)
    
    st.subheader("Climb detection settings")
    min_length = st.number_input("Minimal climb length [m]", min_value=100, max_value=20000, value=500, step=100)
    min_gain = st.number_input("Minimal climb ascent [m]", min_value=10, max_value=2000, value=30, step=10)
    min_avg_slope = st.number_input("Minimal average grade (%)", min_value=2.0, max_value=15.0, value=2.0, step=0.5)
    merge_gap_m = st.number_input("Maximum descent/platau length during an ascent [m]", min_value=0, max_value=1000, value=100, step=50)
    smooth_window = st.number_input("Profile smoothing window", min_value=1, max_value=20, value=5, step=2)

    st.subheader("Slope ranges for profile coloring")
    slope_thresholds_str = st.text_input("Thresholds (in %), separated by commas", value="2,4,6,8")
    try:
        slope_thresholds = tuple(float(x.strip()) for x in slope_thresholds_str.split(",") if x.strip() != "")
    except Exception:
        slope_thresholds = (2, 4, 6, 8)
        st.warning("Incorrect threshold format - using defaults: 2,4,6,8")

# =====================
# GPX Upload Handling
# =====================
if uploaded_file is None:
    st.info("Upload a GPX file to analyze its elevation profile and detect climbs.")
    st.stop()

# =====================
# Parse GPX to DataFrame
# =====================
parser = GPXParser(uploaded_file.read())
track_df = parser.parse_to_dataframe()

# Initialize ElevationProfile
main_profile = ElevationProfile(track_df, seg_unit_km=0.5, smooth_window=smooth_window)

total_distance_km = float(track_df["km"].max())
total_ascent_m = float(main_profile.total_ascent())
total_descent_m = float(main_profile.total_descent())
highest_point_m = float(main_profile.highest_point())
lowest_point_m = float(main_profile.lowest_point())

# =====================
# Main Elevation Profile Plot
# =====================
fig_main, ax = main_profile.plot_profile(
    show_labels=False,
    show_background=True,
    slope_thresholds=slope_thresholds
)
if total_distance_km < 10:
    ax.set_xticks(np.arange(0, total_distance_km, 1))
elif total_distance_km < 50:
    ax.set_xticks(np.arange(0, total_distance_km, 5))
else:
    ax.set_xticks(np.arange(0, total_distance_km, 10))

# =====================
# Detect Climbs
# =====================
climbs_df = main_profile.detect_climbs(
    min_length_m=min_length,
    min_gain_m=min_gain,
    min_avg_slope=min_avg_slope,
    merge_gap_m=merge_gap_m,
)

track_df_for_map = main_profile.route_data
base_map = build_base_map_with_detected_climbs(track_df_for_map, climbs_df)

if not climbs_df.empty:
    climbs_df.index = np.arange(1, len(climbs_df) + 1)
    climbs_df['start_km'] = climbs_df['start_km'].round(2)
    climbs_df['end_km'] = climbs_df['end_km'].round(2)
    climbs_df['start-end km'] = climbs_df["start_km"].astype(str).str.cat(climbs_df["end_km"].astype(str), sep=" - ") + " km"
    climbs_df['Difficulty score'] = climbs_df.apply(lambda row: classify_climb_difficulty(row.length_m, row.avg_grade_pct)[1], axis=1)
    climbs_df['Difficulty category'] = climbs_df.apply(lambda row: classify_climb_difficulty(row.length_m, row.avg_grade_pct)[0], axis=1)
    for number, row in climbs_df.iterrows():
        popup_text = (
            f"Podjazd {number}<br>"
            f"Długość: {row['length_m']} m<br>"
            f"Wznios: +{row['gain_m']} m<br>"
            f"Śr. nachylenie: {row['avg_grade_pct']}%"
        )
        folium.Marker(
            [row["start_lat"], row["start_lon"]],
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(color="orange", icon="arrow-up")
        ).add_to(base_map)
        folium.Marker(
            [row["end_lat"], row["end_lon"]],
            popup=folium.Popup(f"Koniec podjazdu: {number}", max_width=250),
            icon=folium.Icon(color="blue", icon="flag")
        ).add_to(base_map)

# =====================
# Stats & Map
# =====================
col1, col2 = st.columns([1, 5])
with col1:
    st.subheader("Mapa trasy i statystyki")
    st.metric("Długość trasy", f"{total_distance_km:.2f} km")
    st.metric("Całkowite przewyższenie", f"{total_ascent_m:.0f} m")
    st.metric("Całkowity zjazd", f"{total_descent_m:.0f} m")
    st.metric("Najwyższy punkt", f"{highest_point_m:.0f} m n.p.m.")
    st.metric("Najniższy punkt", f"{lowest_point_m:.0f} m n.p.m.")
with col2:
    map_html = base_map.get_root().render()
    components.html(map_html, height=550, width=2000)

# =====================
# Main Profile Display
# =====================
with st.expander("Główny profil wysokościowy", expanded=True):
    st.pyplot(fig_main, use_container_width=True)

# =====================
# Segment Analysis
# =====================
with st.expander("Wybrany fragment profilu trasy", expanded=False):
    col1, col2 = st.columns([1, 1])
    with col1:
        min_km = float(track_df["km"].min())
        max_km = float(track_df["km"].max())
        selected_range = st.slider(
            "Wybierz zakres odległości [km]:",
            min_value=min_km,
            max_value=max_km,
            value=(min_km, max_km),
            step=0.1
        )
    with col2:
        st.markdown(f"**Zakres:** {selected_range[0]:.2f} km - {selected_range[1]:.2f} km")

    if selected_range[0] < selected_range[1]:
        segment_df = track_df[(track_df["km"] >= selected_range[0]) & (track_df["km"] <= selected_range[1])]
        if len(segment_df) >= 3:
            segment_profile = ElevationProfile(segment_df, seg_unit_km=0.5, smooth_window=smooth_window)
            fig_s, ax_s = segment_profile.plot_profile(
                show_labels=False,
                show_background=True,
                slope_thresholds=slope_thresholds
            )
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)
        else:
            st.info("Zbyt mało punktów w tym zakresie.")

# =====================
# Slope Distribution Table
# =====================
slope_df = main_profile.compute_slope_lengths(slope_thresholds=slope_thresholds)
uphill_downhill = main_profile.compute_slope_lengths(slope_thresholds=(-2, 2))
uphill_downhill.loc[:, 'slope_range'] = ['Downhill (< -2%)', 'Flat', 'Uphill (> 2%)'] 

fig = px.pie(
    names=uphill_downhill['slope_range'],
    values=uphill_downhill['length_km'],
    width=300, height=300
)
fig.update_traces(textinfo='label+percent', textfont_size=15, showlegend=False)

with st.expander("Tabela długości wg przedziałów nachylenia", expanded=False):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(slope_df, use_container_width=True, hide_index=True)
    with col2:
        st.plotly_chart(fig, use_container_width=True)

# =====================
# Climbs Table & Profiles
# =====================
if climbs_df.empty:
    st.warning("Nie wykryto podjazdów dla podanych parametrów.")
else:
    with st.expander("Wykryte podjazdy", expanded=False):
        st.dataframe(climbs_df[['start-end km', 'length_m', 'gain_m', 'avg_grade_pct', 'Difficulty category']], use_container_width=True)

    st.subheader("Profile wysokości podjazdów")
    tab_titles = [f"Podjazd {i+1}" for i in range(len(climbs_df))]
    tabs = st.tabs(tab_titles)

    for i, row in enumerate(climbs_df.itertuples()):
        with tabs[i]:
            climb_df = track_df.iloc[
                track_df['km'].sub(row.start_km).abs().idxmin():
                track_df['km'].sub(row.end_km).abs().idxmin() + 1
            ].reset_index(drop=True)
            if len(climb_df) >= 3:
                climb_df["km"] -= climb_df["km"].iloc[0]
                climb_profile = ElevationProfile(climb_df, seg_unit_km=0.1, smooth_window=smooth_window)
                fig_c, ax_c = climb_profile.plot_profile(
                    show_labels=False,
                    show_background=False,
                    slope_thresholds=slope_thresholds
                )
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Długość", f"{row.length_m} m")
                    st.metric("Wznios", f"{row.gain_m} m")
                    st.metric("Śr. nachylenie", f"{row.avg_grade_pct} %")
                    st.metric("Kategoria", classify_climb_difficulty(row.length_m, row.avg_grade_pct)[0])
                with col2:
                    st.pyplot(fig_c, use_container_width=True)
                plt.close(fig_c)
            else:
                st.info("Zbyt mało punktów do sensownego wykresu.")


# =====================
# Stopka
# =====================
st.markdown("---")
st.caption("© Aplikacja demonstracyjna – analiza GPX w Streamlit. Ustawienia wykrywania podjazdów: długość i wznios w panelu bocznym.")
