import folium
import pandas as pd
import streamlit as st

from scripts.map_builder import build_base_map_with_detected_climbs


@st.cache_data
def generate_map(track_df: pd.DataFrame, climbs_df: pd.DataFrame) -> folium.Map:
    base_map = build_base_map_with_detected_climbs(track_df, climbs_df)
    for number, row in climbs_df.iterrows():
        popup_text = (
            f"Climb {number}<br>"
            f"Length: {row['length_m']} m<br>"
            f"Ascent: +{row['gain_m']} m<br>"
            f"AVG. slope: {row['avg_grade_pct']}%"
        )
        folium.Marker(
            [row["start_lat"], row["start_lon"]],
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(color="orange", icon="arrow-up"),
        ).add_to(base_map)
        folium.Marker(
            [row["end_lat"], row["end_lon"]],
            popup=folium.Popup(f"END<br> climb: {number}", max_width=250),
            icon=folium.Icon(color="blue", icon="flag"),
        ).add_to(base_map)
    return base_map
