import streamlit as st
import streamlit.components.v1 as components

def render_main_stats_and_map(stats, base_map):
    col1, col2 = st.columns([1, 5])
    with col1:
        st.subheader("Map and stats")
        st.metric("Distance", f"{stats['distance_km']:.2f} km")
        st.metric("Ascent", f"{stats['ascent_m']:.0f} m")
        st.metric("Descent", f"{stats['descent_m']:.0f} m")
        st.metric("Highest point", f"{stats['highest_point']:.0f} m n.p.m.")
        st.metric("Lowest point", f"{stats['lowest_point']:.0f} m n.p.m.")
    with col2:
        map_html = base_map.get_root().render()
        components.html(map_html, height=550, width=2000)