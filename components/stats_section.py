import streamlit as st
import streamlit.components.v1 as components


def render_main_stats_and_map2(stats, base_map) -> None:
    col1, col2 = st.columns([1, 5])
    with col1:
        st.subheader("Map and stats")
        st.metric("Distance", f"{stats['distance_km']:.2f} km")
        st.metric("Total elevation gain", f"{stats['ascent_m']:.0f} m")
        st.metric("Total elevation loss", f"{stats['descent_m']:.0f} m")
        st.metric("Highest point", f"{stats['highest_point']:.0f} m n.p.m.")
        st.metric("Lowest point", f"{stats['lowest_point']:.0f} m n.p.m.")
    with col2:
        map_html = base_map.get_root().render()
        components.html(map_html, height=550, width=2000)


def render_main_stats_and_map(stats, base_map) -> None:
    with st.expander("Show map and main stats", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Distance", f"{stats['distance_km']:.2f} km", border=True)
        with col2:
            st.metric("Total elevation gain", f"{stats['ascent_m']:.0f} m", border=True)
        with col3:
            st.metric(
                "Total elevation loss", f"{stats['descent_m']:.0f} m", border=True
            )
        with col4:
            st.metric(
                "Highest point", f"{stats['highest_point']:.0f} m n.p.m.", border=True
            )
        with col5:
            st.metric(
                "Lowest point", f"{stats['lowest_point']:.0f} m n.p.m.", border=True
            )

        map_html = base_map.get_root().render()
        components.html(map_html, height=550, width=2000)
