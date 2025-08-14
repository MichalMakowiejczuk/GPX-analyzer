import folium

def build_base_map_with_detected_climbs(track_df, climbs_df, default_color="blue", climb_color="orange"):
    """
    track_df: DataFrame z pełną trasą
    climbs_df: DataFrame z wykrytymi podjazdami (z kolumnami 'start_idx' i 'end_idx')
    """
    start_coords = (track_df["latitude"].iloc[0], track_df["longitude"].iloc[0])
    m = folium.Map(location=start_coords, zoom_start=13, control_scale=True)

    # Tworzymy listę wszystkich segmentów z kolorami
    climb_segments = []
    if not climbs_df.empty:
        for _, row in climbs_df.iterrows():
            climb_segments.append((row['start_idx'], row['end_idx']))

    for i in range(len(track_df) - 1):
        # Sprawdzamy, czy punkt należy do podjazdu
        color = default_color
        for start, end in climb_segments:
            if start <= i < end:
                color = climb_color
                break

        coords = [
            (track_df["latitude"].iloc[i], track_df["longitude"].iloc[i]),
            (track_df["latitude"].iloc[i + 1], track_df["longitude"].iloc[i + 1])
        ]
        folium.PolyLine(coords, color=color, weight=5, opacity=1).add_to(m)

    # Start i meta
    folium.Marker((track_df["latitude"].iloc[0], track_df["longitude"].iloc[0]),
                  popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((track_df["latitude"].iloc[-1], track_df["longitude"].iloc[-1]),
                  popup="Meta", icon=folium.Icon(color="red")).add_to(m)

    return m
