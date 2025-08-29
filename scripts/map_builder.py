import folium
import pandas as pd


def build_base_map_with_detected_climbs(
    track_df: pd.DataFrame,
    climbs_df: pd.DataFrame,
    default_color: str = "blue",
    climb_color: str = "orange",
) -> folium.Map:
    """
    Build an interactive map with a route and highlighted climb segments.

    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame containing the full route with at least the following columns:
        - 'latitude'
        - 'longitude'
    climbs_df : pd.DataFrame
        DataFrame containing detected climbs with the following columns:
        - 'start_idx': starting index of the climb in track_df
        - 'end_idx': ending index of the climb in track_df
    default_color : str, optional
        Color for non-climb segments (default is "blue").
    climb_color : str, optional
        Color for climb segments (default is "orange").

    Returns
    -------
    folium.Map
        A Folium map object displaying the route and climbs.
    """
    if track_df.empty:
        raise ValueError("track_df is empty. Cannot build map.")

    start_coords = (track_df["latitude"].iloc[0], track_df["longitude"].iloc[0])
    m = folium.Map(location=start_coords, zoom_start=13, control_scale=True)

    # Extract climb index ranges
    climb_segments = []
    if not climbs_df.empty:
        for _, row in climbs_df.iterrows():
            climb_segments.append((row["start_idx"], row["end_idx"]))

    # Add route segments to the map
    for i in range(len(track_df) - 1):
        # Check if current segment is part of a climb
        color = default_color
        for start, end in climb_segments:
            if start <= i < end:
                color = climb_color
                break

        coords = [
            (track_df["latitude"].iloc[i], track_df["longitude"].iloc[i]),
            (track_df["latitude"].iloc[i + 1], track_df["longitude"].iloc[i + 1]),
        ]
        folium.PolyLine(coords, color=color, weight=5, opacity=0.8).add_to(m)

    # Add start and finish markers
    # if start and finish is very close, add only one marker
    if (
        abs(track_df["latitude"].iloc[0] - track_df["latitude"].iloc[-1]) < 1e-5
        or abs(track_df["longitude"].iloc[0] - track_df["longitude"].iloc[-1]) < 1e-5
    ):  # ~1 meter
        folium.Marker(
            (track_df["latitude"].iloc[-1], track_df["longitude"].iloc[-1]),
            popup="Start/Finish",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)
    else:
        folium.Marker(
            (track_df["latitude"].iloc[0], track_df["longitude"].iloc[0]),
            popup="Start",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)

        folium.Marker(
            (track_df["latitude"].iloc[-1], track_df["longitude"].iloc[-1]),
            popup="Finish",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(m)

    return m
