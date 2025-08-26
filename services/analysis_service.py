def compute_route_stats(profile):
    route_df = profile.get_route_data()
    return {
        "distance_km": float(route_df["km"].max()),
        "ascent_m": float(profile.get_total_ascent()),
        "descent_m": float(profile.get_total_descent()),
        "highest_point": float(profile.get_highest_point()),
        "lowest_point": float(profile.get_lowest_point())
    }

def detect_climbs(profile, min_length, min_gain, min_avg_slope, merge_gap_m):
    return profile.detect_climbs(
        min_length_m=min_length,
        min_gain_m=min_gain,
        min_avg_slope=min_avg_slope,
        merge_gap_m=merge_gap_m
    )