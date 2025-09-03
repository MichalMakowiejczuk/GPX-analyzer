import numpy as np
import pandas as pd

from scripts.climb_classification import classify_climb_difficulty
from services.analysis_service import detect_climbs


def prepare_climbs(
    profile, min_length: int, min_avg_slope: float, merge_gap_m: int
) -> pd.DataFrame:
    climbs_df = detect_climbs(profile, min_length, min_avg_slope, merge_gap_m)

    if climbs_df.empty:
        return climbs_df

    # Reindex and format
    climbs_df.index = np.arange(1, len(climbs_df) + 1)
    climbs_df[["start_km", "end_km"]] = climbs_df[["start_km", "end_km"]].round(2)

    climbs_df["start-end km"] = (
        climbs_df["start_km"]
        .astype(str)
        .str.cat(climbs_df["end_km"].astype(str), sep=" - ")
        + " km"
    )

    # Difficulty classification
    climbs_df[["Difficulty category", "Difficulty score"]] = climbs_df.apply(
        lambda row: classify_climb_difficulty(row.length_m, row.avg_grade_pct),
        axis=1,
        result_type="expand",
    )

    return climbs_df
