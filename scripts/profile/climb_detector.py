from typing import Any, Optional

import numpy as np
import pandas as pd


class ClimbDetector:
    """Detect climbs in elevation profile data.

    Parameters:
    - min_length_m: Minimum length of a climb in meters.
    - min_avg_slope: Minimum average slope (%) of a climb.
    - window_m: Smoothing window size in meters for rolling slope calculation.
    - merge_gap_m: Maximum gap in meters to merge close climb segments.
    - base_detection_slope: Base slope (%) threshold to consider as uphill.

    Returns:
    - DataFrame with detected climbs and their statistics."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def detect(
        self,
        min_length_m: float = 500,
        min_avg_slope: float = 2.0,
        window_m: float = 200,
        merge_gap_m: float = 100,
        base_detection_slope: float = 2.0,  # minimal slope to consider as uphill
    ) -> pd.DataFrame:
        df = self.df.dropna(subset=["km", "elev_smooth"]).copy()
        if df.empty:
            return pd.DataFrame([])

        dist_m = df["km"] * 1000.0
        elev = df["elev_smooth"]

        # rolling slope
        avg_spacing = float(np.mean(np.diff(dist_m))) if len(dist_m) > 1 else window_m
        window_pts = max(2, int(window_m / max(avg_spacing, 1e-9)))

        df["rolling_gain"] = elev.diff().rolling(window_pts, min_periods=1).sum()
        df["rolling_dist"] = dist_m.diff().rolling(window_pts, min_periods=1).sum()
        df["rolling_slope"] = (df["rolling_gain"] / df["rolling_dist"]) * 100
        df["rolling_slope"] = df["rolling_slope"].fillna(0)

        is_uphill = df["rolling_slope"] > base_detection_slope  # / 2.0
        regions: list[tuple[int, int]] = []
        start_idx: Optional[int] = None
        for i, up in enumerate(is_uphill):
            if up and start_idx is None:
                start_idx = i
            elif not up and start_idx is not None:
                if i - start_idx > 1:
                    regions.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            regions.append((start_idx, len(df) - 1))

        # merge close segments
        merged: list[tuple[int, int]] = []
        for seg in regions:
            if not merged:
                merged.append(seg)
            else:
                ps, pe = merged[-1]
                if (dist_m.iloc[seg[0]] - dist_m.iloc[pe]) <= merge_gap_m:
                    merged[-1] = (ps, seg[1])
                else:
                    merged.append(seg)

        climbs: list[dict[str, Any]] = []
        slopes = (
            df["datapoint_slope"].values
            if "datapoint_slope" in df
            else np.zeros(len(df))
        )
        for s, e in merged:
            start_row = df.iloc[s]
            end_row = df.iloc[e]
            length_m = (end_row["km"] - start_row["km"]) * 1000.0
            segment_elev = elev.iloc[s : e + 1].values
            segment_gain = np.diff(segment_elev)
            gain_m = float(np.sum(segment_gain[segment_gain > 0]))

            if length_m < min_length_m:
                continue

            avg_slope = (gain_m / max(length_m, 1e-9)) * 100
            if avg_slope < min_avg_slope:
                continue

            climbs.append(
                {
                    "start_idx": s,
                    "end_idx": e,
                    "start_km": float(start_row["km"]),
                    "end_km": float(end_row["km"]),
                    "length_m": round(length_m, 1),
                    "gain_m": round(gain_m, 1),
                    "avg_grade_pct": round(avg_slope, 1),
                    "max_grade_pct": round(float(np.max(slopes[s : e + 1])), 1),
                    "start_lat": float(start_row["latitude"]),
                    "start_lon": float(start_row["longitude"]),
                    "end_lat": float(end_row["latitude"]),
                    "end_lon": float(end_row["longitude"]),
                }
            )

        return pd.DataFrame(climbs)
