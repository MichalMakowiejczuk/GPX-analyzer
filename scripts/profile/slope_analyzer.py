import pandas as pd
from typing import Tuple
from .utils import _get_slope_bins

class SlopeAnalyzer:
    """Compute slopes and slope-based statistics."""

    def __init__(self, track_df: pd.DataFrame) -> None:
        self.df = track_df.copy()
        self._compute_slopes()

    def _compute_slopes(self) -> None:
        df = self.df
        df["delta_km"] = df["km"].diff()
        df["delta_elev"] = df["elev_smooth"].diff()
        df["delta_dist_m"] = df["delta_km"] * 1000.0
        df["slope_pct"] = (df["delta_elev"] / df["delta_dist_m"]) * 100.0
        # slope na poziomie punktu (tak jak miałeś)
        df["datapoint_slope"] = df["slope_pct"]
        # slope uśredniony na segment
        df["segment_slope"] = df.groupby("segment")["slope_pct"].transform("mean")
        self.df = df

    # ----- Aggregate stats -----
    def get_total_ascent(self) -> float:
        return float(self.df["delta_elev"].clip(lower=0).sum())

    def get_total_descent(self) -> float:
        return float(-self.df["delta_elev"].clip(upper=0).sum())

    def get_highest_point(self) -> float:
        return float(self.df["elev_smooth"].max())

    def get_lowest_point(self) -> float:
        return float(self.df["elev_smooth"].min())

    def compute_slope_lengths(
        self,
        slope_thresholds: Tuple[float, ...] = (2, 4, 6, 8),
        min_delta_km: float = 1e-4,
    ) -> pd.DataFrame:
        df = self.df.dropna(subset=["delta_km", "delta_elev"]).copy()
        df = df[df["delta_km"] > min_delta_km]

        thresholds, labels = _get_slope_bins(slope_thresholds)
        df["slope_range"] = pd.cut(df["slope_pct"], bins=thresholds, labels=labels, right=True)

        result = (
            df.groupby("slope_range")["delta_km"]
            .sum()
            .reset_index()
            .rename(columns={"delta_km": "length_km"})
        )
        total = result["length_km"].sum()
        result["length_km"] = result["length_km"].round(2)
        result["% of total"] = (result["length_km"] / total * 100).round(2) if total else 0
        return result