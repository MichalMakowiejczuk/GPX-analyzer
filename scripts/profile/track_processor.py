import numpy as np
import pandas as pd
from dataclasses import dataclass
from .utils import _validate_track_df

@dataclass
class TrackDataProcessor:
    """Prepare and enrich raw track data.

    - Validates input DataFrame
    - Smooths elevation (rolling mean)
    - Assigns distance-based segments
    """

    track_df: pd.DataFrame
    seg_unit_km: float = 0.5
    smooth_window: int = 5
    min_distance_m: float = 1.0

    def __post_init__(self) -> None:
        df = _validate_track_df(self.track_df)
        self.track_df = self._smooth_elevation_profile(df, self.smooth_window, self.min_distance_m)
        self.track_df = self._assign_segments_by_distance(self.track_df, self.seg_unit_km)

    @staticmethod
    def _smooth_elevation_profile(df: pd.DataFrame, mean_window: int = 5, min_distance_m: float = 1.0) -> pd.DataFrame:
        """Smooth elevation using a centered rolling mean and drop tightly-stacked points.

        Parameters
        ----------
        mean_window : int
            Window size for mean filter.
        min_distance_m : float
            Minimum distance between consecutive points; shorter gaps are dropped.
        """
        out = df.copy()
        min_km = min_distance_m / 1000.0
        out["km_diff"] = out["km"].diff()
        out = out[out["km_diff"].fillna(min_km) >= min_km].reset_index(drop=True)
        out["elev_smooth"] = (
            out["elevation"].rolling(window=mean_window, center=True, min_periods=1).mean()
        )
        return out

    @staticmethod
    def _assign_segments_by_distance(df: pd.DataFrame, seg_unit_km: float) -> pd.DataFrame:
        out = df.copy()
        out["segment"] = np.floor(out["km"] / seg_unit_km).astype(int)
        return out
