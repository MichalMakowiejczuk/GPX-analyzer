from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .climb_detector import ClimbDetector
from .gpx_quality_analyzer import GpxQualityAnalyzer
from .place_geolocator import PlaceGeolocator
from .profile_plotter import ProfilePlotter
from .slope_analyzer import SlopeAnalyzer
from .track_processor import TrackDataProcessor


class ElevationProfile:
    """Facade providing a clean, high-level API while delegating to components.

    Parameters
    ----------
    track_df : pd.DataFrame
        Must contain columns: km, elevation, latitude, longitude
    seg_unit_km : float, default 0.5
    smooth_window : int, default 5
    """

    def __init__(
        self, track_df: pd.DataFrame, seg_unit_km: float = 0.5, smooth_window: int = 5
    ) -> None:
        processor = TrackDataProcessor(
            track_df, seg_unit_km=seg_unit_km, smooth_window=smooth_window
        )
        self.data: pd.DataFrame = processor.track_df

        # Analysis components
        self.slope = SlopeAnalyzer(self.data)
        # keep a unified working dataframe from slope analyzer (has computed columns)
        self.data = self.slope.df
        self.climb_detector = ClimbDetector(self.data)
        self.plotter = ProfilePlotter(self.data)

        # Optional, filled on demand
        self.places_df: Optional[pd.DataFrame] = None
        self._geolocator: Optional[PlaceGeolocator] = None

    # ---------- Geolocation ----------
    def geolocate_places(
        self,
        min_distance_km: float = 5.0,
        cache_file: str = "places_cache.json",
        rate_limit_sec: int = 1,
        user_agent: str = "ElevationProfileApp",
    ) -> pd.DataFrame:
        if self._geolocator is None or (
            self._geolocator and self._geolocator.cache_file != cache_file
        ):
            self._geolocator = PlaceGeolocator(
                cache_file=cache_file, user_agent=user_agent
            )
        self.places_df = self._geolocator.geolocate(
            self.data, min_distance_km=min_distance_km, rate_limit_sec=rate_limit_sec
        )
        return self.places_df

    # ---------- Stats ----------
    def compute_slope_lengths(
        self,
        slope_thresholds: Tuple[float, ...] = (2, 4, 6, 8),
        min_delta_km: float = 1e-4,
    ) -> pd.DataFrame:
        return self.slope.compute_slope_lengths(
            slope_thresholds=slope_thresholds, min_delta_km=min_delta_km
        )

    def detect_climbs(
        self,
        min_length_m: float = 500,
        min_avg_slope: float = 2.0,
        window_m: float = 200,
        merge_gap_m: float = 100,
    ) -> pd.DataFrame:
        return self.climb_detector.detect(
            min_length_m=min_length_m,
            min_avg_slope=min_avg_slope,
            window_m=window_m,
            merge_gap_m=merge_gap_m,
        )

    def get_total_ascent(self) -> float:
        return self.slope.get_total_ascent()

    def get_total_descent(self) -> float:
        return self.slope.get_total_descent()

    def get_highest_point(self) -> float:
        return self.slope.get_highest_point()

    def get_lowest_point(self) -> float:
        return self.slope.get_lowest_point()

    # ---------- Plotting ----------
    def plot_profile(
        self,
        show_labels: bool = True,
        show_background: bool = True,
        background_color: str = "gray",
        background_shift_km: float = 0.5,
        background_shift_elev: float = 15.0,
        slope_thresholds: Tuple[float, ...] = (2, 4, 6, 8),
        slope_colors: Tuple[str, ...] = (
            "lightgreen",
            "yellow",
            "orange",
            "orangered",
            "maroon",
        ),
        slope_labels: Optional[List[str]] = None,
        slope_type: str = "segment",
    ) -> Tuple[plt.Figure, plt.Axes]:
        return self.plotter.plot_profile(
            places_df=self.places_df,
            show_labels=show_labels,
            show_background=show_background,
            background_color=background_color,
            background_shift_km=background_shift_km,
            background_shift_elev=background_shift_elev,
            slope_thresholds=slope_thresholds,
            slope_colors=slope_colors,
            slope_labels=slope_labels,
            slope_type=slope_type,
        )

    # ---------- Data access ----------
    def get_route_data(self) -> pd.DataFrame:
        return self.data.copy()

    def summary(self) -> dict[str, float | int | str]:
        total_distance_km = float(self.data["km"].max() - self.data["km"].min())
        gpx_quality = GpxQualityAnalyzer(self.data).analyze()

        stats = {
            "distance_km": round(total_distance_km, 2),
            "total_ascent_m": round(self.slope.get_total_ascent(), 1),
            "total_descent_m": round(self.slope.get_total_descent(), 1),
            "highest_point_m": round(self.slope.get_highest_point(), 1),
            "lowest_point_m": round(self.slope.get_lowest_point(), 1),
            "AVG slope (%)": (
                round(
                    (self.slope.get_total_ascent() - self.slope.get_total_descent())
                    / max(total_distance_km * 10, 1e-9),
                    2,
                )
                if total_distance_km > 0
                else 0.0
            ),  # uwzględnia zjazdy
        }
        return {**stats, **gpx_quality}
