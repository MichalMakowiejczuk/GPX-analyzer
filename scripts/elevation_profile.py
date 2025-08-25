from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim

# ========================
# Utilities
# ========================

def _validate_track_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and non-empty DataFrame; return a copy."""
    required = {"km", "elevation", "latitude", "longitude"}
    if df is None or df.empty:
        raise ValueError("The input DataFrame is empty.")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"The DataFrame must contain the following columns: {required}; missing: {missing}")
    return df.copy()


def _get_slope_bins(slope_thresholds: Tuple[float, ...]) -> Tuple[List[float], List[str]]:
    """Generate slope bin edges and human-friendly labels.

    Returns
    -------
    thresholds : list[float]
        Bin edges with -inf and +inf added.
    labels : list[str]
        Labels for each bin, e.g. ["< 2%", "2 ~ 4%", ..., ">= 8%"].
    """
    thresholds = [-np.inf] + list(slope_thresholds) + [np.inf]
    labels: List[str] = []
    for low, high in zip(thresholds[:-1], thresholds[1:]):
        if low == -np.inf:
            labels.append(f"< {high}%")
        elif high == np.inf:
            labels.append(f">= {low}%")
        else:
            labels.append(f"{low} ~ {high}%")
    return thresholds, labels


# ========================
# 1) TrackDataProcessor
# ========================

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


# ========================
# 2) SlopeAnalyzer
# ========================

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


# ========================
# 3) ClimbDetector
# ========================

class ClimbDetector:
    """Detect climbs based on length, gain, and slope criteria."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def detect(
        self,
        min_length_m: float = 500,
        min_gain_m: float = 30,
        min_avg_slope: float = 2.0,
        window_m: float = 200,
        merge_gap_m: float = 100,
    ) -> pd.DataFrame:
        df = self.df.dropna(subset=["km", "elev_smooth"]).copy()
        if df.empty:
            return pd.DataFrame([])

        dist_m = df["km"] * 1000.0
        elev = df["elev_smooth"]

        # średni odstęp między punktami
        avg_spacing = float(np.mean(np.diff(dist_m))) if len(dist_m) > 1 else window_m
        window_pts = max(2, int(window_m / max(avg_spacing, 1e-9)))

        # rolling slope [%] = rolling delta_elev / rolling delta_dist
        df["rolling_gain"] = elev.diff().rolling(window_pts, min_periods=1).sum()
        df["rolling_dist"] = dist_m.diff().rolling(window_pts, min_periods=1).sum()
        df["rolling_slope"] = (df["rolling_gain"] / df["rolling_dist"]) * 100
        df["rolling_slope"] = df["rolling_slope"].fillna(0)

        # regiony gdzie slope > połowy progu
        is_uphill = df["rolling_slope"] > (min_avg_slope / 2.0)
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

        # scalanie regionów blisko siebie
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
        slopes = df["datapoint_slope"].values if "datapoint_slope" in df else np.zeros(len(df))
        for s, e in merged:
            start_row = df.iloc[s]
            end_row = df.iloc[e]
            length_m = (end_row["km"] - start_row["km"]) * 1000.0
            gain_m = end_row["elev_smooth"] - start_row["elev_smooth"]

            if length_m < min_length_m or gain_m < min_gain_m:
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

# ========================
# 4) PlaceGeolocator (with caching)
# ========================

class PlaceGeolocator:
    """Reverse-geocode notable places along the route and de-duplicate by distance."""

    def __init__(self, cache_file: str = "places_cache.json", user_agent: str = "ElevationProfileApp") -> None:
        self.cache_file = cache_file
        self.cache: Dict[str, Optional[str]] = self._load_cache(cache_file)
        self.geolocator = Nominatim(user_agent=user_agent)

    @staticmethod
    def _load_cache(cache_file: str) -> Dict[str, Optional[str]]:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except IOError as e:
            raise IOError(f"Failed to save cache to {self.cache_file}") from e

    def geolocate(
        self,
        df: pd.DataFrame,
        min_distance_km: float = 5.0,
        rate_limit_sec: int = 1,
    ) -> pd.DataFrame:
        places: List[List[Any]] = []
        place_last_km: Dict[str, float] = {}
        place_group = 0

        for segment in df["segment"].unique():
            segment_df = df[df["segment"] == segment]
            row = segment_df.loc[segment_df["km"].idxmin()]
            lat, lon, elev, km = row["latitude"], row["longitude"], row["elevation"], row["km"]
            coords_key = f"{lat:.5f},{lon:.5f}"

            place_name = self.cache.get(coords_key)
            if place_name is None:
                try:
                    location = self.geolocator.reverse(f"{lat},{lon}", exactly_one=True, timeout=10)
                    address = location.raw.get("address", {}) if location else {}
                    place_name = address.get("city") or address.get("town") or address.get("village")
                    self.cache[coords_key] = place_name
                    time.sleep(rate_limit_sec)
                except (GeocoderTimedOut, GeocoderUnavailable):
                    continue
                except Exception:
                    continue

            if place_name and ((place_name not in place_last_km) or (km - place_last_km[place_name] >= min_distance_km)):
                place_group += 1
                place_last_km[place_name] = float(km)
                places.append([int(segment), str(place_name), float(elev), float(km), int(place_group)])

        self._save_cache()
        places_df = pd.DataFrame(places, columns=["segment", "place", "elevation", "km", "group"]).drop_duplicates(
            subset=["place"]
        )
        return places_df.sort_values(["group", "km"]).reset_index(drop=True)


# ========================
# 5) ProfilePlotter (Matplotlib)
# ========================

class ProfilePlotter:
    """Plot elevation profile with slope-based coloring and optional labels."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def plot_profile(
        self,
        places_df: Optional[pd.DataFrame] = None,
        show_labels: bool = True,
        show_background: bool = True,
        background_color: str = "gray",
        background_shift_km: float = 0.5,
        background_shift_elev: float = 15.0,
        slope_thresholds: Tuple[float, ...] = (2, 4, 6, 8),
        slope_colors: Tuple[str, ...] = ("lightgreen", "yellow", "orange", "orangered", "maroon"),
        slope_labels: Optional[List[str]] = None,
        slope_type: str = "segment",
    ) -> Tuple[plt.Figure, plt.Axes]:
        thresholds, default_labels = _get_slope_bins(slope_thresholds)
        if slope_labels is None:
            slope_labels = default_labels
        if len(slope_colors) != len(slope_labels):
            raise ValueError("slope_colors length must match slope_labels length.")
        if slope_type not in {"segment", "datapoint"}:
            raise ValueError("slope_type must be 'segment' or 'datapoint'.")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_xlabel("Kilometers")
        ax.set_ylabel("Elevation [m]")
        ax.spines[["right", "top"]].set_visible(False)

        if show_background:
            ax.fill_between(
                self.df["km"] + background_shift_km,
                self.df["elev_smooth"] + background_shift_elev,
                color=background_color,
                zorder=0,
            )

        legend: List[mpatches.Patch] = []
        for i, color in enumerate(slope_colors):
            if slope_type == "segment":
                mask = (self.df["segment_slope"] >= thresholds[i]) & (self.df["segment_slope"] < thresholds[i + 1])
            else:
                mask = (self.df["datapoint_slope"] >= thresholds[i]) & (self.df["datapoint_slope"] < thresholds[i + 1])
            ax.fill_between(
                self.df["km"],
                self.df["elev_smooth"],
                where=mask,
                color=color,
                zorder=1,
                interpolate=True,
            )
            legend.append(mpatches.Patch(color=color, label=slope_labels[i]))

        if show_labels and places_df is not None and not places_df.empty:
            annotations_anchor = float(self.df["elevation"].max()) * 1.1
            last_label_km = -5.0
            for _, row in places_df.iterrows():
                if row["km"] - last_label_km >= 5.0:
                    ax.annotate(
                        str(row["place"]),
                        xy=(float(row["km"]), float(row["elevation"])),
                        xytext=(float(row["km"]), annotations_anchor),
                        arrowprops=dict(arrowstyle="-", color="lightgray"),
                        horizontalalignment="center",
                        rotation=90,
                        size=10,
                        color="gray",
                    )
                    last_label_km = float(row["km"])

        ax.plot(self.df["km"], self.df["elev_smooth"], color="darkgrey", linewidth=0.15)
        y_lower_bound = math.floor(float(self.df["elevation"].min()) * 0.8 / 100.0) * 100.0
        ax.set_ylim(y_lower_bound, float(self.df["elevation"].max()) * 1.1)
        ax.set_xlim(float(self.df["km"].min()), float(self.df["km"].max()))
        ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        return fig, ax


# ========================
# 6) Facade: ElevationProfile
# ========================

class ElevationProfile:
    """Facade providing a clean, high-level API while delegating to components.

    Parameters
    ----------
    track_df : pd.DataFrame
        Must contain columns: km, elevation, latitude, longitude
    seg_unit_km : float, default 0.5
    smooth_window : int, default 5
    """

    def __init__(self, track_df: pd.DataFrame, seg_unit_km: float = 0.5, smooth_window: int = 5) -> None:
        processor = TrackDataProcessor(track_df, seg_unit_km=seg_unit_km, smooth_window=smooth_window)
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
        if self._geolocator is None or (self._geolocator and self._geolocator.cache_file != cache_file):
            self._geolocator = PlaceGeolocator(cache_file=cache_file, user_agent=user_agent)
        self.places_df = self._geolocator.geolocate(self.data, min_distance_km=min_distance_km, rate_limit_sec=rate_limit_sec)
        return self.places_df

    # ---------- Stats ----------
    def compute_slope_lengths(
        self, slope_thresholds: Tuple[float, ...] = (2, 4, 6, 8), min_delta_km: float = 1e-4
    ) -> pd.DataFrame:
        return self.slope.compute_slope_lengths(slope_thresholds=slope_thresholds, min_delta_km=min_delta_km)

    def detect_climbs(
        self,
        min_length_m: float = 500,
        min_gain_m: float = 30,
        min_avg_slope: float = 2.0,
        window_m: float = 200,
        merge_gap_m: float = 100,
    ) -> pd.DataFrame:
        return self.climb_detector.detect(
            min_length_m=min_length_m,
            min_gain_m=min_gain_m,
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
        slope_colors: Tuple[str, ...] = ("lightgreen", "yellow", "orange", "orangered", "maroon"),
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

    def summary(self) -> dict[str, float | int]:
            """Return a dictionary with key route statistics and GPX quality info."""
            total_distance_km = float(self.data["km"].max() - self.data["km"].min())
            climbs_df = self.detect_climbs()

            # punkty i jakość GPX
            num_points = len(self.data)
            delta_m = self.data["km"].diff().dropna() * 1000.0
            avg_spacing = float(delta_m.mean()) if not delta_m.empty else 0.0
            median_spacing = float(delta_m.median()) if not delta_m.empty else 0.0

            return {
                "distance_km": round(total_distance_km, 2),
                "total_ascent_m": round(self.get_total_ascent(), 1),
                "total_descent_m": round(self.get_total_descent(), 1),
                "highest_point_m": round(self.get_highest_point(), 1),
                "lowest_point_m": round(self.get_lowest_point(), 1),
                "num_climbs": int(len(climbs_df)),
                "num_points": num_points,
                "avg_spacing_m": round(avg_spacing, 2),
                "median_spacing_m": round(median_spacing, 2),
            }

if __name__ == "__main__":
    from gpx_parser import GPXParser

    gpx_file = "sample_data/cycling_track.gpx"
    parser = GPXParser(gpx_file)
    track_df = parser.parse_to_dataframe()

    profile = ElevationProfile(track_df, seg_unit_km=0.5, smooth_window=5)
    climbs = profile.detect_climbs()
    bins = profile.compute_slope_lengths()
    fig, ax = profile.plot_profile()
    plt.show()
