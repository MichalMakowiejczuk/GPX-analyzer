import json
import os
import time
import math
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from scipy.signal import savgol_filter

class ElevationProfile:
    """
    Analyze, segment, and visualize the elevation profile of a route.

    The `ElevationProfile` class provides tools for processing GPS-based route data,
    calculating slopes, segmenting the route, detecting climbs, and generating
    elevation profile visualizations with slope-based coloring and annotations.

    It operates on a track represented as a Pandas DataFrame containing route
    distance, elevation, and geographic coordinates. The class computes smoothed
    elevation data and precomputes slopes for both individual points and aggregated
    segments for further analysis.

    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame containing the route track with the following required columns:
        - `km` (float): Cumulative distance in kilometers.
        - `elevation` (float): Elevation in meters.
        - `latitude` (float): Latitude in decimal degrees.
        - `longitude` (float): Longitude in decimal degrees.
    seg_unit_km : float, optional, default=0.5
        Segment length in kilometers used for grouping points and computing
        segment-based slopes.
    smooth_window : int, optional, default=5
        Window size (number of points) for elevation smoothing.

    Attributes
    ----------
    track_df : pd.DataFrame
        Copy of the input DataFrame, enriched with additional computed columns:
        - `elev_smooth` (float): Smoothed elevation profile.
        - `segment` (int): Segment index for each track point.
        - `datapoint_slope` (float): Instantaneous slope between consecutive points.
        - `segment_slope` (float): Average slope per segment.
    seg_unit_km : float
        Segment unit length in kilometers.
    smooth_window : int
        Window size for smoothing elevation.
    places_df : Optional[pd.DataFrame]
        DataFrame of notable places along the route for annotation (optional).

    Raises
    ------
    ValueError
        If `track_df` is empty or missing required columns.

    Notes
    -----
    - Elevation smoothing reduces noise in raw elevation data, improving slope
      accuracy.
    - The class is intended for cycling, hiking, and route planning analysis where
      elevation and slope characteristics matter.
    - Visualization relies on Matplotlib.

    Examples
    --------
    >>> track = pd.read_csv("route.csv")  # must contain km, elevation, latitude, longitude
    >>> profile = ElevationProfile(track, seg_unit_km=1.0, smooth_window=5)
    >>> profile.compute_slope_lengths()
    >>> profile.detect_climbs(min_length_m=800, min_gain_m=50)
    >>> fig, ax = profile.plot_profile(show_labels=True)
    >>> plt.show()
    """
    def __init__(self, track_df: pd.DataFrame, seg_unit_km: float = 0.5, smooth_window: int = 5) -> None:
        if track_df.empty:
            raise ValueError("The input DataFrame is empty.")
        required_cols = {"km", "elevation", "latitude", "longitude"}
        if not required_cols.issubset(track_df.columns):
            raise ValueError(f"The DataFrame must contain the following columns: {required_cols}")

        self.track_df: pd.DataFrame = track_df.copy()
        self.seg_unit_km: float = seg_unit_km
        self.smooth_window: int = smooth_window
        self.places_df: Optional[pd.DataFrame] = None

        # Process data
        self._smooth_elevation_profile(self.smooth_window)
        self._assign_segments_by_distance()
        self._compute_datapoint_slopes()
        self._compute_segment_slopes()

    # ========================
    # Private methods
    # ========================

    def _assign_segments_by_distance(self) -> None:
        """Assign segment numbers based on distance intervals."""
        self.track_df["segment"] = np.floor(self.track_df["km"] / self.seg_unit_km).astype(int)

    def _smooth_elevation_profile(self, mean_window: int = 5, min_distance_m: float = 1) -> None:
        """Smooth the elevation profile using mean and Savitzky-Golay filters.
        
        Args:
            mean_window (int): Window size for mean filter.
            savgol_window (int): Window size for Savitzky-Golay filter (must be odd).
            poly_order (int): Polynomial order for Savitzky-Golay filter.
            min_distance_m (float): Discard stacked points (eliminates spikes on plot) .
        """
        min_km = min_distance_m / 1000.0
        self.track_df['km_diff'] = self.track_df['km'].diff()
        self.track_df = self.track_df[self.track_df['km_diff'] >= min_km].reset_index(drop=True)

        elev_cleaned = self.track_df['elevation'].values
        mean_smoothed = pd.Series(elev_cleaned).rolling(window=mean_window, center=True, min_periods=1).mean()
        self.track_df['elev_smooth'] = mean_smoothed

    def _compute_datapoint_slopes(self) -> None:
        """Compute slope at each data point.
         1. Calculate elevation and distance differences between consecutive points.
         2. Compute slope as (delta_elev / delta_dist) * 100.
        """
        df = self.track_df.copy()
        df['delta_elev'] = df['elev_smooth'].diff()
        df['delta_dist'] = df['km'].diff() * 1000

        slope = (df['delta_elev'] / df['delta_dist']) * 100

        self.track_df['datapoint_slope'] = slope

    def _compute_segment_slopes(self) -> None:
        """Compute average slope for each segment."""
        slopes_df = (
            self.track_df.groupby("segment")
            .agg(segment_slope=('datapoint_slope', 'mean'))
            .reset_index()
        )
        self.track_df = self.track_df.merge(slopes_df, on="segment", how="left")

    def _load_cache(self, cache_file: str) -> Dict[str, Optional[str]]:
        """Load geolocation cache from a JSON file."""
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache(self, cache: Dict[str, Optional[str]], cache_file: str) -> None:
        """Save geolocation cache to a JSON file."""
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except IOError as e:
            raise IOError(f"Failed to save cache to {cache_file}") from e

    def _get_slope_bins(self, slope_thresholds: Tuple[float, ...]) -> Tuple[List[float], List[str]]:
        """Generate slope bins and corresponding labels."""
        thresholds = [-np.inf] + list(slope_thresholds) + [np.inf]
        labels = []
        for low, high in zip(thresholds[:-1], thresholds[1:]):
            if low == -np.inf:
                labels.append(f"< {high}%")
            elif high == np.inf:
                labels.append(f">= {low}%")
            else:
                labels.append(f"{low} ~ {high}%")
        return thresholds, labels

    # ========================
    # Public methods
    # ========================

    def geolocate_places(self, min_distance_km: float = 5, cache_file: str = "places_cache.json", rate_limit_sec: int = 1) -> None:
        """Geolocate significant places along the route using reverse geocoding.
        Args:
            min_distance_km (float): Minimum distance between places to consider them distinct.
            cache_file (str): Path to the JSON file for caching geolocation results.
            rate_limit_sec (int): Delay between geocoding requests to avoid rate limiting.
        """
        geolocator = Nominatim(user_agent="ElevationProfileApp")
        places: List[List[Any]] = []
        place_last_km: Dict[str, float] = {}
        place_group: int = 0
        cache = self._load_cache(cache_file)

        for segment in self.track_df["segment"].unique():
            segment_df = self.track_df[self.track_df["segment"] == segment]
            row = segment_df.loc[segment_df["km"].idxmin()]
            lat, lon, elev, km = row["latitude"], row["longitude"], row["elevation"], row["km"]
            coords_key = f"{lat:.5f},{lon:.5f}"

            place_name = cache.get(coords_key)
            if place_name is None:
                try:
                    location = geolocator.reverse(f"{lat},{lon}", exactly_one=True, timeout=10)
                    address = location.raw.get("address", {})
                    place_name = address.get("city") or address.get("town") or address.get("village")
                    cache[coords_key] = place_name
                    time.sleep(rate_limit_sec)
                except (GeocoderTimedOut, GeocoderUnavailable):
                    continue
                except Exception as e:
                    print(f"Geolocation error: {e}")
                    continue

            if place_name and ((place_name not in place_last_km) or (km - place_last_km[place_name] >= min_distance_km)):
                place_group += 1
                place_last_km[place_name] = km
                places.append([segment, place_name, elev, km, place_group])

        self._save_cache(cache, cache_file)
        self.places_df = pd.DataFrame(places, columns=["segment", "place", "elevation", "km", "group"])
        self.places_df = self.places_df.drop_duplicates(subset=["place"]).sort_values(["group", "km"])

    def compute_slope_lengths(self, slope_thresholds: Tuple[float, ...] = (2, 4, 6, 8), min_delta_km: float = 1e-4) -> pd.DataFrame:
        """
        Compute the total distance covered in different slope ranges.

        This method calculates the slope for each consecutive pair of track points
        using the smoothed elevation and cumulative distance. It then aggregates
        the total length of segments that fall within specified slope ranges.

        The slope is computed as:
            slope (%) = (delta_elevation / delta_distance_m) * 100

        Parameters
        ----------
        slope_thresholds : Tuple[float, ...], optional, default=(2, 4, 6, 8)
            A tuple of slope percentage thresholds used to define slope bins.
            For example, (2, 4, 6, 8) creates bins:
            - "< 2%", "2 ~ 4%", "4 ~ 6%", "6 ~ 8%", ">= 8%".
        min_delta_km : float, optional, default=1e-4
            Minimum segment length in kilometers to consider when calculating slope.
            This filters out very short noisy segments.

        Returns
        -------
        pd.DataFrame
            A DataFrame with three columns:
            - `slope_range` (str): The slope bin label (e.g., "< 2%", "4 ~ 6%").
            - `length_km` (float): Total distance in kilometers for this slope range.
            - `% of total` (float): Percentage of the total route length in this slope range.

        Notes
        -----
        - The method uses smoothed elevation (`elev_smooth`) from the track data.
        - Distance differences below `min_delta_km` are ignored to reduce noise.
        - Slope bins include the right edge (`right=True`).

        Examples
        --------
        >>> analyzer.compute_slope_lengths(slope_thresholds=(3, 6, 9))
        slope_range  length_km  % of total
        0       < 3%       12.34       65.43
        1    3 ~ 6%        4.56       24.18
        2    6 ~ 9%        1.23        6.52
        3      >= 9%       0.67        3.87
        """
        df = self.track_df.copy()
        df['delta_elev'] = df['elev_smooth'].diff()
        df['delta_km'] = df['km'].diff()
        df = df.dropna(subset=['delta_elev', 'delta_km'])
        df = df[df['delta_km'] > min_delta_km]

        df['slope'] = (df['delta_elev'] / (df['delta_km'] * 1000)) * 100
        df['segment_length_km'] = df['delta_km']

        thresholds, labels = self._get_slope_bins(slope_thresholds)
        df['slope_range'] = pd.cut(df['slope'], bins=thresholds, labels=labels, right=True)

        result = (
            df.groupby('slope_range')['segment_length_km']
            .sum()
            .reset_index()
            .rename(columns={'segment_length_km': 'length_km'})
        )

        return pd.DataFrame({
            'slope_range': result['slope_range'],
            'length_km': result['length_km'].round(2),
            '% of total': (result['length_km'] / result['length_km'].sum() * 100).round(2)
        })

    def plot_profile(
        self,
        show_labels: bool = True,
        show_background: bool = True,
        background_color: str = "gray",
        background_shift_km: float = 0.5,
        background_shift_elev: float = 15,
        slope_thresholds: Tuple[float, ...] = (2, 4, 6, 8),
        slope_colors: Tuple[str, ...] = ("lightgreen", "yellow", "orange", "orangered", "maroon"),
        slope_labels: Optional[List[str]] = None,
        slope_type: str = "segment"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the elevation profile of the route with color-coded slope zones.

        This method visualizes the route elevation profile and highlights sections
        based on slope categories. Slope zones are defined by `slope_thresholds`
        and colored according to `slope_colors`. The plot can optionally include
        background shading and place labels.

        Parameters
        ----------
        show_labels : bool, optional, default=True
            Whether to display annotated place names from `places_df`.
        show_background : bool, optional, default=True
            Whether to add a shifted background profile for visual depth.
        background_color : str, optional, default="gray"
            Color of the background profile shading.
        background_shift_km : float, optional, default=0.5
            Horizontal shift in kilometers for the background profile.
        background_shift_elev : float, optional, default=15
            Vertical shift in meters for the background profile.
        slope_thresholds : Tuple[float, ...], optional, default=(2, 4, 6, 8)
            Tuple of slope percentage thresholds for defining slope bins.
        slope_colors : Tuple[str, ...], optional, default=("lightgreen", "yellow", "orange", "orangered", "maroon")
            Colors corresponding to each slope bin.
        slope_labels : Optional[List[str]], optional
            Custom labels for slope bins. If None, default labels are generated.
        slope_type : str, optional, default="segment"
            Type of slope metric to use for color-coding:
            - `"segment"`: Uses precomputed segment slopes.
            - `"datapoint"`: Uses point-to-point slopes.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            The created Matplotlib Figure and Axes objects for further customization.

        Raises
        ------
        ValueError
            If `slope_colors` length does not match `slope_labels` length.
            If `slope_type` is not `"segment"` or `"datapoint"`.

        Notes
        -----
        - Requires `track_df` with columns: `km`, `elev_smooth`, and slope columns.
        - If `places_df` is available, annotated labels will be plotted.

        Examples
        --------
        >>> fig, ax = analyzer.plot_profile(
        ...     show_labels=True,
        ...     slope_thresholds=(3, 6, 9),
        ...     slope_colors=("lightblue", "green", "orange", "red")
        ... )
        >>> plt.show()
        """
        thresholds, default_labels = self._get_slope_bins(slope_thresholds)
        if slope_labels is None:
            slope_labels = default_labels
        if len(slope_colors) != len(slope_labels):
            raise ValueError("slope_colors length must match slope_labels length.")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_xlabel("Kilometers")
        ax.set_ylabel("Elevation [m]")
        ax.spines[["right", "top"]].set_visible(False)

        if show_background:
            ax.fill_between(
                self.track_df["km"] + background_shift_km,
                self.track_df['elev_smooth'] + background_shift_elev,
                color=background_color,
                zorder=0,
            )

        legend = []
        for i, color in enumerate(slope_colors):
            if slope_type == "segment":
                mask = (self.track_df["segment_slope"] >= thresholds[i]) & (self.track_df["segment_slope"] < thresholds[i + 1])
            elif slope_type == "datapoint":
                mask = (self.track_df["datapoint_slope"] >= thresholds[i]) & (self.track_df["datapoint_slope"] < thresholds[i + 1])
            else:
                raise ValueError("slope_type must be 'segment' or 'datapoint'.")
            ax.fill_between(self.track_df["km"], self.track_df['elev_smooth'], where=mask, color=color, zorder=1, interpolate=True)
            legend.append(mpatches.Patch(color=color, label=slope_labels[i]))

        if show_labels and self.places_df is not None:
            annotations_anchor = self.track_df["elevation"].max() * 1.1
            last_label_km = -5
            for _, row in self.places_df.iterrows():
                if row["km"] - last_label_km >= 5:
                    ax.annotate(
                        row["place"],
                        xy=(row["km"], row["elevation"]),
                        xytext=(row["km"], annotations_anchor),
                        arrowprops=dict(arrowstyle="-", color="lightgray"),
                        horizontalalignment="center",
                        rotation=90,
                        size=10,
                        color="gray",
                    )
                    last_label_km = row["km"]

        ax.plot(self.track_df["km"], self.track_df['elev_smooth'], color="darkgrey", linewidth=0.15)
        y_lower_bound = math.floor(self.track_df["elevation"].min() * 0.8 / 100) * 100
        ax.set_ylim(y_lower_bound, self.track_df["elevation"].max() * 1.1)
        ax.set_xlim(self.track_df["km"].min(), self.track_df["km"].max())
        ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1, 0.5))

        return fig, ax

    def detect_climbs(
        self,
        min_length_m: float = 500,
        min_gain_m: float = 30,
        min_avg_slope: float = 2.0,
        window_m: float = 200,
        merge_gap_m: float = 100
    ) -> pd.DataFrame:
        """
        Detect climbs along the route based on distance, elevation gain, and slope criteria.

        This method identifies uphill segments of the route that meet minimum thresholds for
        length, elevation gain, and average slope. It uses a rolling window to smooth the
        slope profile, then merges nearby climbs and filters out insignificant segments.

        Parameters
        ----------
        min_length_m : float, optional, default=500
            Minimum climb length in meters to consider as a valid climb.
        min_gain_m : float, optional, default=30
            Minimum elevation gain in meters for a segment to be considered a climb.
        min_avg_slope : float, optional, default=2.0
            Minimum average slope (percentage) required for a segment to qualify as a climb.
        window_m : float, optional, default=200
            Length in meters for the rolling window used to compute smoothed slope.
            Larger values make the detection less sensitive to short fluctuations.
        merge_gap_m : float, optional, default=100
            Maximum gap in meters between adjacent climbs to merge them into a single climb.

        Returns
        -------
        pd.DataFrame
            DataFrame containing detected climbs with columns:
            - `start_idx` (int): Start index in the original track.
            - `end_idx` (int): End index in the original track.
            - `start_km` (float): Start position in kilometers.
            - `end_km` (float): End position in kilometers.
            - `length_m` (float): Climb length in meters.
            - `gain_m` (float): Elevation gain in meters.
            - `avg_grade_pct` (float): Average slope percentage.
            - `max_grade_pct` (float): Maximum slope percentage within the climb.
            - `start_lat`, `start_lon` (float): Coordinates of climb start point.
            - `end_lat`, `end_lon` (float): Coordinates of climb end point.

        Notes
        -----
        - Uses smoothed elevation (`elev_smooth`) for slope calculation.
        - Climb detection is based on rolling slope over `window_m` and then merging
        adjacent uphill segments that are close to each other.
        - Very short or low-gain uphill segments are excluded.

        Examples
        --------
        >>> climbs_df = analyzer.detect_climbs(min_length_m=1000, min_gain_m=50)
        >>> climbs_df.head()
        start_km  end_km  length_m  gain_m  avg_grade_pct  max_grade_pct
        0      2.10    3.50   1400.0    60.0            4.3            7.8
        """
        df = self.track_df.copy()
        dist = df['km'].values * 1000
        elev = df['elev_smooth'].values
        slopes = df['datapoint_slope'].values

        if len(dist) < 2:
            return pd.DataFrame([])

        window_pts = max(2, int(window_m / np.mean(np.diff(dist))))
        rolling_gain = np.convolve(np.r_[0, np.diff(elev)], np.ones(window_pts), mode='same')
        rolling_dist = np.convolve(np.r_[0, np.diff(dist)], np.ones(window_pts), mode='same')
        rolling_slope = (rolling_gain / rolling_dist) * 100
        rolling_slope[np.isnan(rolling_slope)] = 0

        is_uphill = rolling_slope > (min_avg_slope / 2)
        regions = []
        start_idx = None

        for i, uphill in enumerate(is_uphill):
            if uphill and start_idx is None:
                start_idx = i
            elif not uphill and start_idx is not None:
                if i - start_idx > 1:
                    regions.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            regions.append((start_idx, len(is_uphill) - 1))

        merged = []
        for seg in regions:
            if not merged:
                merged.append(seg)
            else:
                prev_start, prev_end = merged[-1]
                if (dist[seg[0]] - dist[prev_end]) <= merge_gap_m:
                    merged[-1] = (prev_start, seg[1])
                else:
                    merged.append(seg)

        climbs = []
        for (s, e) in merged:
            start_row = df.iloc[s]
            end_row = df.iloc[e]
            length_m = (end_row['km'] - start_row['km']) * 1000
            gain_m = elev[e] - elev[s]
            if length_m < min_length_m or gain_m < min_gain_m:
                continue
            avg_slope = (gain_m / length_m) * 100
            if avg_slope < min_avg_slope:
                continue

            climbs.append({
                "start_idx": s,
                "end_idx": e,
                "start_km": start_row["km"],
                "end_km": end_row["km"],
                "length_m": round(length_m, 1),
                "gain_m": round(gain_m, 1),
                "avg_grade_pct": round(avg_slope, 1),
                "max_grade_pct": round(np.max(slopes[s:e + 1]), 1),
                "start_lat": start_row["latitude"],
                "start_lon": start_row["longitude"],
                "end_lat": end_row["latitude"],
                "end_lon": end_row["longitude"]
            })

        return pd.DataFrame(climbs)

    def get_total_ascent(self) -> float:
        """
        Calculate the total positive elevation gain along the route.

        This method sums all positive elevation changes between consecutive points,
        based on the smoothed elevation profile.

        Returns
        -------
        float
            Total ascent in meters.

        Notes
        -----
        - Uses `elev_smooth` column to reduce noise in elevation data.
        - Only positive differences (uphill segments) are considered.

        Examples
        --------
        >>> profile.get_total_ascent()
        1250.5
        """
        elevation_diff = self.track_df["elev_smooth"].diff()
        return float(elevation_diff[elevation_diff > 0].sum())

    def get_total_descent(self) -> float:
        """
        Calculate the total negative elevation loss along the route.

        This method sums all negative elevation changes between consecutive points,
        based on the smoothed elevation profile.

        Returns
        -------
        float
            Total descent in meters (positive value).

        Notes
        -----
        - Uses `elev_smooth` column to reduce noise in elevation data.
        - Only negative differences (downhill segments) are considered.

        Examples
        --------
        >>> profile.get_total_descent()
        1248.0
        """
        elevation_diff = self.track_df["elev_smooth"].diff()
        return float(-elevation_diff[elevation_diff < 0].sum())

    def get_highest_point(self) -> float:
        """
        Get the highest elevation point on the route.

        Returns
        -------
        float
            Maximum elevation in meters based on the smoothed elevation profile.

        Examples
        --------
        >>> profile.get_highest_point()
        1825.0
        """
        return float(self.track_df["elev_smooth"].max())

    def get_lowest_point(self) -> float:
        """
        Get the lowest elevation point on the route.

        Returns
        -------
        float
            Minimum elevation in meters based on the smoothed elevation profile.

        Examples
        --------
        >>> profile.get_lowest_point()
        340.0
        """
        return float(self.track_df["elev_smooth"].min())

    def get_route_data(self) -> pd.DataFrame:
        """
        Retrieve the full processed route data.

        This includes the original track columns along with computed values such as:
        - `elev_smooth` (smoothed elevation)
        - `segment` (segment index)
        - `datapoint_slope` (instantaneous slope)
        - `segment_slope` (average slope per segment)

        Returns
        -------
        pd.DataFrame
            Copy of the processed track DataFrame.

        Examples
        --------
        >>> df = profile.get_route_data()
        >>> df.head()
            km  elevation  elev_smooth  latitude  longitude  segment  datapoint_slope segment_slope
        0  0.0     340.0       340.0   47.1234    10.1234        0              0.0         0.0
        """
        return self.track_df.copy()



if __name__ == "__main__":
    from gpx_parser import GPXParser

    gpx_file = "data/track/orbita25.gpx"
    parser = GPXParser(gpx_file)
    track_df = parser.parse_to_dataframe()

    profile = ElevationProfile(track_df)
    #profile.geolocate_places(min_distance_km=10)

    fig, ax = profile.plot()
    fig.set_size_inches(12, 4)
    # ax.set_title("Profil wysokości - Orbita 25")
    ax.set_xlabel("Dystans [km]")
    ax.set_ylabel("Wysokość [m]")
    ax.set_ylim(200, profile.track_df["elevation"].max() * 1.1)
    #plt.savefig("static/elevation_profile.png", bbox_inches="tight", dpi=300)
    plt.show()