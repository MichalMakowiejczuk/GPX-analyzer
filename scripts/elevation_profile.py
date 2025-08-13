import json
import os
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable


class ElevationProfile:
    """
    Analiza i wizualizacja profilu wysokościowego trasy.

    Atrybuty
    --------
    track_df : pd.DataFrame
        Dane trasy zawierające kolumny: ['km', 'elevation', 'latitude', 'longitude'].
    seg_unit_km : float
        Długość odcinka (w km) używanego do segmentacji.
    places_df : pd.DataFrame | None
        Wyniki geolokacji punktów charakterystycznych.
    """

    def __init__(self, track_df: pd.DataFrame, seg_unit_km: float = 0.5):
        if track_df.empty:
            raise ValueError("DataFrame jest pusty.")
        required_cols = {"km", "elevation", "latitude", "longitude"}
        if not required_cols.issubset(track_df.columns):
            raise ValueError(f"DataFrame musi zawierać kolumny: {required_cols}")

        self.track_df = track_df.copy()
        self.seg_unit_km = seg_unit_km
        self._assign_segments()
        self._compute_slopes()
        self.places_df = None

    # ========================
    # Metody prywatne
    # ========================

    def _assign_segments(self) -> None:
        """Przypisuje numer segmentu na podstawie odległości i jednostki segmentacji."""
        self.track_df["segment"] = np.floor(self.track_df["km"] / self.seg_unit_km).astype(int)

    def _compute_slopes(self) -> None:
        """Oblicza średnie nachylenie dla każdego segmentu."""
        slopes_df = (
            self.track_df.groupby("segment")
            .agg(
                start_km=("km", "first"),
                end_km=("km", "last"),
                start_elev=("elevation", "first"),
                end_elev=("elevation", "last"),
            )
            .assign(
                slope=lambda df: np.where(
                    df["end_km"] == df["start_km"], 0,
                    (df["end_elev"] - df["start_elev"]) /
                    ((df["end_km"] - df["start_km"]) * 1000) * 100
                )
            )
            .reset_index()[["segment", "slope"]]
        )
        self.track_df = self.track_df.merge(slopes_df, on="segment", how="left")

    def _load_cache(self, cache_file: str) -> dict:
        """Ładuje cache miejscowości z pliku JSON."""
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache: dict, cache_file: str) -> None:
        """Zapisuje cache miejscowości do pliku JSON."""
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def _get_slope_bins(self, slope_thresholds):
        """Zwraca progi i etykiety dla zakresów nachyleń."""
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
    # Metody publiczne
    # ========================

    def geolocate_places(self, min_distance_km=5, cache_file="places_cache.json", rate_limit_sec=1):
        """
        Wyszukuje miejscowości wzdłuż trasy.
        """
        geolocator = Nominatim(user_agent="ElevationProfileApp")
        places = []
        place_last_km = {}
        place_group = 0
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
                    time.sleep(rate_limit_sec)  # unikanie blokad Nominatim
                except (GeocoderTimedOut, GeocoderUnavailable):
                    continue

            if place_name and ((place_name not in place_last_km) or (km - place_last_km[place_name] >= min_distance_km)):
                place_group += 1
                place_last_km[place_name] = km
                places.append([segment, place_name, elev, km, place_group])

        self._save_cache(cache, cache_file)
        self.places_df = pd.DataFrame(places, columns=["segment", "place", "elevation", "km", "group"])
        self.places_df = self.places_df.drop_duplicates(subset=["place"]).sort_values(["group", "km"])

    def smooth_profile(self, smooth_window=5):
        """Zwraca wygładzony profil wysokościowy."""
        return self.track_df['elevation'].rolling(window=smooth_window, center=True, min_periods=1).mean()

    def compute_slope_lengths(self, smooth_window=5, slope_thresholds=(2, 4, 6, 8), min_delta_km=1e-4):
        """
        Oblicza długość odcinków w zadanych zakresach nachylenia.
        """
        df = self.track_df.copy()
        df['elev_smooth'] = self.smooth_profile(smooth_window)
        df['delta_elev'] = df['elev_smooth'].diff()
        df['delta_km'] = df['km'].diff()

        df = df.dropna(subset=['delta_elev', 'delta_km'])
        df = df[df['delta_km'] > min_delta_km]  # filtr minimalnej długości

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

    def plot(
        self,
        show_labels=True,
        show_background=True,
        background_color="gray",
        background_shift_km=0.5,
        background_shift_elev=15,
        smooth_window=5,
        slope_thresholds=(2, 4, 5, 8),
        slope_colors=("lightgreen", "yellow", "orange", "orangered", "maroon"),
        slope_labels=None
    ):
        """
        Rysuje profil wysokości z kolorami nachylenia.
        """
        thresholds, default_labels = self._get_slope_bins(slope_thresholds)
        if slope_labels is None:
            slope_labels = default_labels
        if len(slope_colors) != len(slope_labels):
            raise ValueError("Długość slope_colors musi być równa długości slope_labels.")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_xlabel("Kilometers")
        ax.set_ylabel("Elevation [m]")
        ax.spines[["right", "top"]].set_visible(False)

        elevation_smooth = self.smooth_profile(smooth_window)

        if show_background:
            ax.fill_between(
                self.track_df["km"] + background_shift_km,
                elevation_smooth + background_shift_elev,
                color=background_color,
                zorder=0,
            )

        legend = []
        for i, color in enumerate(slope_colors):
            mask = (self.track_df["slope"] >= thresholds[i]) & (self.track_df["slope"] < thresholds[i + 1])
            ax.fill_between(self.track_df["km"], elevation_smooth, where=mask, color=color, zorder=1, interpolate=True)
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

        ax.plot(self.track_df["km"], elevation_smooth, color="darkgrey", linewidth=0.15)
        y_lower_bound = math.floor(self.track_df["elevation"].min() * 0.8 / 100) *100
        ax.set_ylim(y_lower_bound, self.track_df["elevation"].max() * 1.1)
        ax.set_xlim(self.track_df["km"].min(), self.track_df["km"].max())
        ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1, 0.5))

        return fig, ax
    
    def detect_climbs(self, min_length_m=500, min_gain_m=30, smooth_window=5, min_avg_slope=2.0):
        """
        Wykrywa podjazdy spełniające kryteria.
        Podjazd = odcinek o dodatnim przewyższeniu i wymaganej długości.
        """
        df = self.track_df.copy()
        df['elev_smooth'] = self.smooth_profile(smooth_window)

        climbs = []
        in_climb = False
        start_idx = None
        gain = 0.0
        length = 0.0

        for i in range(1, len(df)):
            delta_h = df.loc[i, 'elev_smooth'] - df.loc[i-1, 'elev_smooth']
            delta_d = (df.loc[i, 'km'] - df.loc[i-1, 'km']) * 1000  # m

            if delta_h > 0:
                if not in_climb:
                    in_climb = True
                    start_idx = i-1
                    gain = 0.0
                    length = 0.0
                gain += delta_h
                length += delta_d
            else:
                if in_climb:
                    avg_slope = (gain / length * 100) if length > 0 else 0
                    if length >= min_length_m and gain >= min_gain_m and avg_slope >= min_avg_slope:
                        start_row = df.loc[start_idx]
                        end_row = df.loc[i-1]
                        avg_grade = (gain / length) * 100
                        climbs.append({
                            "start_idx": start_idx,
                            "end_idx": i-1,
                            "start_km": start_row["km"],
                            "end_km": end_row["km"],
                            "length_m": round(length, 1),
                            "gain_m": round(gain, 1),
                            "avg_grade_pct": round(avg_grade, 1),
                            "start_lat": start_row["latitude"],
                            "start_lon": start_row["longitude"],
                            "end_lat": end_row["latitude"],
                            "end_lon": end_row["longitude"]
                        })
                    in_climb = False

        return pd.DataFrame(climbs)


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