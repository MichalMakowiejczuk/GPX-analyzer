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
from scipy.signal import savgol_filter


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

    def __init__(self, track_df: pd.DataFrame, seg_unit_km: float = 0.5, smooth_window: int = 5):
        if track_df.empty:
            raise ValueError("DataFrame jest pusty.")
        required_cols = {"km", "elevation", "latitude", "longitude"}
        if not required_cols.issubset(track_df.columns):
            raise ValueError(f"DataFrame musi zawierać kolumny: {required_cols}")

        self.track_df = track_df.copy()
        self.seg_unit_km = seg_unit_km
        self.smooth_window = smooth_window
        self._smooth_profile(self.smooth_window, )
        self._assign_segments()
        self._compute_slope_for_each_datapoint()
        self._compute_slopes_for_segments()
        self.places_df = None

    # ========================
    # Metody prywatne
    # ========================
    def _assign_segments(self) -> None:
        """Przypisuje numer segmentu na podstawie odległości i jednostki segmentacji."""
        self.track_df["segment"] = np.floor(self.track_df["km"] / self.seg_unit_km).astype(int)

    def _smooth_profile(self, median_window=5, savgol_window=5, poly_order=2):
        """
        Wygładza profil wysokościowy używając kombinacji filtra medianowego i Savitzky-Golay.
        
        Parametry:
        ----------
        median_window : int
            Okno filtra medianowego (usuwa skoki/outliery). Powinno być nieparzyste.
        savgol_window : int
            Okno filtra Savitzky-Golay (wygładza krzywą). Musi być nieparzyste.
        poly_order : int
            Rząd wielomianu dla Savitzky-Golay.
        """
        median_smoothed = self.track_df['elevation'].rolling(
            window=median_window, center=True, min_periods=1
        ).median()

        self.track_df['elev_smooth'] = savgol_filter(
            median_smoothed, window_length=savgol_window, polyorder=poly_order
        )
    
    def _compute_slope_for_each_datapoint(self, min_dist_m: float = 7) -> None:
        """
        Oblicza nachylenie dla każdego punktu na podstawie różnicy wysokości i odległości,
        filtrując bardzo krótkie odcinki, ale wypełniając je nachyleniem sąsiednich punktów.
        """
        df = self.track_df.copy()

        df['delta_elev'] = df['elev_smooth'].diff()
        df['delta_dist'] = df['km'].diff() * 1000  # km → m

        # Obliczamy nachylenie punktowe
        slope = (df['delta_elev'] / df['delta_dist']) * 100

        # Zastępujemy wartości dla krótkich odcinków
        short_mask = df['delta_dist'] < min_dist_m
        slope[short_mask] = np.nan

        # Wypełnienie braków np. metodą 'ffill' (wartość poprzedniego punktu) lub 'bfill'
        slope = slope.fillna(method='ffill').fillna(method='bfill')

        self.track_df['datapoint_slope'] = slope

    def _compute_slopes_for_segments(self) -> None:
        """Oblicza średnie nachylenie dla każdego segmentu na podstawie nachylenia punktowego."""
        slopes_df = (
            self.track_df.groupby("segment")
            .agg(segment_slope=('datapoint_slope', 'mean'))
            .reset_index()
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

    def compute_slope_lengths(self, slope_thresholds=(2, 4, 6, 8), min_delta_km=1e-4):
        """
        Oblicza długość odcinków w zadanych zakresach nachylenia.
        """
        df = self.track_df.copy()
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
        slope_thresholds=(2, 4, 6, 8),
        slope_colors=("lightgreen", "yellow", "orange", "orangered", "maroon"),
        slope_labels=None,
        slope_type="segment"  # "segment" lub "datapoint"
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
                raise ValueError("slope_type musi być 'segment' lub 'datapoint'.")
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
        y_lower_bound = math.floor(self.track_df["elevation"].min() * 0.8 / 100) *100
        ax.set_ylim(y_lower_bound, self.track_df["elevation"].max() * 1.1)
        ax.set_xlim(self.track_df["km"].min(), self.track_df["km"].max())
        ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1, 0.5))

        return fig, ax
    
    def detect_climbs(self, 
                  min_length_m=500, 
                  min_gain_m=30, 
                  min_avg_slope=2.0,
                  max_tolerant_drop_len=500,     # m - maks. długość tolerowanego spadku
                  max_tolerant_drop_slope=12.0):  # % - maks. nachylenie tolerowanego spadku
        """
        Wykrywa podjazdy spełniające kryteria, z tolerancją na krótkie, łagodne spadki w środku.
        Na końcu odcina końcowy zjazd, jeśli po nim nie ma już podjazdu.
        """

        df = self.track_df.copy()

        climbs = []
        in_climb = False
        start_idx = None
        gain = 0.0  # suma dodatnich przyrostów wysokości

        # Bufor aktualnego spadku
        drop_len = 0.0
        drop_gain_loss = 0.0
        last_up_idx = None  # indeks ostatniego punktu „w górę”

        for i in range(1, len(df)):
            delta_h = df.loc[i, 'elev_smooth'] - df.loc[i-1, 'elev_smooth']
            delta_d = (df.loc[i, 'km'] - df.loc[i-1, 'km']) * 1000  # m

            if delta_h > 0:  # odcinek w górę
                if not in_climb:
                    in_climb = True
                    start_idx = i-1
                    gain = 0.0
                    drop_len = 0.0
                    drop_gain_loss = 0.0

                gain += delta_h
                # reset bufora spadku
                drop_len = 0.0
                drop_gain_loss = 0.0
                last_up_idx = i

            else:  # spadek lub płasko
                if in_climb:
                    drop_len += delta_d
                    drop_gain_loss += delta_h  # ujemne lub zero

                    drop_slope = abs(drop_gain_loss / drop_len * 100) if drop_len > 0 else 0.0

                    if drop_len <= max_tolerant_drop_len and drop_slope <= max_tolerant_drop_slope:
                        # tolerujemy spadek w środku
                        continue
                    else:
                        # koniec podjazdu — odcinamy końcowy spadek
                        end_idx = last_up_idx if last_up_idx is not None else i-1

                        start_row = df.loc[start_idx]
                        end_row = df.loc[end_idx]
                        length_m = (end_row["km"] - start_row["km"]) * 1000
                        avg_slope = (gain / length_m * 100) if length_m > 0 else 0.0

                        if length_m >= min_length_m and gain >= min_gain_m and avg_slope >= min_avg_slope:
                            climbs.append({
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                                "start_km": start_row["km"],
                                "end_km": end_row["km"],
                                "length_m": round(length_m, 1),
                                "gain_m": round(gain, 1),
                                "avg_grade_pct": round(avg_slope, 1),
                                "max_grade_pct": round(max(df.loc[start_idx:end_idx, 'datapoint_slope']), 1),
                                "start_lat": start_row["latitude"],
                                "start_lon": start_row["longitude"],
                                "end_lat": end_row["latitude"],
                                "end_lon": end_row["longitude"]
                            })

                        # reset stanu
                        in_climb = False
                        start_idx = None
                        gain = 0.0
                        drop_len = 0.0
                        drop_gain_loss = 0.0
                        last_up_idx = None

        # Obsługa końca trasy
        if in_climb:
            if drop_len > 0 and last_up_idx is not None:
                end_idx = last_up_idx  # ucinamy końcowy spadek
            else:
                end_idx = len(df) - 1

            start_row = df.loc[start_idx]
            end_row = df.loc[end_idx]
            length_m = (end_row["km"] - start_row["km"]) * 1000
            avg_slope = (gain / length_m * 100) if length_m > 0 else 0.0

            if length_m >= min_length_m and gain >= min_gain_m and avg_slope >= min_avg_slope:
                climbs.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_km": start_row["km"],
                    "end_km": end_row["km"],
                    "length_m": round(length_m, 1),
                    "gain_m": round(gain, 1),
                    "avg_grade_pct": round(avg_slope, 1),
                    "max_grade_pct": round(max(df.loc[start_idx:end_idx, 'datapoint_slope']), 1),
                    "start_lat": start_row["latitude"],
                    "start_lon": start_row["longitude"],
                    "end_lat": end_row["latitude"],
                    "end_lon": end_row["longitude"]
                })

        return pd.DataFrame(climbs)

    def get_total_ascent(self):
        """Całkowite przewyższenie"""
        elevation_diff = self.track_df["elev_smooth"].diff()
        total_ascent = elevation_diff[elevation_diff > 0].sum()
        return total_ascent
    
    def get_total_descent(self):
        """Całkowite zjazd"""
        elevation_diff = self.track_df["elev_smooth"].diff()
        total_descent = -elevation_diff[elevation_diff < 0].sum()
        return total_descent
    
    def get_highest_point(self):
        """Najwyższy punkt trasy"""
        return self.track_df["elev_smooth"].max()
    
    def get_lowest_point(self):
        """Najniższy punkt trasy"""
        return self.track_df["elev_smooth"].min()



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