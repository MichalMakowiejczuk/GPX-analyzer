import os
import time
import json
import pandas as pd
from typing import Any, Dict, List, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

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