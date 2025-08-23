import gpxpy
import pandas as pd
from geopy.distance import geodesic
import io

class GPXParser:
    """Parser GPX -> DataFrame"""

    def __init__(self, gpx_source):
        """
        gpx_source może być:
        - ścieżką do pliku GPX (str)
        - bajtami pliku (bytes)
        - obiektem plikowym (np. z file_uploader)
        """
        self.gpx_source = gpx_source
        self.track_df = None

    def _get_distance(self, lat1, lon1, lat2, lon2):
        if None in (lat1, lon1, lat2, lon2):
            return 0
        return geodesic((lat1, lon1), (lat2, lon2)).km

    def parse_to_dataframe(self):
        """Parsuje GPX i zwraca DataFrame z kolumnami: km, latitude, longitude, elevation"""
        # Wczytanie GPX z różnych źródeł
        if isinstance(self.gpx_source, (bytes, bytearray)):
            gpx_file = io.BytesIO(self.gpx_source)
            gpx_text = gpx_file.read().decode("utf-8")
        elif hasattr(self.gpx_source, "read"):  # plik w pamięci (np. Streamlit UploadedFile)
            gpx_text = self.gpx_source.read().decode("utf-8")
        elif isinstance(self.gpx_source, str):  # ścieżka pliku
            with open(self.gpx_source, "r", encoding="utf-8") as f:
                gpx_text = f.read()
        else:
            raise ValueError("Nieobsługiwany typ źródła GPX.")

        gpx = gpxpy.parse(io.StringIO(gpx_text))

        track_data = []
        km = 0
        last_lat, last_lon = None, None

        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    dist = self._get_distance(last_lat, last_lon, point.latitude, point.longitude)
                    km += dist
                    track_data.append([km, point.latitude, point.longitude, point.elevation])
                    last_lat, last_lon = point.latitude, point.longitude

        self.track_df = pd.DataFrame(track_data, columns=["km", "latitude", "longitude", "elevation"])
        if self.track_df.empty:
            raise ValueError("Brak punktów w ścieżce GPX.")
        return self.track_df
