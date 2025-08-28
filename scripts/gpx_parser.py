import io

import gpxpy
import pandas as pd
from geopy.distance import geodesic


class GPXParser:
    """Parse GPX data into a pandas DataFrame."""

    def __init__(self, gpx_source):
        """
        Initialize the GPX parser.

        Parameters
        ----------
        gpx_source : str | bytes | file-like object
            The GPX data source, which can be:
            - A file path to a GPX file (str)
            - GPX file content as bytes
            - A file-like object (e.g., from an upload widget)
        """
        self.gpx_source = gpx_source
        self.track_df = None

    def _get_distance(self, lat1, lon1, lat2, lon2) -> float:
        """
        Compute the distance between two geographic coordinates in kilometers.

        Parameters
        ----------
        lat1, lon1 : float
            Latitude and longitude of the first point.
        lat2, lon2 : float
            Latitude and longitude of the second point.

        Returns
        -------
        float
            Distance in kilometers. Returns 0 if any coordinate is None.
        """
        if None in (lat1, lon1, lat2, lon2):
            return 0
        return geodesic((lat1, lon1), (lat2, lon2)).km

    def parse_to_dataframe(self) -> pd.DataFrame:
        """
        Parse the GPX data into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the following columns:
            - 'km': cumulative distance in kilometers
            - 'latitude': latitude of the point
            - 'longitude': longitude of the point
            - 'elevation': elevation in meters

        Raises
        ------
        ValueError
            If the GPX source type is unsupported or the parsed track is empty.
        """
        # Load GPX data from various input types
        if isinstance(self.gpx_source, (bytes, bytearray)):
            gpx_text = io.BytesIO(self.gpx_source).read().decode("utf-8")
        elif hasattr(self.gpx_source, "read"):  # File-like object
            gpx_text = self.gpx_source.read().decode("utf-8")
        elif isinstance(self.gpx_source, str):  # File path
            with open(self.gpx_source, "r", encoding="utf-8") as f:
                gpx_text = f.read()
        else:
            raise ValueError("Unsupported GPX source type.")

        # Parse GPX content
        gpx = gpxpy.parse(io.StringIO(gpx_text))

        track_data = []
        km = 0.0
        last_lat, last_lon = None, None

        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    dist = self._get_distance(
                        last_lat, last_lon, point.latitude, point.longitude
                    )
                    km += dist
                    track_data.append(
                        [km, point.latitude, point.longitude, point.elevation]
                    )
                    last_lat, last_lon = point.latitude, point.longitude

        # Create DataFrame
        self.track_df = pd.DataFrame(
            track_data, columns=["km", "latitude", "longitude", "elevation"]
        )
        if self.track_df.empty:
            raise ValueError("No points found in the GPX track.")

        return self.track_df
