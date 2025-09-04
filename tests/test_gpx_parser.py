import io

import pandas as pd
import pytest

from scripts.gpx_parser import GPXParser

VALID_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="pytest">
  <trk><name>Test Track</name><trkseg>
    <trkpt lat="50.06143" lon="19.93658"><ele>200.0</ele></trkpt>
    <trkpt lat="50.06200" lon="19.94000"><ele>205.0</ele></trkpt>
  </trkseg></trk>
</gpx>
"""

EMPTY_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="pytest">
  <trk><name>Empty Track</name><trkseg>
  </trkseg></trk>
</gpx>
"""


def test_parse_from_str(tmp_path):
    """Parsing from source file (str)."""
    file_path = tmp_path / "test.gpx"
    file_path.write_text(VALID_GPX, encoding="utf-8")

    parser = GPXParser(str(file_path))
    df = parser.parse_to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["km", "latitude", "longitude", "elevation"]
    assert df["km"].iloc[0] == pytest.approx(0.0)


def test_parse_from_bytes():
    """Parsing from bytes"""
    gpx_bytes = VALID_GPX.encode("utf-8")
    parser = GPXParser(gpx_bytes)
    df = parser.parse_to_dataframe()

    assert len(df) == 2
    assert df["elevation"].iloc[1] == 205.0


def test_parse_from_filelike():
    """Parsing from file-like (np. upload)."""
    gpx_bytesio = io.BytesIO(VALID_GPX.encode("utf-8"))
    parser = GPXParser(gpx_bytesio)
    df = parser.parse_to_dataframe()

    assert df.iloc[0]["latitude"] == pytest.approx(50.06143, rel=1e-5)


def test_empty_gpx_raises(tmp_path):
    """Empty track should give ValueError."""
    file_path = tmp_path / "empty.gpx"
    file_path.write_text(EMPTY_GPX, encoding="utf-8")

    parser = GPXParser(str(file_path))
    with pytest.raises(ValueError, match="No points found"):
        parser.parse_to_dataframe()


def test_get_distance():
    """Test for the helper method _get_distance method."""
    parser = GPXParser("dummy.gpx")
    d = parser._get_distance(50.06143, 19.93658, 50.06200, 19.94000)

    assert d > 0.2
    # If one point None -> 0
    assert parser._get_distance(None, 19.9, 50.0, 19.9) == 0
