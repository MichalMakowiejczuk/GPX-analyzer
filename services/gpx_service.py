from typing import Tuple

import pandas as pd
import streamlit as st

from scripts.gpx_parser import GPXParser
from scripts.profile import ElevationProfile


@st.cache_data
def load_gpx(
    uploaded_file: "UploadedFile", smooth_window: float, seg_unit_km: float = 0.25
) -> Tuple[pd.DataFrame, ElevationProfile]:
    parser = GPXParser(uploaded_file.read())
    track_df = parser.parse_to_dataframe()
    profile = ElevationProfile(
        track_df, seg_unit_km=seg_unit_km, smooth_window=smooth_window
    )
    return track_df, profile
