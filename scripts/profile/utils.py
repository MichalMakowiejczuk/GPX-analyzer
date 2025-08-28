from typing import List, Tuple

import numpy as np
import pandas as pd


def _get_slope_bins(
    slope_thresholds: Tuple[float, ...],
) -> Tuple[List[float], List[str]]:
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


def _validate_track_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and non-empty DataFrame; return a copy."""
    required = {"km", "elevation", "latitude", "longitude"}
    if df is None or df.empty:
        raise ValueError("The input DataFrame is empty.")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"The DataFrame must contain the following columns: {required}; missing: {missing}"
        )
    return df.copy()
