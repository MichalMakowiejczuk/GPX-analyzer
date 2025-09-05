import math
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from .utils import _get_slope_bins


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
        slope_colors: Tuple[str, ...] = (
            "lightgreen",
            "yellow",
            "orange",
            "orangered",
            "maroon",
        ),
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
                mask = (self.df["segment_slope"] >= thresholds[i]) & (
                    self.df["segment_slope"] < thresholds[i + 1]
                )
            else:
                mask = (self.df["datapoint_slope"] >= thresholds[i]) & (
                    self.df["datapoint_slope"] < thresholds[i + 1]
                )

            # znajdź spójne fragmenty maski
            in_block = False
            start_idx = None
            for j in range(len(mask)):
                if mask.iloc[j] and not in_block:
                    in_block = True
                    start_idx = j
                elif not mask.iloc[j] and in_block:
                    # koniec bloku -> rysujemy
                    ax.fill_between(
                        self.df["km"].iloc[start_idx : j + 1],
                        self.df["elev_smooth"].iloc[start_idx : j + 1],
                        color=color,
                        zorder=1,
                    )
                    in_block = False
            # domknij ostatni fragment
            if in_block:
                ax.fill_between(
                    self.df["km"].iloc[start_idx:],
                    self.df["elev_smooth"].iloc[start_idx:],
                    color=color,
                    zorder=1,
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
        y_lower_bound = (
            math.floor(float(self.df["elevation"].min()) * 0.9 / 100.0) * 100.0
        )
        ax.set_ylim(y_lower_bound, float(self.df["elevation"].max()) * 1.1)
        ax.set_xlim(float(self.df["km"].min()), float(self.df["km"].max()))
        ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        return fig, ax
