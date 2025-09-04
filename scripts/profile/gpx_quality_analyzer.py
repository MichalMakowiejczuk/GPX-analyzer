import pandas as pd


class GpxQualityAnalyzer:
    """Analyze GPX track quality based on point density and spacing."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def analyze(self) -> dict[str, float | int | str]:
        num_points = len(self.df)
        if num_points < 2:
            return {
                "num_points": num_points,
                "avg_spacing_m": 0.0,
                "median_spacing_m": 0.0,
            }

        delta_m = self.df["km"].diff().dropna() * 1000.0
        avg_spacing = float(delta_m.mean())
        median_spacing = float(delta_m.median())

        return {
            "num_points": num_points,
            "avg_spacing_m": round(avg_spacing, 2),
            "median_spacing_m": round(median_spacing, 2),
        }
