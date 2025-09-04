from typing import Tuple

DEFAULT_THRESHOLDS: Tuple[float, ...] = (2, 4, 6, 8)


def validate_slope_thresholds(threshold_str: str) -> Tuple[float, ...]:
    """
    Validates and parses a comma-separated string of slope thresholds.
    Ensures there are exactly four numeric values in increasing order.
    """
    try:
        # Parsing
        thresholds = tuple(
            float(x.strip()) for x in threshold_str.split(",") if x.strip() != ""
        )
        # Length check
        if len(thresholds) != 4:
            return DEFAULT_THRESHOLDS
        # Increasing check
        if list(thresholds) != sorted(thresholds):
            return DEFAULT_THRESHOLDS
        return thresholds
    except (ValueError, TypeError):
        # Error handling
        return DEFAULT_THRESHOLDS
