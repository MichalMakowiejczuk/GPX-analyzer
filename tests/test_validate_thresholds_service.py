import pytest

from services.validate_thresholds_service import (
    DEFAULT_THRESHOLDS,
    validate_slope_thresholds,
)


def test_valid_input():
    inp = "1,2,3,4"
    assert validate_slope_thresholds(inp) == (1, 2, 3, 4)


def test_non_numeric_input():
    inp = "2(),8"
    assert validate_slope_thresholds(inp) == DEFAULT_THRESHOLDS


def test_too_few_numbers():
    inp = "1,2,3"
    assert validate_slope_thresholds(inp) == DEFAULT_THRESHOLDS


def test_not_increasing():
    inp = "4,3,5,6"
    assert validate_slope_thresholds(inp) == DEFAULT_THRESHOLDS


def test_empty_string():
    inp = ""
    assert validate_slope_thresholds(inp) == DEFAULT_THRESHOLDS
