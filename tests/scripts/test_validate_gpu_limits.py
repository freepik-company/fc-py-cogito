import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "validate_gpu_limits.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("validate_gpu_limits", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_throughput_uses_the_medians(script):
    # An outlier in each worker does not move its median.
    result = script.compare_throughput([24, 25, 99, 23, 24], [54, 1, 53, 55, 54])

    assert result["ratio"] == 0.44
    assert result["threshold"] == 0.75
    assert result["passed"]


def test_compare_throughput_passes_at_the_threshold(script):
    assert script.compare_throughput([30, 30, 30], [40, 40, 40])["passed"]


def test_compare_throughput_fails_above_the_threshold(script):
    result = script.compare_throughput([38, 39, 40], [50, 50, 50])

    assert result["ratio"] == 0.78
    assert not result["passed"]


@pytest.mark.parametrize("quarter, full", [([], [50]), ([20], []), ([], [])])
def test_compare_throughput_fails_without_samples(script, quarter, full):
    result = script.compare_throughput(quarter, full)

    assert result["ratio"] is None
    assert not result["passed"]
