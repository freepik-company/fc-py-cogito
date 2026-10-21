import logging
import sys
import types

import pytest

from cogito.gpu import (
    GPU_MEMORY_LIMIT_ENV,
    apply_memory_limit,
    format_pinned_device_mem_limit,
)

GB = 1024**3


class FakeCuda:
    def __init__(self, totals, available=True, fail=False):
        self.totals = totals
        self.available = available
        self.fail = fail
        self.fractions = {}

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.totals)

    def get_device_properties(self, device):
        return types.SimpleNamespace(total_memory=self.totals[device])

    def set_per_process_memory_fraction(self, fraction, device):
        if self.fail:
            raise RuntimeError("CUDA error: no CUDA-capable device is detected")
        self.fractions[device] = fraction


@pytest.fixture
def fake_torch(monkeypatch):
    """Install a fake torch module and return a function that sets its devices."""

    def install(totals, **kwargs):
        cuda = FakeCuda(totals, **kwargs)
        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
        return cuda

    return install


def test_format_pinned_device_mem_limit_names_devices_by_worker_ordinal():
    # Physical GPUs 3 and 5 are devices 0 and 1 inside the worker.
    assert format_pinned_device_mem_limit("3,5", 4 * GB) == "0=4096MB,1=4096MB"


def test_format_pinned_device_mem_limit_names_uuids_by_worker_ordinal():
    assert (
        format_pinned_device_mem_limit("GPU-8f3a, GPU-12bc", 512 * 1024**2)
        == "0=512MB,1=512MB"
    )


@pytest.mark.parametrize("devices", ["0,,1", "0,-1", "0,-1,2", "0, -1"])
def test_format_pinned_device_mem_limit_stops_where_cuda_stops(devices):
    # CUDA sees only device 0 in each of these: an entry for a second one would
    # make the driver drop every limit.
    assert format_pinned_device_mem_limit(devices, 4 * GB) == "0=4096MB"


def test_format_pinned_device_mem_limit_rounds_down_to_megabytes():
    assert format_pinned_device_mem_limit("1", 1024**2 * 3 + 1023) == "0=3MB"


def test_apply_memory_limit_sets_the_fraction_of_each_device(fake_torch):
    cuda = fake_torch([16 * GB, 8 * GB])

    assert apply_memory_limit(4 * GB) is True

    assert cuda.fractions == {0: 0.25, 1: 0.5}


def test_apply_memory_limit_never_goes_beyond_the_whole_device(fake_torch):
    cuda = fake_torch([8 * GB])

    assert apply_memory_limit(16 * GB) is True

    assert cuda.fractions == {0: 1.0}


def test_apply_memory_limit_reads_the_worker_env(fake_torch, monkeypatch):
    cuda = fake_torch([16 * GB])
    monkeypatch.setenv(GPU_MEMORY_LIMIT_ENV, str(8 * GB))

    assert apply_memory_limit() is True

    assert cuda.fractions == {0: 0.5}


def test_apply_memory_limit_without_a_limit_does_nothing(fake_torch, monkeypatch):
    cuda = fake_torch([16 * GB])
    monkeypatch.delenv(GPU_MEMORY_LIMIT_ENV, raising=False)

    assert apply_memory_limit() is False

    assert cuda.fractions == {}


@pytest.mark.parametrize("value", ["lots", "0", "-1"])
def test_apply_memory_limit_ignores_an_invalid_limit(
    fake_torch, monkeypatch, caplog, value
):
    cuda = fake_torch([16 * GB])
    monkeypatch.setenv(GPU_MEMORY_LIMIT_ENV, value)

    with caplog.at_level(logging.WARNING, logger="cogito.gpu"):
        assert apply_memory_limit() is False

    assert cuda.fractions == {}
    assert caplog.records


def test_apply_memory_limit_without_torch_does_nothing(monkeypatch):
    # None in sys.modules makes `import torch` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)

    assert apply_memory_limit(4 * GB) is False


def test_apply_memory_limit_survives_a_broken_torch_import(
    tmp_path, monkeypatch, caplog
):
    (tmp_path / "torch.py").write_text("raise OSError('libcudart.so: not found')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    with caplog.at_level(logging.WARNING, logger="cogito.gpu"):
        assert apply_memory_limit(4 * GB) is False

    assert "Unable to import torch" in caplog.text


def test_apply_memory_limit_without_cuda_does_nothing(fake_torch):
    cuda = fake_torch([16 * GB], available=False)

    assert apply_memory_limit(4 * GB) is False

    assert cuda.fractions == {}


def test_apply_memory_limit_logs_a_failure_instead_of_raising(fake_torch, caplog):
    fake_torch([16 * GB], fail=True)

    with caplog.at_level(logging.WARNING, logger="cogito.gpu"):
        assert apply_memory_limit(4 * GB) is False

    assert "Unable to cap the PyTorch allocator" in caplog.text
