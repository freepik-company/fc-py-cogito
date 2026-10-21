import logging
import os
import socket
import sys
import tempfile
import types

import pytest

from cogito import gpu
from cogito.gpu import (
    GPU_MEMORY_LIMIT_ENV,
    apply_memory_limit,
    format_active_thread_percentage,
    format_pinned_device_mem_limit,
    mps_running,
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


def test_mps_running_with_the_control_pipe(tmp_path):
    os.mkfifo(tmp_path / "control")

    assert mps_running(str(tmp_path)) is True


def test_mps_running_with_a_control_socket():
    # A short directory: a UNIX socket path has a small length limit.
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(os.path.join(directory, "control"))

            assert mps_running(directory) is True
        finally:
            server.close()


def test_mps_running_without_the_control_pipe(tmp_path):
    assert mps_running(str(tmp_path)) is False


def test_mps_running_ignores_a_regular_control_file(tmp_path):
    (tmp_path / "control").write_text("")

    assert mps_running(str(tmp_path)) is False


def test_mps_running_without_the_pipe_directory(tmp_path):
    assert mps_running(str(tmp_path / "missing")) is False


def test_mps_running_reads_the_pipe_directory_from_the_env(tmp_path, monkeypatch):
    os.mkfifo(tmp_path / "control")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path))

    assert mps_running() is True


def test_mps_running_defaults_to_the_nvidia_pipe_directory(monkeypatch):
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    checked = []

    def fake_stat(path):
        checked.append(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(gpu.os, "stat", fake_stat)

    assert mps_running() is False
    assert checked == ["/tmp/nvidia-mps/control"]


def test_mps_running_never_raises(monkeypatch):
    def fake_stat(path):
        raise PermissionError(path)

    monkeypatch.setattr(gpu.os, "stat", fake_stat)

    assert mps_running("/somewhere") is False


@pytest.mark.parametrize(
    "compute, percentage",
    [
        (1.0, "100"),
        (0.5, "50"),
        (0.29, "29"),
        (0.125, "13"),
        (0.145, "15"),
        (0.285, "29"),
        (0.575, "58"),
        (0.004, "1"),
        (0.001, "1"),
    ],
)
def test_format_active_thread_percentage(compute, percentage):
    assert format_active_thread_percentage(compute) == percentage
