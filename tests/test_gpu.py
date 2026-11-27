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
    GPUDevice,
    GPUInventory,
    apply_memory_limit,
    describe_memory_cap,
    format_active_thread_percentage,
    format_pinned_device_mem_limit,
    gpu_inventory,
    is_cuda_out_of_memory,
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


def torch_error(module, name="OutOfMemoryError"):
    """A class standing in for one of torch's, without importing torch."""
    return type(name, (RuntimeError,), {"__module__": module})


@pytest.mark.parametrize(
    "error",
    [
        torch_error("torch")("Allocation failed"),
        torch_error("torch.cuda")("Allocation failed"),
        type("Custom", (torch_error("torch.cuda"),), {})("Allocation failed"),
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("CUDA error: out of memory\nCUDA kernel errors might be ..."),
    ],
)
def test_is_cuda_out_of_memory(error):
    assert is_cuda_out_of_memory(error)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("model exploded"),
        MemoryError("out of memory"),
        torch_error("not_torch")("Allocation failed"),
        torch_error("torchvision")("Allocation failed"),
        torch_error("torch", name="AcceleratorError")("device-side assert"),
    ],
)
def test_is_not_cuda_out_of_memory(error):
    assert not is_cuda_out_of_memory(error)


@pytest.mark.parametrize("raised", [RuntimeError, SystemExit, KeyboardInterrupt])
def test_is_cuda_out_of_memory_never_raises(raised):
    class BrokenStr(Exception):
        def __str__(self):
            raise raised("cannot render")

    assert not is_cuda_out_of_memory(BrokenStr())


def test_describe_memory_cap():
    assert describe_memory_cap(1536 * 1024**2 + 1) == "cap: 1536MB per device"
    assert describe_memory_cap(None) == "no GPU memory cap set"


class FakeNvml(types.ModuleType):
    """A fake pynvml module serving the given (uuid, name, total memory) devices."""

    def __init__(
        self,
        devices,
        init_error=None,
        count_error=None,
        failing=(),
        as_bytes=False,
    ):
        super().__init__("pynvml")
        self.devices = devices
        self.init_error = init_error
        self.count_error = count_error
        # Indices whose handle cannot be read, e.g. NVML_ERROR_NO_PERMISSION.
        self.failing = failing
        self.as_bytes = as_bytes
        self.shutdowns = 0

    def nvmlInit(self):
        if self.init_error:
            raise self.init_error

    def nvmlShutdown(self):
        self.shutdowns += 1

    def nvmlDeviceGetCount(self):
        if self.count_error:
            raise self.count_error
        return len(self.devices)

    def nvmlDeviceGetHandleByIndex(self, index):
        if index in self.failing:
            raise RuntimeError("Insufficient Permissions")
        return index

    def _string(self, value):
        return value.encode() if self.as_bytes else value

    def nvmlDeviceGetUUID(self, handle):
        return self._string(self.devices[handle][0])

    def nvmlDeviceGetName(self, handle):
        return self._string(self.devices[handle][1])

    def nvmlDeviceGetMemoryInfo(self, handle):
        return types.SimpleNamespace(total=self.devices[handle][2])


THREE_GPUS = [
    ("GPU-aaaa-0000", "NVIDIA A100-SXM4-80GB", 80 * GB),
    ("GPU-bbbb-1111", "NVIDIA A100-SXM4-80GB", 80 * GB),
    ("GPU-bbbc-2222", "NVIDIA L4", 24 * GB),
]


@pytest.fixture
def fake_nvml(monkeypatch):
    """
    Install a fake pynvml module, unset CUDA_VISIBLE_DEVICES and order CUDA devices
    by PCI bus, as workers are.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    def install(devices=THREE_GPUS, **kwargs):
        nvml = FakeNvml(devices, **kwargs)
        monkeypatch.setitem(sys.modules, "pynvml", nvml)
        return nvml

    return install


def _indices(inventory):
    return [device.index for device in inventory.devices]


def test_gpu_inventory_lists_every_device(fake_nvml):
    nvml = fake_nvml()

    inventory = gpu_inventory()

    assert inventory == GPUInventory(
        devices=(
            GPUDevice(0, "GPU-aaaa-0000", "NVIDIA A100-SXM4-80GB", 80 * GB),
            GPUDevice(1, "GPU-bbbb-1111", "NVIDIA A100-SXM4-80GB", 80 * GB),
            GPUDevice(2, "GPU-bbbc-2222", "NVIDIA L4", 24 * GB),
        )
    )
    assert nvml.shutdowns == 1


def test_gpu_inventory_decodes_bytes_from_older_bindings(fake_nvml):
    fake_nvml(as_bytes=True)

    device = gpu_inventory().devices[0]

    assert (device.uuid, device.name) == ("GPU-aaaa-0000", "NVIDIA A100-SXM4-80GB")


def test_gpu_inventory_follows_cuda_visible_devices_from_env(fake_nvml, monkeypatch):
    fake_nvml()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")

    assert _indices(gpu_inventory()) == [2, 0]


def test_gpu_inventory_argument_overrides_env(fake_nvml, monkeypatch):
    fake_nvml()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    assert _indices(gpu_inventory("1")) == [1]


def test_gpu_inventory_is_empty_when_every_gpu_is_hidden(fake_nvml, monkeypatch):
    fake_nvml()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    assert gpu_inventory() == GPUInventory()


@pytest.mark.parametrize(
    "devices, expected",
    [
        ("GPU-bbbc-2222", [2]),
        ("GPU-aaaa, 1", [0, 1]),
        # CUDA stops at the first invalid entry.
        ("0,,1", [0]),
        ("0,-1,2", [0]),
        ("0,7,1", [0]),
        ("0,GPU-ffff,1", [0]),
        # Not documented by CUDA: never guess, never count a device twice.
        ("0,GPU-bbb,2", [0]),
        ("1,1,2", [1]),
        ("1,GPU-bbbb-1111,2", [1]),
        # MIG instances are not listed.
        ("0,MIG-GPU-aaaa-0000/1/0,1", [0]),
        # Only ASCII digits make an index.
        ("0,\u00b2,1", [0]),
        ("0,\u0663,1", [0]),
    ],
)
def test_gpu_inventory_reads_cuda_visible_devices_like_cuda(
    fake_nvml, devices, expected
):
    fake_nvml()

    assert _indices(gpu_inventory(devices)) == expected


def test_gpu_inventory_is_empty_without_pynvml(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # A None entry makes "import pynvml" raise ImportError.
    monkeypatch.setitem(sys.modules, "pynvml", None)

    assert gpu_inventory() == GPUInventory()


def test_gpu_inventory_is_empty_when_nvml_is_unavailable(fake_nvml):
    nvml = fake_nvml(init_error=RuntimeError("NVML Shared Library Not Found"))

    assert gpu_inventory() == GPUInventory()
    assert nvml.shutdowns == 0


def test_gpu_inventory_is_empty_and_shuts_nvml_down_when_counting_fails(fake_nvml):
    nvml = fake_nvml(count_error=RuntimeError("Unknown Error"))

    assert gpu_inventory() == GPUInventory()
    assert nvml.shutdowns == 1


def test_gpu_inventory_leaves_out_only_a_gpu_nvml_cannot_read(fake_nvml, caplog):
    nvml = fake_nvml(failing={1})

    with caplog.at_level(logging.WARNING, logger="cogito.gpu"):
        inventory = gpu_inventory()

    assert _indices(inventory) == [0, 2]
    assert "Leaving GPU 1 out of the inventory" in caplog.text
    assert nvml.shutdowns == 1


def test_gpu_inventory_logs_the_entry_that_ends_the_list(fake_nvml, caplog):
    fake_nvml()

    with caplog.at_level(logging.WARNING, logger="cogito.gpu"):
        gpu_inventory("0,GPU-ffff,1")

    assert "'GPU-ffff'" in caplog.text
    assert "GPU-ffff,1" in caplog.text


@pytest.mark.parametrize("devices", ["0,-1,1", "0,,1", "0,1"])
def test_gpu_inventory_does_not_log_a_list_cuda_reads_the_same_way(
    fake_nvml, caplog, devices
):
    fake_nvml()

    with caplog.at_level(logging.WARNING, logger="cogito.gpu"):
        gpu_inventory(devices)

    assert caplog.text == ""


def test_gpu_inventory_warns_about_indices_without_pci_bus_order(
    fake_nvml, monkeypatch, caplog
):
    fake_nvml()
    monkeypatch.delenv("CUDA_DEVICE_ORDER")

    with caplog.at_level(logging.WARNING, logger="cogito.gpu"):
        assert _indices(gpu_inventory("1")) == [1]

    assert "CUDA_DEVICE_ORDER is not PCI_BUS_ID" in caplog.text
