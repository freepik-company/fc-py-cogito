"""
GPU memory cap and compute share of a predictor worker.

Each worker can be given a cap on the device memory it may allocate, in bytes per
visible device (WorkerSpec.gpu_memory_limit). It is enforced at two levels:

- With CUDA MPS running, the worker starts with CUDA_MPS_PINNED_DEVICE_MEM_LIMIT.
  The driver enforces it for every library in the process (PyTorch, cuDNN,
  TensorRT, JAX, custom kernels): an allocation beyond the cap fails in that
  worker only. This is the hard limit.
- In the worker, before the predictor is loaded, apply_memory_limit() caps the
  PyTorch caching allocator with torch.cuda.set_per_process_memory_fraction when
  torch is installed, so PyTorch fails early with a clear out-of-memory error.
  Without MPS this is the only limit, and a soft one: it binds PyTorch's
  allocator alone, not other libraries.

Other frameworks have their own knob, set through WorkerSpec.env, e.g. JAX:
XLA_PYTHON_CLIENT_MEM_FRACTION (a fraction of the device memory).

A worker can also be given a share of each GPU's streaming multiprocessors
(WorkerSpec.gpu_compute). Only CUDA MPS can enforce it, through
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE; without MPS there is no compute limit.

mps_running() tells whether an MPS control daemon is reachable from this process.

cogito never depends on torch: without it apply_memory_limit() does nothing.
"""

import logging
import os
import stat
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

# Cap in bytes per visible device, read by apply_memory_limit() in the worker.
GPU_MEMORY_LIMIT_ENV = "COGITO_GPU_MEMORY_LIMIT"
MPS_PINNED_DEVICE_MEM_LIMIT_ENV = "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"
MPS_ACTIVE_THREAD_PERCENTAGE_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
MPS_PIPE_DIRECTORY_ENV = "CUDA_MPS_PIPE_DIRECTORY"
DEFAULT_MPS_PIPE_DIRECTORY = "/tmp/nvidia-mps"

MB = 1024 * 1024

_logger = logging.getLogger(__name__)


def mps_running(pipe_directory: Optional[str] = None) -> bool:
    """
    Whether a CUDA MPS control daemon serves clients using pipe_directory.

    pipe_directory defaults to CUDA_MPS_PIPE_DIRECTORY, or /tmp/nvidia-mps when it
    is unset, the same directory a CUDA client looks in. The control daemon
    listens on the "control" pipe there; a regular file by that name does not
    count. The pipe is never opened, so the daemon sees nothing. A pipe left
    behind by a daemon that was killed still counts as running.
    It never raises: anything that cannot be checked means no MPS.
    """
    if pipe_directory is None:
        pipe_directory = (
            os.environ.get(MPS_PIPE_DIRECTORY_ENV) or DEFAULT_MPS_PIPE_DIRECTORY
        )
    try:
        mode = os.stat(os.path.join(pipe_directory, "control")).st_mode
    except (OSError, ValueError):
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)


def format_active_thread_percentage(compute: float) -> str:
    """
    Value of CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for a compute share in (0, 1].

    The share is rounded half up to a whole percentage, and never below 1, e.g.
    "50" for 0.5 and "13" for 0.125. It is rounded from its decimal form, so
    0.145 gives "15" even though the float is slightly below it.
    """
    percentage = (Decimal(str(compute)) * 100).quantize(
        Decimal(1), rounding=ROUND_HALF_UP
    )
    return str(max(1, int(percentage)))


def format_pinned_device_mem_limit(devices: str, limit_bytes: int) -> str:
    """
    Value of CUDA_MPS_PINNED_DEVICE_MEM_LIMIT capping each of devices at limit_bytes.

    devices is the worker's CUDA_VISIBLE_DEVICES. The client reads the ordinals of
    CUDA_MPS_PINNED_DEVICE_MEM_LIMIT within the devices it sees, and an entry for
    a device it cannot see drops every limit, so each visible device is named by
    its ordinal inside the worker, e.g. "0=1024MB,1=1024MB" for "3,5". Ordinals
    also hold for a UUID, which CUDA_VISIBLE_DEVICES accepts as a prefix.
    CUDA stops reading CUDA_VISIBLE_DEVICES at its first invalid entry, so the
    count stops at the first empty one or negative index (e.g. "0,-1"); every
    device listed before it must exist, or the worker sees fewer devices than
    there are entries.
    The limit is rounded down to whole megabytes.
    """
    limit = f"{limit_bytes // MB}MB"
    count = 0
    for device in devices.split(","):
        device = device.strip()
        if not device or device.startswith("-"):
            break
        count += 1
    return ",".join(f"{ordinal}={limit}" for ordinal in range(count))


def apply_memory_limit(limit_bytes: Optional[int] = None) -> bool:
    """
    Cap the PyTorch caching allocator at limit_bytes on every visible device.

    limit_bytes defaults to COGITO_GPU_MEMORY_LIMIT. Workers call it before loading
    their predictor; an in-process predictor may call it at the top of setup().
    Returns whether the cap was applied. It never raises: without torch, without
    CUDA or without a limit there is nothing to do, and a failure is logged.
    """
    if limit_bytes is None:
        value = os.environ.get(GPU_MEMORY_LIMIT_ENV)
        if not value:
            return False
        try:
            limit_bytes = int(value)
        except ValueError:
            _logger.warning("Ignoring invalid %s=%r", GPU_MEMORY_LIMIT_ENV, value)
            return False
    if limit_bytes <= 0:
        _logger.warning("Ignoring non-positive GPU memory limit %r", limit_bytes)
        return False

    try:
        import torch
    except ImportError:
        return False
    except Exception:
        # An installed but broken torch (e.g. a missing shared library) must not
        # stop a predictor that may not even use it.
        _logger.warning("Unable to import torch", exc_info=True)
        return False

    try:
        if not torch.cuda.is_available():
            return False
        for device in range(torch.cuda.device_count()):
            # The physical total, not the one MPS may already cap: the fraction
            # must give the same limit in bytes with or without MPS.
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, limit_bytes / total), device
            )
    except Exception:
        _logger.warning(
            "Unable to cap the PyTorch allocator at %d bytes",
            limit_bytes,
            exc_info=True,
        )
        return False
    return True
