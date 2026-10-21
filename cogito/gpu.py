"""
GPU memory cap of a predictor worker.

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

cogito never depends on torch: without it apply_memory_limit() does nothing.
"""

import logging
import os
from typing import Optional

# Cap in bytes per visible device, read by apply_memory_limit() in the worker.
GPU_MEMORY_LIMIT_ENV = "COGITO_GPU_MEMORY_LIMIT"
MPS_PINNED_DEVICE_MEM_LIMIT_ENV = "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"

MB = 1024 * 1024

_logger = logging.getLogger(__name__)


def format_pinned_device_mem_limit(devices: str, limit_bytes: int) -> str:
    """
    Value of CUDA_MPS_PINNED_DEVICE_MEM_LIMIT capping each of devices at limit_bytes.

    devices is a CUDA_VISIBLE_DEVICES value: the variable takes the same device
    ordinals or UUIDs, each with its limit after an equals, e.g. "0=1024MB".
    The limit is rounded down to whole megabytes.
    """
    limit = f"{limit_bytes // MB}MB"
    return ",".join(
        f"{device}={limit}"
        for device in (device.strip() for device in devices.split(","))
        if device
    )


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
