"""
Check on a real GPU that cogito enforces the GPU limits of its predictor workers.

Run it on a host with an NVIDIA GPU (Volta or newer), CUDA and torch installed,
from a checkout with cogito installed (pip install -e .):

    python scripts/validate_gpu_limits.py              # without CUDA MPS
    nvidia-cuda-mps-control -d                         # start the MPS daemon
    python scripts/validate_gpu_limits.py              # with CUDA MPS
    echo quit | nvidia-cuda-mps-control                # stop it

It runs on a single GPU, --device (an index or a UUID, as nvidia-smi names it;
0 by default), whatever CUDA_VISIBLE_DEVICES was. It detects MPS the way cogito
does and checks:

- memory: a worker capped at --cap-mb fails with PredictorOutOfMemoryError beyond
  its cap while an uncapped sibling keeps allocating. With MPS the cap holds even
  after the predictor lifts PyTorch's own cap (driver-level, hard); without MPS
  it does not (PyTorch only, soft).
- compute (MPS only): a worker given a 0.25 compute share is clearly slower at a
  matmul than one given the whole GPU: the median of interleaved measurements is
  at most 0.75 of it. Not 0.25: with fewer SMs active, a power- or clock-bound
  GPU runs each faster (an L4 measured 0.45).
- degraded mode: without MPS, cogito logs its degraded-mode warning once.

It prints a JSON report and exits 1 if any check fails.
"""

import argparse
import asyncio
import csv
import io
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time

# This file is also the module of the probe predictor the workers load.
PROBE = f"{os.path.splitext(os.path.basename(__file__))[0]}:Probe"
MB = 1024 * 1024
COMPUTE_SAMPLES = 5
# A 0.25 share must leave the worker clearly slower than the whole GPU.
COMPUTE_RATIO_THRESHOLD = 0.75
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "COGITO_GPU_MEMORY_LIMIT",
    "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
)


class Probe:
    def setup(self):
        import torch

        torch.cuda.init()

    def predict(
        self,
        action: str,
        mb: int = 0,
        size: int = 8192,
        runs: int = 20,
        warmup: int = 5,
    ):
        import torch

        if action == "env":
            return {key: os.environ.get(key) for key in ENV_KEYS}
        if action == "alloc":
            block = torch.empty(mb * MB, dtype=torch.uint8, device="cuda")
            del block
            torch.cuda.empty_cache()
            return {"allocated_mb": mb}
        if action == "unbounded_alloc":
            # Lift PyTorch's own cap, so only MPS can stop the allocation.
            for device in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(1.0, device)
            block = torch.empty(mb * MB, dtype=torch.uint8, device="cuda")
            del block
            torch.cuda.empty_cache()
            return {"allocated_mb": mb}
        if action == "matmul":
            a = torch.randn(size, size, device="cuda", dtype=torch.float16)
            b = torch.randn(size, size, device="cuda", dtype=torch.float16)
            for _ in range(warmup):
                torch.matmul(a, b)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(runs):
                torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            return {"tflops": round(2 * size**3 * runs / elapsed / 1e12, 2)}
        raise ValueError(f"Unknown action {action}")


class _Captured(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _predict(worker, **inputs):
    """The result of a prediction, or the name and message of its error."""
    try:
        return asyncio.run(worker.predict(inputs, timeout=300))
    except Exception as error:
        return {"error": type(error).__name__, "message": str(error)[:300]}


def _oom(result):
    return result.get("error") == "PredictorOutOfMemoryError"


def check_memory(mps, cap_mb):
    from cogito.core.worker import WorkerSpec, WorkerSupervisor

    supervisor = WorkerSupervisor(
        [
            WorkerSpec(name="capped", class_path=PROBE, gpu_memory_limit=cap_mb * MB),
            WorkerSpec(name="sibling", class_path=PROBE),
        ]
    )
    captured = _Captured()
    logging.getLogger("cogito.core.worker").addHandler(captured)
    supervisor.start()
    try:
        capped, sibling = supervisor["capped"], supervisor["sibling"]
        env = _predict(capped, action="env")
        within = _predict(capped, action="alloc", mb=cap_mb // 2)
        beyond = _predict(capped, action="alloc", mb=cap_mb + cap_mb // 2)
        # With its cap still in place: unbounded_alloc lifts PyTorch's.
        after = _predict(capped, action="alloc", mb=cap_mb // 2)
        sibling_beyond = _predict(sibling, action="alloc", mb=cap_mb + cap_mb // 2)
        unbounded = _predict(capped, action="unbounded_alloc", mb=cap_mb + cap_mb // 2)
        alive = capped.is_available() and sibling.is_available()
    finally:
        supervisor.stop()
        logging.getLogger("cogito.core.worker").removeHandler(captured)

    warnings = [
        record.getMessage()
        for record in captured.records
        if "degraded mode" in record.getMessage()
    ]
    checks = {
        "cap env set": env.get("COGITO_GPU_MEMORY_LIMIT") == str(cap_mb * MB),
        "allocation within the cap succeeds": "error" not in within,
        "allocation beyond the cap is a PredictorOutOfMemoryError": _oom(beyond),
        "uncapped sibling allocates the same amount": "error" not in sibling_beyond,
        "capped worker still serves after its OOM": "error" not in after,
        "both workers stay available": alive,
    }
    if mps:
        checks["MPS pinned memory limit set"] = (
            env.get("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT") == f"0={cap_mb}MB"
        )
        checks["cap holds with PyTorch's cap lifted (hard)"] = _oom(unbounded)
        checks["no degraded-mode warning"] = not warnings
    else:
        checks["no MPS pinned memory limit"] = (
            env.get("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT") is None
        )
        checks["cap gone with PyTorch's cap lifted (soft)"] = "error" not in unbounded
        checks["one degraded-mode warning"] = len(warnings) == 1
    return {
        "env": env,
        "results": {
            "within": within,
            "beyond": beyond,
            "sibling_beyond": sibling_beyond,
            "unbounded": unbounded,
            "after": after,
        },
        "warnings": warnings,
        "checks": checks,
    }


def compare_throughput(quarter, full, threshold=COMPUTE_RATIO_THRESHOLD):
    """
    Compare the TFLOPS samples of the 0.25 and the whole-GPU workers by their
    medians. The ratio is None when either worker has no sample.
    """
    result = {"quarter": quarter, "full": full, "threshold": threshold}
    if not quarter or not full:
        return {**result, "ratio": None, "passed": False}
    ratio = round(statistics.median(quarter) / statistics.median(full), 2)
    return {**result, "ratio": ratio, "passed": ratio <= threshold}


def check_compute():
    from cogito.core.worker import WorkerSpec, WorkerSupervisor

    supervisor = WorkerSupervisor(
        [
            WorkerSpec(name="quarter", class_path=PROBE, gpu_compute=0.25),
            WorkerSpec(name="full", class_path=PROBE, gpu_compute=1.0),
        ]
    )
    supervisor.start()
    try:
        quarter, full = supervisor["quarter"], supervisor["full"]
        quarter_env = _predict(quarter, action="env")
        full_env = _predict(full, action="env")
        # One at a time: the share caps a worker even when it has the GPU alone.
        # Interleaved, so a change in clocks or temperature hits both workers.
        samples = {"quarter": [], "full": []}
        errors = []
        for _ in range(COMPUTE_SAMPLES):
            for name, worker in (("quarter", quarter), ("full", full)):
                speed = _predict(worker, action="matmul")
                if "tflops" in speed:
                    samples[name].append(speed["tflops"])
                else:
                    errors.append({name: speed})
    finally:
        supervisor.stop()

    throughput = compare_throughput(samples["quarter"], samples["full"])
    if errors:
        throughput = {**throughput, "ratio": None, "passed": False, "errors": errors}
    return {
        "env": {"quarter": quarter_env, "full": full_env},
        "results": throughput,
        "checks": {
            "active thread percentage 25 for 0.25": quarter_env.get(
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
            )
            == "25",
            "active thread percentage 100 for 1.0": full_env.get(
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
            )
            == "100",
            f"0.25 share is at most {COMPUTE_RATIO_THRESHOLD} as fast (medians)": (
                throughput["passed"]
            ),
        },
    }


def _gpu(device):
    """The GPU, read without creating a CUDA context in this process."""
    from importlib.metadata import version

    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    [[name, memory, driver]] = csv.reader(io.StringIO(output), skipinitialspace=True)
    return {"name": name, "memory": memory, "driver": driver, "torch": version("torch")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument(
        "--device",
        default="0",
        help="the GPU to test, an index or a UUID as nvidia-smi names it",
    )
    parser.add_argument(
        "--cap-mb", type=int, default=2048, help="memory cap of the capped worker"
    )
    args = parser.parse_args()

    from cogito.gpu import mps_running

    # One GPU, the one the report describes: the checks expect a single device.
    # PCI bus order numbers the GPUs as nvidia-smi does.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    mps = mps_running()
    report = {
        "python": platform.python_version(),
        "gpu": _gpu(args.device),
        "mps": mps,
        "memory": check_memory(mps, args.cap_mb),
    }
    if mps:
        report["compute"] = check_compute()
    print(json.dumps(report, indent=2))

    failed = [
        f"{scenario}: {name}"
        for scenario in ("memory", "compute")
        for name, passed in report.get(scenario, {}).get("checks", {}).items()
        if not passed
    ]
    for line in failed:
        print(f"FAIL {line}", file=sys.stderr)
    print("PASS" if not failed else "FAIL", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
