import importlib
import multiprocessing
import os
import sys
import traceback
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from typing import Any, Dict, Mapping, Optional

from cogito.core.exceptions import WorkerStartError

# Workers always use the spawn start method: a forked child would inherit any CUDA
# context the parent created, and the env must be in place before the predictor
# module (and the libraries it imports) is loaded.
_MP_CONTEXT = multiprocessing.get_context("spawn")

STOP_MESSAGE = "stop"


@dataclass(frozen=True)
class WorkerSpec:
    """
    What a worker process runs and the environment it runs with.

    cuda_visible_devices is the final value for CUDA_VISIBLE_DEVICES: None inherits
    the parent's value, an empty string hides every GPU (CPU-only placement).
    env holds extra resource variables, applied after the ones cogito sets.
    """

    name: str
    class_path: str
    cuda_visible_devices: Optional[str] = None
    env: Mapping[str, str] = field(default_factory=dict)


def build_worker_env(spec: WorkerSpec) -> Dict[str, str]:
    """
    Environment variables a worker applies before importing its predictor
    """
    env = {
        "COGITO_PREDICTOR_NAME": spec.name,
        # Match the device ordering NVML reports, so allocated indices point to
        # the same physical GPUs inside the worker.
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    if spec.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices
    env.update(spec.env)
    return env


def _load_predictor(class_path: str) -> Any:
    path, class_name = class_path.split(":")
    module = importlib.import_module(path)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class {class_name} not found in module {path}")
    return getattr(module, class_name)()


def _worker_main(spec: WorkerSpec, env: Dict[str, str], conn: Connection) -> None:
    """
    Entry point of the worker process
    """
    # The env must be set before the predictor module is imported.
    os.environ.update(env)

    try:
        predictor = _load_predictor(spec.class_path)
    except BaseException:
        conn.send(
            {
                "status": "error",
                "pid": os.getpid(),
                "error": traceback.format_exc(),
            }
        )
        conn.close()
        sys.exit(1)

    conn.send(
        {
            "status": "ready",
            "pid": os.getpid(),
            "env": {key: os.environ.get(key) for key in env},
            "predictor": f"{type(predictor).__module__}.{type(predictor).__qualname__}",
        }
    )

    # Wait until the parent asks the worker to stop or goes away.
    while True:
        try:
            message = conn.recv()
        except EOFError:
            break
        if message == STOP_MESSAGE:
            break

    conn.close()


class PredictorWorker:
    """
    Handle to a running worker process
    """

    def __init__(
        self,
        spec: WorkerSpec,
        env: Dict[str, str],
        process: multiprocessing.process.BaseProcess,
        conn: Connection,
        handshake: Dict[str, Any],
    ):
        self.spec = spec
        self.env = env
        self.process = process
        self.conn = conn
        self.handshake = handshake

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def stop(self, timeout: float = 10.0) -> None:
        """
        Ask the worker to stop, and terminate it if it does not exit in time
        """
        if self.process.is_alive():
            try:
                self.conn.send(STOP_MESSAGE)
            except (BrokenPipeError, OSError):
                pass
        _reap(self.process, timeout)
        self.conn.close()


def _reap(process: multiprocessing.process.BaseProcess, timeout: float) -> None:
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join()


def spawn_worker(spec: WorkerSpec, start_timeout: float = 120.0) -> PredictorWorker:
    """
    Start a worker process for a predictor and wait until the predictor is loaded.

    The predictor is imported only inside the worker, after its env is applied.
    Raises WorkerStartError if the predictor cannot be loaded in time.
    """
    env = build_worker_env(spec)
    # The spec is pickled to reach the worker: copy env into a plain dict, since
    # some Mapping types (e.g. MappingProxyType) cannot be pickled.
    spec = replace(spec, env=dict(spec.env))
    parent_conn, child_conn = _MP_CONTEXT.Pipe()
    process = _MP_CONTEXT.Process(
        target=_worker_main,
        args=(spec, env, child_conn),
        name=f"cogito-worker-{spec.name}",
    )
    process.start()
    # Only the worker keeps its end open, so its exit shows up as EOF here.
    child_conn.close()

    try:
        if not parent_conn.poll(start_timeout):
            raise WorkerStartError(
                spec.name, f"predictor not loaded after {start_timeout}s"
            )
        handshake = parent_conn.recv()
    except EOFError:
        _reap(process, timeout=10.0)
        parent_conn.close()
        raise WorkerStartError(
            spec.name, f"worker exited with code {process.exitcode}"
        ) from None
    except BaseException:
        _reap(process, timeout=0)
        parent_conn.close()
        raise

    if handshake.get("status") != "ready":
        _reap(process, timeout=10.0)
        parent_conn.close()
        raise WorkerStartError(spec.name, handshake.get("error", "unknown error"))

    return PredictorWorker(spec, env, process, parent_conn, handshake)
