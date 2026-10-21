import asyncio
import concurrent.futures
import contextlib
import json
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from types import MappingProxyType

import pytest
from pydantic import BaseModel

from cogito.api.handlers import create_health_check_handler

from cogito.core.exceptioin_handlers import (
    worker_timeout_exception_handler,
    worker_unavailable_exception_handler,
)
from cogito.core.exceptions import (
    BadRequestError,
    WorkerPredictionError,
    WorkerStartError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from cogito.core.utils import readiness_context
from cogito.core.worker import (
    RestartPolicy,
    SupervisedWorker,
    WorkerSpec,
    WorkerSupervisor,
    _spawn_environ,
    build_worker_env,
    spawn_worker,
)

TOY_PREDICTOR = textwrap.dedent(
    """
    class Predictor:
        def setup(self):
            pass

        def predict(self):
            return None
    """
)

# Records the env it sees at import time, to prove the worker env is applied before
# the predictor module is imported.
PROBE_PREDICTOR = textwrap.dedent(
    """
    import json
    import os

    with open(os.environ["COGITO_TEST_OUT"], "w") as out:
        json.dump(
            {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER",
                    "COGITO_PREDICTOR_NAME",
                    "COGITO_TEST_RESOURCE",
                )
            },
            out,
        )


    class Predictor:
        def setup(self):
            pass

        def predict(self):
            return None
    """
)


# A stand-in for torch inside the worker: records the cap it is asked to apply.
FAKE_TORCH = textwrap.dedent(
    """
    import json
    import os
    import types


    def _set_per_process_memory_fraction(fraction, device):
        with open(os.environ["COGITO_TEST_TORCH_OUT"], "w") as out:
            json.dump({"fraction": fraction, "device": device}, out)


    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda device: types.SimpleNamespace(
            total_memory=16 * 1024**3
        ),
        set_per_process_memory_fraction=_set_per_process_memory_fraction,
    )
    """
)

# Records, at import time, the memory cap variables and whether torch was capped.
MEMORY_PROBE_PREDICTOR = textwrap.dedent(
    """
    import json
    import os

    with open(os.environ["COGITO_TEST_OUT"], "w") as out:
        json.dump(
            {
                "COGITO_GPU_MEMORY_LIMIT": os.environ.get("COGITO_GPU_MEMORY_LIMIT"),
                "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT": os.environ.get(
                    "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"
                ),
                "torch_capped": os.path.exists(os.environ["COGITO_TEST_TORCH_OUT"]),
            },
            out,
        )


    class Predictor:
        def setup(self):
            pass

        def predict(self):
            return None
    """
)


# A launcher whose main module is re-imported by every spawned worker during its
# bootstrap, before the worker entry point runs.
LAUNCHER = textwrap.dedent(
    """
    import json
    import os
    import sys

    if __name__ == "__mp_main__":
        with open(os.environ["COGITO_TEST_OUT"], "w") as out:
            json.dump({"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}, out)

    from cogito.core.worker import WorkerSpec, spawn_worker

    if __name__ == "__main__":
        worker = spawn_worker(
            WorkerSpec(
                name="toy",
                class_path="cogito_toy_predictor:Predictor",
                cuda_visible_devices="5",
                env={"COGITO_TEST_OUT": sys.argv[1]},
            ),
            start_timeout=60.0,
        )
        worker.stop(timeout=5.0)
    """
)


@pytest.fixture
def predictors_path(tmp_path, monkeypatch):
    """
    Toy predictor modules the parent never imports. Spawned workers inherit the
    parent's sys.path, so they can import them.
    """
    (tmp_path / "cogito_toy_predictor.py").write_text(TOY_PREDICTOR)
    (tmp_path / "cogito_probe_predictor.py").write_text(PROBE_PREDICTOR)
    (tmp_path / "cogito_broken_predictor.py").write_text(
        "raise RuntimeError('broken on import')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


@pytest.fixture
def workers():
    """Stop every spawned worker, even when the test fails."""
    started = []
    yield started
    for worker in started:
        worker.stop(timeout=5.0)


def test_build_worker_env_pins_visible_devices():
    spec = WorkerSpec(name="sdxl", class_path="m:P", cuda_visible_devices="0,2")

    env = build_worker_env(spec)

    assert env == {
        "COGITO_PREDICTOR_NAME": "sdxl",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,2",
    }


def test_build_worker_env_hides_every_gpu_for_cpu_placement():
    spec = WorkerSpec(name="clip", class_path="m:P", cuda_visible_devices="")

    assert build_worker_env(spec)["CUDA_VISIBLE_DEVICES"] == ""


def test_build_worker_env_inherits_visible_devices_when_unset():
    spec = WorkerSpec(name="clip", class_path="m:P")

    assert "CUDA_VISIBLE_DEVICES" not in build_worker_env(spec)


def test_build_worker_env_adds_resource_variables():
    spec = WorkerSpec(
        name="sdxl",
        class_path="m:P",
        cuda_visible_devices="1",
        env={"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50"},
    )

    env = build_worker_env(spec)

    assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "50"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_build_worker_env_caps_gpu_memory_through_mps():
    spec = WorkerSpec(
        name="sdxl",
        class_path="m:P",
        cuda_visible_devices="0,2",
        gpu_memory_limit=4 * 1024**3,
        mps=True,
    )

    env = build_worker_env(spec)

    assert env["COGITO_GPU_MEMORY_LIMIT"] == str(4 * 1024**3)
    assert env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=4096MB,1=4096MB"


def test_build_worker_env_names_uuid_devices_by_worker_ordinal():
    spec = WorkerSpec(
        name="sdxl",
        class_path="m:P",
        cuda_visible_devices="GPU-8f3a",
        gpu_memory_limit=1024**3,
        mps=True,
    )

    assert build_worker_env(spec)["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=1024MB"


def test_build_worker_env_without_mps_only_caps_torch():
    spec = WorkerSpec(
        name="sdxl",
        class_path="m:P",
        cuda_visible_devices="1",
        gpu_memory_limit=1024**3,
    )

    env = build_worker_env(spec)

    assert env["COGITO_GPU_MEMORY_LIMIT"] == str(1024**3)
    assert "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" not in env


def test_build_worker_env_sets_no_mps_cap_when_every_gpu_is_hidden():
    spec = WorkerSpec(
        name="clip",
        class_path="m:P",
        cuda_visible_devices="",
        gpu_memory_limit=1024**3,
        mps=True,
    )

    assert "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" not in build_worker_env(spec)


def test_build_worker_env_caps_the_inherited_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    spec = WorkerSpec(name="sdxl", class_path="m:P", gpu_memory_limit=1024**3, mps=True)

    env = build_worker_env(spec)

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=1024MB,1=1024MB"


def test_build_worker_env_ignores_the_devices_of_a_worker_being_started(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    spec = WorkerSpec(name="sdxl", class_path="m:P", gpu_memory_limit=1024**3, mps=True)
    entered, release = threading.Event(), threading.Event()

    def start_another_worker():
        # What spawning a worker pinned to "0,1" does to this process meanwhile.
        with _spawn_environ({"CUDA_VISIBLE_DEVICES": "0,1"}):
            entered.set()
            release.wait(5.0)

    starter = threading.Thread(target=start_another_worker)
    starter.start()
    assert entered.wait(5.0)
    built = {}
    builder = threading.Thread(target=lambda: built.update(build_worker_env(spec)))
    builder.start()
    builder.join(0.2)
    # It waits for the other worker's start instead of reading its devices.
    assert builder.is_alive()
    release.set()
    starter.join(5.0)
    builder.join(5.0)

    assert built["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=1024MB"


def test_build_worker_env_warns_when_the_mps_cap_has_no_device_list(
    monkeypatch, caplog
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    spec = WorkerSpec(name="sdxl", class_path="m:P", gpu_memory_limit=1024**3, mps=True)

    with caplog.at_level(logging.WARNING, logger="cogito.core.worker"):
        env = build_worker_env(spec)

    assert "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" not in env
    assert env["COGITO_GPU_MEMORY_LIMIT"] == str(1024**3)
    assert "No MPS memory cap for worker sdxl" in caplog.text


def test_build_worker_env_sets_no_cap_without_a_limit():
    spec = WorkerSpec(name="sdxl", class_path="m:P", cuda_visible_devices="0", mps=True)

    env = build_worker_env(spec)

    assert "COGITO_GPU_MEMORY_LIMIT" not in env
    assert "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" not in env


def test_build_worker_env_lets_resource_variables_override_the_mps_cap():
    spec = WorkerSpec(
        name="sdxl",
        class_path="m:P",
        cuda_visible_devices="0",
        gpu_memory_limit=1024**3,
        mps=True,
        env={"CUDA_MPS_PINNED_DEVICE_MEM_LIMIT": "0=512MB"},
    )

    assert build_worker_env(spec)["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=512MB"


def test_build_worker_env_caps_the_devices_set_through_resource_variables():
    spec = WorkerSpec(
        name="sdxl",
        class_path="m:P",
        cuda_visible_devices="GPU-aaaa",
        gpu_memory_limit=1024**3,
        mps=True,
        env={"CUDA_VISIBLE_DEVICES": "GPU-bbbb,GPU-cccc"},
    )

    env = build_worker_env(spec)

    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-bbbb,GPU-cccc"
    assert env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=1024MB,1=1024MB"


def test_build_worker_env_sets_no_mps_cap_when_resource_variables_hide_the_gpus():
    spec = WorkerSpec(
        name="sdxl",
        class_path="m:P",
        cuda_visible_devices="0",
        gpu_memory_limit=1024**3,
        mps=True,
        env={"CUDA_VISIBLE_DEVICES": ""},
    )

    assert "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" not in build_worker_env(spec)


@pytest.mark.parametrize("limit", [0, -1, 1024**2 - 1])
def test_worker_spec_rejects_a_gpu_memory_limit_below_one_megabyte(limit):
    with pytest.raises(ValueError, match="gpu_memory_limit"):
        WorkerSpec(name="sdxl", class_path="m:P", gpu_memory_limit=limit)


def test_gpu_memory_cap_is_applied_before_the_predictor_import(
    predictors_path, workers
):
    (predictors_path / "torch.py").write_text(FAKE_TORCH)
    (predictors_path / "cogito_memory_probe_predictor.py").write_text(
        MEMORY_PROBE_PREDICTOR
    )
    out = predictors_path / "memory-probe.json"
    torch_out = predictors_path / "torch.json"
    spec = WorkerSpec(
        name="memory-probe",
        class_path="cogito_memory_probe_predictor:Predictor",
        cuda_visible_devices="3",
        gpu_memory_limit=4 * 1024**3,
        mps=True,
        env={"COGITO_TEST_OUT": str(out), "COGITO_TEST_TORCH_OUT": str(torch_out)},
    )

    worker = spawn_worker(spec, start_timeout=60.0)
    workers.append(worker)

    assert worker.handshake["status"] == "ready"
    # What the predictor module saw at import time, inside the worker.
    assert json.loads(out.read_text()) == {
        "COGITO_GPU_MEMORY_LIMIT": str(4 * 1024**3),
        "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT": "0=4096MB",
        "torch_capped": True,
    }
    assert json.loads(torch_out.read_text()) == {"fraction": 0.25, "device": 0}


def test_worker_env_is_applied_before_the_predictor_import(
    predictors_path, workers, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-devices")
    out = predictors_path / "probe.json"
    spec = WorkerSpec(
        name="probe",
        class_path="cogito_probe_predictor:Predictor",
        cuda_visible_devices="3",
        env={"COGITO_TEST_RESOURCE": "quota", "COGITO_TEST_OUT": str(out)},
    )

    worker = spawn_worker(spec, start_timeout=60.0)
    workers.append(worker)

    assert worker.process._start_method == "spawn"
    assert worker.handshake["status"] == "ready"
    assert worker.handshake["pid"] == worker.pid != os.getpid()
    assert worker.handshake["predictor"] == "cogito_probe_predictor.Predictor"
    assert worker.handshake["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    # What the predictor module saw at import time, inside the worker.
    assert json.loads(out.read_text()) == {
        "CUDA_VISIBLE_DEVICES": "3",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "COGITO_PREDICTOR_NAME": "probe",
        "COGITO_TEST_RESOURCE": "quota",
    }


def test_predictor_is_imported_only_inside_the_worker(
    predictors_path, workers, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-devices")
    spec = WorkerSpec(
        name="toy",
        class_path="cogito_toy_predictor:Predictor",
        cuda_visible_devices="",
    )

    worker = spawn_worker(spec, start_timeout=60.0)
    workers.append(worker)

    assert worker.handshake["status"] == "ready"
    assert "cogito_toy_predictor" not in sys.modules
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-devices"
    assert "COGITO_PREDICTOR_NAME" not in os.environ


def test_one_worker_per_predictor_each_with_its_own_devices(predictors_path, workers):
    for device in ("0", "1"):
        spec = WorkerSpec(
            name=f"probe-{device}",
            class_path="cogito_probe_predictor:Predictor",
            cuda_visible_devices=device,
            env={"COGITO_TEST_OUT": str(predictors_path / f"probe-{device}.json")},
        )
        workers.append(spawn_worker(spec, start_timeout=60.0))

    assert workers[0].pid != workers[1].pid
    for device in ("0", "1"):
        seen = json.loads((predictors_path / f"probe-{device}.json").read_text())
        assert seen["CUDA_VISIBLE_DEVICES"] == device
        assert seen["COGITO_PREDICTOR_NAME"] == f"probe-{device}"


def test_spawn_accepts_a_read_only_env_mapping(predictors_path, workers):
    spec = WorkerSpec(
        name="toy",
        class_path="cogito_toy_predictor:Predictor",
        env=MappingProxyType({"COGITO_TEST_RESOURCE": "quota"}),
    )

    worker = spawn_worker(spec, start_timeout=60.0)
    workers.append(worker)

    assert worker.handshake["env"]["COGITO_TEST_RESOURCE"] == "quota"


def test_worker_env_is_applied_before_the_launcher_bootstrap(predictors_path):
    launcher = predictors_path / "launcher.py"
    launcher.write_text(LAUNCHER)
    out = predictors_path / "bootstrap.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="parent-devices")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(predictors_path), env.get("PYTHONPATH")])
    )

    subprocess.run(
        [sys.executable, str(launcher), str(out)],
        env=env,
        check=True,
        timeout=120,
    )

    assert json.loads(out.read_text()) == {"CUDA_VISIBLE_DEVICES": "5"}


def test_stop_leaves_no_live_process(predictors_path):
    worker = spawn_worker(
        WorkerSpec(name="toy", class_path="cogito_toy_predictor:Predictor"),
        start_timeout=60.0,
    )
    assert worker.is_alive()

    worker.stop(timeout=5.0)

    assert not worker.is_alive()
    assert worker.process.exitcode == 0


def test_import_failure_raises_worker_start_error(predictors_path):
    spec = WorkerSpec(name="broken", class_path="cogito_broken_predictor:Predictor")

    with pytest.raises(WorkerStartError) as error:
        spawn_worker(spec, start_timeout=60.0)

    assert error.value.predictor == "broken"
    assert "broken on import" in error.value.detail


def test_missing_class_raises_worker_start_error(predictors_path):
    spec = WorkerSpec(name="toy", class_path="cogito_toy_predictor:Missing")

    with pytest.raises(WorkerStartError) as error:
        spawn_worker(spec, start_timeout=60.0)

    assert "Class Missing not found" in error.value.detail


# Predictors for the request/response tests. Like the ones above, the parent never
# imports them.
IPC_PREDICTORS = {
    "cogito_sleepy_predictor": """
        import os
        import time


        class Predictor:
            def setup(self):
                pass

            def predict(self, value: int, delay: float = 0.0) -> dict:
                time.sleep(delay)
                return {"value": value, "pid": os.getpid()}
    """,
    "cogito_model_predictor": """
        from pydantic import BaseModel


        class Output(BaseModel):
            text: str
            length: int


        class Predictor:
            def setup(self):
                pass

            def predict(self, text: str) -> Output:
                return Output(text=text.upper(), length=len(text))
    """,
    "cogito_failing_predictor": """
        from cogito.core.exceptions import BadRequestError


        class Predictor:
            def setup(self):
                pass

            def predict(self, mode: str):
                if mode == "bad-request":
                    raise BadRequestError("value must be positive")
                raise RuntimeError("model exploded")
    """,
    "cogito_async_predictor": """
        import asyncio


        class Predictor:
            def setup(self):
                pass

            async def predict(self, value: int, delay: float = 0.0) -> int:
                await asyncio.sleep(delay)
                return value * 2
    """,
    "cogito_exit_predictor": """
        import sys


        class Predictor:
            def setup(self):
                pass

            def predict(self, value: int) -> int:
                if value < 0:
                    sys.exit("negative value")
                return value
    """,
    "cogito_async_exit_predictor": """
        import sys


        class Predictor:
            def setup(self):
                pass

            async def predict(self, value: int) -> int:
                if value < 0:
                    sys.exit("negative value")
                return value
    """,
    "cogito_nested_exit_predictor": """
        import asyncio
        import sys


        async def child():
            sys.exit("child gave up")


        class Predictor:
            def setup(self):
                pass

            async def predict(self) -> int:
                await asyncio.gather(child())
                return 0
    """,
    "cogito_odd_errors_predictor": """
        from cogito.core.exceptions import BadRequestError


        class NoMessage(BadRequestError):
            def __init__(self):
                pass  # skips BadRequestError.__init__: no .message


        class BrokenStr(Exception):
            def __str__(self):
                raise RuntimeError("cannot render")


        class Predictor:
            def setup(self):
                pass

            def predict(self, kind: str):
                raise NoMessage() if kind == "no-message" else BrokenStr()
    """,
    "cogito_foreign_bad_request_predictor": """
        class BadRequestError(Exception):
            pass


        class Predictor:
            def setup(self):
                pass

            def predict(self):
                raise BadRequestError("not cogito's")
    """,
    "cogito_subclass_predictor": """
        class Score(float):
            pass


        class Label(str):
            pass


        class Predictor:
            def setup(self):
                pass

            def predict(self) -> dict:
                return {"score": Score(0.5), "label": Label("cat"), "scores": [Score(1)]}
    """,
    "cogito_exiting_predictor": """
        import os


        class Predictor:
            def setup(self):
                pass

            def predict(self):
                os._exit(3)
    """,
}


class TextInput(BaseModel):
    text: str


@pytest.fixture
def ipc_predictors_path(predictors_path):
    for module, source in IPC_PREDICTORS.items():
        (predictors_path / f"{module}.py").write_text(textwrap.dedent(source))
    return predictors_path


def start(workers, name, module, **spec_args):
    worker = spawn_worker(
        WorkerSpec(name=name, class_path=f"{module}:Predictor", **spec_args),
        start_timeout=60.0,
    )
    workers.append(worker)
    return worker


def test_concurrent_requests_resolve_to_the_right_caller(ipc_predictors_path, workers):
    worker = start(workers, "sleepy", "cogito_sleepy_predictor", concurrency=4)
    values = list(range(12))

    async def run():
        # Earlier requests sleep longer, so responses come back out of order.
        return await asyncio.gather(
            *(
                worker.predict({"value": value, "delay": (12 - value) * 0.02})
                for value in values
            )
        )

    results = asyncio.run(run())

    assert [result["value"] for result in results] == values
    assert {result["pid"] for result in results} == {worker.pid}


def test_timeout_raises_and_the_late_response_is_dropped(ipc_predictors_path, workers):
    worker = start(workers, "sleepy", "cogito_sleepy_predictor")

    async def run():
        with pytest.raises(WorkerTimeoutError) as error:
            await worker.predict({"value": 1, "delay": 1.0}, timeout=0.2)
        # Queued behind the timed-out request: it must get its own response, not
        # the late one.
        return error.value, await worker.predict({"value": 2}, timeout=30.0)

    error, result = asyncio.run(run())

    assert error.predictor == "sleepy"
    assert str(error) == "Predictor sleepy did not respond within 0.2s"
    assert result["value"] == 2
    assert not worker._pending


def test_timeout_surfaces_as_504_with_a_clear_message(ipc_predictors_path, workers):
    worker = start(workers, "sleepy", "cogito_sleepy_predictor")

    async def run():
        try:
            await worker.predict({"value": 1, "delay": 1.0}, timeout=0.2)
        except WorkerTimeoutError as error:
            return await worker_timeout_exception_handler(None, error)

    response = asyncio.run(run())

    assert response.status_code == 504
    assert json.loads(response.body) == {
        "detail": "Predictor sleepy did not respond within 0.2s"
    }


def test_pydantic_input_and_output_cross_as_plain_data(ipc_predictors_path, workers):
    worker = start(workers, "model", "cogito_model_predictor")

    result = asyncio.run(worker.predict(TextInput(text="hello")))

    assert result == {"text": "HELLO", "length": 5}
    assert "cogito_model_predictor" not in sys.modules


def test_bad_request_in_the_worker_raises_bad_request_error(
    ipc_predictors_path, workers
):
    worker = start(workers, "failing", "cogito_failing_predictor")

    with pytest.raises(BadRequestError) as error:
        asyncio.run(worker.predict({"mode": "bad-request"}))

    assert error.value.message == "value must be positive"


def test_predictor_failure_raises_with_the_worker_traceback(
    ipc_predictors_path, workers
):
    worker = start(workers, "failing", "cogito_failing_predictor")

    with pytest.raises(WorkerPredictionError) as error:
        asyncio.run(worker.predict({"mode": "crash"}))

    assert error.value.predictor == "failing"
    assert error.value.message == "model exploded"
    assert "RuntimeError: model exploded" in error.value.traceback
    assert worker.is_alive()


def test_async_predictor_runs_in_the_worker(ipc_predictors_path, workers):
    worker = start(workers, "async", "cogito_async_predictor", concurrency=3)

    async def run():
        return await asyncio.gather(
            *(worker.predict({"value": value, "delay": 0.05}) for value in range(5))
        )

    assert asyncio.run(run()) == [0, 2, 4, 6, 8]


def test_requests_in_flight_fail_when_the_worker_exits(ipc_predictors_path, workers):
    worker = start(workers, "exiting", "cogito_exiting_predictor")

    with pytest.raises(WorkerUnavailableError) as error:
        asyncio.run(worker.predict({}, timeout=30.0))

    assert error.value.predictor == "exiting"
    worker.process.join(5.0)
    assert worker.process.exitcode == 3


def test_predict_after_stop_raises_worker_unavailable(ipc_predictors_path):
    worker = spawn_worker(
        WorkerSpec(name="sleepy", class_path="cogito_sleepy_predictor:Predictor"),
        start_timeout=60.0,
    )
    worker.stop(timeout=5.0)

    with pytest.raises(WorkerUnavailableError):
        asyncio.run(worker.predict({"value": 1}))


def test_stop_waits_for_requests_already_running(ipc_predictors_path):
    worker = spawn_worker(
        WorkerSpec(name="sleepy", class_path="cogito_sleepy_predictor:Predictor"),
        start_timeout=60.0,
    )

    async def run():
        request = asyncio.ensure_future(
            worker.predict({"value": 7, "delay": 0.3}, timeout=30.0)
        )
        await asyncio.sleep(0.1)
        await asyncio.to_thread(worker.stop, 5.0)
        return await request

    assert asyncio.run(run())["value"] == 7
    assert worker.process.exitcode == 0


def test_concurrency_must_be_positive():
    with pytest.raises(ValueError):
        WorkerSpec(name="toy", class_path="m:P", concurrency=0)


def test_scalar_subclasses_from_the_predictor_module_cross_as_plain_data(
    ipc_predictors_path, workers
):
    worker = start(workers, "subclass", "cogito_subclass_predictor")

    result = asyncio.run(worker.predict({}))

    assert result == {"score": 0.5, "label": "cat", "scores": [1.0]}
    assert type(result["score"]) is float and type(result["label"]) is str
    assert "cogito_subclass_predictor" not in sys.modules


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="needs SIGSTOP")
def test_a_frozen_worker_times_out_and_still_stops(ipc_predictors_path):
    worker = spawn_worker(
        WorkerSpec(name="sleepy", class_path="cogito_sleepy_predictor:Predictor"),
        start_timeout=60.0,
    )
    # A frozen worker stops reading, so a large request fills the pipe and the
    # send itself blocks.
    os.kill(worker.pid, signal.SIGSTOP)
    padding = "x" * (16 * 1024 * 1024)

    async def run():
        try:
            with pytest.raises(WorkerTimeoutError):
                await worker.predict({"value": 1, "padding": padding}, timeout=0.5)
        finally:
            # The send is still blocked here, holding the send lock: stop() must not
            # wait for it. Killing the worker then fails the blocked send, so this
            # runs before asyncio.run waits for the sender thread.
            await asyncio.to_thread(worker.stop, 1.0)

    asyncio.run(asyncio.wait_for(run(), 30.0))

    assert not worker.is_alive()


@pytest.mark.parametrize(
    "module", ["cogito_exit_predictor", "cogito_async_exit_predictor"]
)
def test_system_exit_in_the_predictor_answers_the_request(
    ipc_predictors_path, workers, module
):
    worker = start(workers, "exit", module)

    async def run():
        with pytest.raises(WorkerPredictionError) as error:
            await worker.predict({"value": -1}, timeout=30.0)
        # The worker keeps serving requests afterwards.
        return error.value, await worker.predict({"value": 4}, timeout=30.0)

    error, result = asyncio.run(run())

    assert error.message == "negative value"
    assert "SystemExit: negative value" in error.traceback
    assert result == 4
    assert worker.is_alive()


def test_a_foreign_bad_request_error_is_a_prediction_error(
    ipc_predictors_path, workers
):
    worker = start(workers, "foreign", "cogito_foreign_bad_request_predictor")

    with pytest.raises(WorkerPredictionError) as error:
        asyncio.run(worker.predict({}))

    assert not isinstance(error.value, BadRequestError)
    assert error.value.message == "not cogito's"


def test_worker_exit_fails_a_request_whose_send_is_still_queued(ipc_predictors_path):
    worker = spawn_worker(
        WorkerSpec(name="sleepy", class_path="cogito_sleepy_predictor:Predictor"),
        start_timeout=60.0,
    )
    release = threading.Event()

    async def run():
        # Occupy the worker's sender thread, so the send stays queued.
        busy = asyncio.wrap_future(worker._sender.submit(release.wait))
        try:
            request = asyncio.ensure_future(worker.predict({"value": 1}))
            await asyncio.sleep(0.1)
            worker.process.kill()
            with pytest.raises(WorkerUnavailableError):
                await asyncio.wait_for(request, 10.0)
        finally:
            release.set()
            await busy

    try:
        asyncio.run(run())
    finally:
        worker.stop(timeout=5.0)


def test_system_exit_in_a_nested_task_fails_the_request(ipc_predictors_path, workers):
    worker = start(workers, "nested", "cogito_nested_exit_predictor")

    with pytest.raises(WorkerUnavailableError):
        asyncio.run(asyncio.wait_for(worker.predict({}), 30.0))

    worker.process.join(5.0)
    assert worker.process.exitcode == 1


def test_sends_do_not_depend_on_the_default_executor(ipc_predictors_path, workers):
    worker = start(workers, "sleepy", "cogito_sleepy_predictor")
    release = threading.Event()

    async def run():
        # A default executor exhausted by anything else (another worker's blocked
        # sends, say) must not delay this worker's requests.
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        busy = loop.run_in_executor(None, release.wait)
        try:
            return await worker.predict({"value": 3}, timeout=10.0)
        finally:
            release.set()
            await busy

    assert asyncio.run(run())["value"] == 3


def test_a_request_abandoned_while_queued_is_skipped_by_the_worker(
    ipc_predictors_path, workers
):
    worker = start(workers, "sleepy", "cogito_sleepy_predictor")

    async def run():
        # Keeps the only worker thread busy for 1s.
        running = asyncio.ensure_future(worker.predict({"value": 1, "delay": 1.0}))
        await asyncio.sleep(0.1)
        # Queued behind it, and abandoned: the worker must not run its 5s.
        with pytest.raises(WorkerTimeoutError):
            await worker.predict({"value": 2, "delay": 5.0}, timeout=0.2)
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        result = await worker.predict({"value": 3}, timeout=30.0)
        return (await running)["value"], result["value"], loop.time() - start_time

    first, third, waited = asyncio.run(run())

    assert (first, third) == (1, 3)
    # Only what is left of the first request, not the abandoned 5s on top.
    assert waited < 3.0


def test_odd_exceptions_still_get_a_reply(ipc_predictors_path, workers):
    worker = start(workers, "odd", "cogito_odd_errors_predictor")

    async def run():
        with pytest.raises(BadRequestError):
            await worker.predict({"kind": "no-message"}, timeout=30.0)
        with pytest.raises(WorkerPredictionError) as error:
            await worker.predict({"kind": "broken-str"}, timeout=30.0)
        return error.value

    error = asyncio.run(run())

    assert "BrokenStr" in error.message


def test_a_request_abandoned_while_its_send_is_running_is_still_cancelled(
    ipc_predictors_path, workers
):
    worker = start(workers, "sleepy", "cogito_sleepy_predictor")
    send = worker._send
    sending = threading.Event()
    release = threading.Event()

    def slow_send(message):
        # Holds the send of the request with value 2 in progress, as a large
        # payload would.
        if isinstance(message, dict) and message.get("inputs", {}).get("value") == 2:
            sending.set()
            release.wait()
        send(message)

    worker._send = slow_send

    async def run():
        running = asyncio.ensure_future(worker.predict({"value": 1, "delay": 1.0}))
        await asyncio.sleep(0.1)
        try:
            with pytest.raises(WorkerTimeoutError):
                await worker.predict({"value": 2, "delay": 5.0}, timeout=0.2)
            assert sending.is_set()
        finally:
            release.set()
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        result = await worker.predict({"value": 3}, timeout=30.0)
        return (await running)["value"], result["value"], loop.time() - start_time

    first, third, waited = asyncio.run(run())

    assert (first, third) == (1, 3)
    # The abandoned 5s request was skipped by the worker.
    assert waited < 3.0


# Predictors for the supervision tests. Like the ones above, the parent never
# imports them.
SUPERVISION_PREDICTORS = {
    "cogito_crashy_predictor": """
        import os
        import time


        class Predictor:
            def setup(self):
                pass

            def predict(self, crash: bool = False, delay: float = 0.0) -> int:
                time.sleep(delay)
                if crash:
                    os._exit(3)
                return os.getpid()
    """,
    # Forks a helper, which inherits the worker's end of the pipe, writes its pid
    # to out, and exits: the pipe never reaches EOF while the helper lives.
    "cogito_forking_predictor": """
        import os
        import time


        class Predictor:
            def setup(self):
                pass

            def predict(self, out: str) -> int:
                helper = os.fork()
                if helper == 0:
                    time.sleep(60)
                    os._exit(0)
                with open(out, "w") as f:
                    f.write(str(helper))
                os._exit(3)
    """,
    # Forks a helper that outlives the request (optionally ignoring SIGTERM) and
    # returns its pid; or sleeps for delay and returns the worker's pid.
    "cogito_helper_predictor": """
        import os
        import signal
        import time


        class Predictor:
            def setup(self):
                pass

            def predict(
                self,
                spawn_helper: bool = False,
                ignore_sigterm: bool = False,
                delay: float = 0.0,
                padding: str = "",
            ) -> int:
                if not spawn_helper:
                    time.sleep(delay)
                    return os.getpid()
                helper = os.fork()
                if helper == 0:
                    if ignore_sigterm:
                        signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    time.sleep(60)
                    os._exit(0)
                return helper
    """,
    # Sends the start of a message on its handshake pipe, then freezes while
    # loading: the gateway sees a handshake that never finishes arriving.
    "cogito_partial_handshake_predictor": """
        import gc
        import os
        import struct
        import time
        from multiprocessing.connection import Connection

        conn = next(o for o in gc.get_objects() if isinstance(o, Connection))
        os.write(conn.fileno(), struct.pack("!i", 1024 * 1024) + b"partial")
        time.sleep(600)


        class Predictor:
            def setup(self):
                pass

            def predict(self) -> int:
                return 0
    """,
    # The same, after forking a helper (its pid goes to COGITO_TEST_OUT), which
    # holds the handshake pipe as well.
    "cogito_partial_handshake_helper_predictor": """
        import gc
        import os
        import struct
        import time
        from multiprocessing.connection import Connection

        conn = next(o for o in gc.get_objects() if isinstance(o, Connection))
        helper = os.fork()
        if helper == 0:
            time.sleep(60)
            os._exit(0)
        with open(os.environ["COGITO_TEST_OUT"], "w") as out:
            out.write(str(helper))
        os.write(conn.fileno(), struct.pack("!i", 1024 * 1024) + b"partial")
        time.sleep(600)


        class Predictor:
            def setup(self):
                pass

            def predict(self) -> int:
                return 0
    """,
    # Forks a helper while loading, writes both pids to COGITO_TEST_OUT, then takes
    # a minute to load.
    "cogito_slow_load_predictor": """
        import json
        import os
        import time

        helper = os.fork()
        if helper == 0:
            time.sleep(60)
            os._exit(0)
        with open(os.environ["COGITO_TEST_OUT"], "w") as out:
            json.dump([os.getpid(), helper], out)
        time.sleep(60)


        class Predictor:
            def setup(self):
                pass

            def predict(self) -> int:
                return os.getpid()
    """,
    # Fails to load while the file named by COGITO_TEST_MARKER exists, and loads
    # slowly while the one named by COGITO_TEST_SLOW exists.
    "cogito_flaky_predictor": """
        import os
        import time

        if os.path.exists(os.environ.get("COGITO_TEST_MARKER", "")):
            raise RuntimeError("cannot load right now")
        if os.path.exists(os.environ.get("COGITO_TEST_SLOW", "")):
            time.sleep(60)


        class Predictor:
            def setup(self):
                pass

            def predict(self) -> int:
                return os.getpid()
    """,
}

# Starts the supervised workers, writes their pids to argv[1] and then, depending
# on argv[2]: returns without stopping them; returns while a send to a frozen
# worker is still blocked (also with a helper the worker forked holding the pipe);
# waits for SIGTERM the way uvicorn does (a handler that
# leaves the interpreter normally); or waits with a helper forked by a worker
# that ignores SIGTERM and a prediction still running; or does the same and then
# stops, waiting for that prediction; or never finishes loading its predictor
# (also after a slow worker bootstrap, during which the gateway dies).
SUPERVISOR_LAUNCHER = textwrap.dedent(
    """
    import asyncio
    import json
    import os
    import signal
    import sys
    import threading
    import time

    from cogito.core.exceptions import WorkerTimeoutError
    from cogito.core.worker import WorkerSpec, WorkerSupervisor

    if __name__ == "__mp_main__" and os.environ.get("COGITO_TEST_BOOT"):
        # Every worker re-imports this module while it bootstraps, before its
        # entry point runs: the gateway dies meanwhile.
        with open(os.environ["COGITO_TEST_BOOT"], "w") as out:
            out.write(str(os.getpid()))
        time.sleep(3)

    if __name__ == "__main__":
        mode = sys.argv[2]
        if mode == "sigterm":
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        if mode == "bootstrap":
            os.environ["COGITO_TEST_BOOT"] = sys.argv[1] + ".boot"
        if mode in ("loading", "bootstrap"):
            # The predictor itself writes the pids, and never finishes loading.
            os.environ["COGITO_TEST_OUT"] = sys.argv[1]
            WorkerSupervisor(
                [WorkerSpec(name="slow", class_path="cogito_slow_load_predictor:Predictor")],
                start_timeout=600.0,
            ).start()
        supervisor = WorkerSupervisor(
            [
                WorkerSpec(name=name, class_path="cogito_helper_predictor:Predictor")
                for name in ("first", "second")
            ],
            start_timeout=60.0,
            stop_timeout=2.0,
        )
        supervisor.start()
        pids = [worker.pid for worker in supervisor]
        if mode in ("blocked-helper", "wait", "stopping"):
            # A helper the worker forked, which holds its end of the pipe as well.
            pids.append(
                asyncio.run(
                    supervisor["first"].predict(
                        {"spawn_helper": True, "ignore_sigterm": mode != "blocked-helper"}
                    )
                )
            )
        if mode in ("wait", "stopping"):
            # A prediction still running when the gateway goes away.
            threading.Thread(
                target=asyncio.run,
                args=(supervisor["second"].predict({"delay": 60.0}),),
                daemon=True,
            ).start()
            time.sleep(0.5)
        if mode.startswith("blocked"):
            # A frozen worker stops reading, so a large request fills the pipe and
            # its send stays blocked after the request times out.
            os.kill(pids[0], signal.SIGSTOP)
            try:
                asyncio.run(
                    supervisor["first"].predict(
                        {"padding": "x" * (16 * 1024 * 1024)}, timeout=0.5
                    )
                )
            except WorkerTimeoutError:
                pass
        with open(sys.argv[1], "w") as out:
            json.dump(pids, out)
        if mode == "stopping":
            # Waits for the running prediction: the gateway dies while draining.
            supervisor.stop(timeout=600.0)
        if mode in ("sigterm", "wait"):
            while True:
                time.sleep(0.1)
    """
)

FAST_RESTARTS = RestartPolicy(initial_delay=0.01, max_delay=0.05, reset_after=60.0)


@pytest.fixture
def supervision_path(ipc_predictors_path):
    for module, source in SUPERVISION_PREDICTORS.items():
        (ipc_predictors_path / f"{module}.py").write_text(textwrap.dedent(source))
    return ipc_predictors_path


@pytest.fixture
def supervised():
    """Stop every supervised worker, even when the test fails."""
    started = []
    yield started
    for worker in started:
        worker.stop(timeout=5.0)


def supervise(supervised, name, module, policy=FAST_RESTARTS, **spec_args):
    worker = SupervisedWorker(
        WorkerSpec(name=name, class_path=f"{module}:Predictor", **spec_args),
        policy=policy,
        start_timeout=60.0,
    )
    supervised.append(worker)
    worker.start()
    return worker


def wait_until(condition, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.02)


def pid_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def restart_delays(caplog, name):
    return [
        record.delay
        for record in caplog.records
        if record.getMessage() == "Restarting worker" and record.predictor == name
    ]


def test_restart_delay_grows_exponentially_up_to_the_cap():
    policy = RestartPolicy(initial_delay=1.0, multiplier=2.0, max_delay=10.0)

    assert [policy.delay(attempt) for attempt in range(6)] == [1, 2, 4, 8, 10, 10]
    assert policy.delay(10_000) == 10.0


@pytest.mark.parametrize(
    "options",
    [
        {"initial_delay": -1},
        {"initial_delay": 5, "max_delay": 1},
        {"multiplier": 0.5},
        {"reset_after": -1},
        {"max_restarts": -1},
    ],
)
def test_restart_policy_rejects_invalid_values(options):
    with pytest.raises(ValueError):
        RestartPolicy(**options)


def test_a_crashed_worker_fails_its_request_and_is_restarted(
    supervision_path, supervised
):
    worker = supervise(supervised, "crashy", "cogito_crashy_predictor")
    first_pid = worker.pid

    with pytest.raises(WorkerUnavailableError):
        asyncio.run(worker.predict({"crash": True}, timeout=30.0))

    wait_until(lambda: worker.restarts == 1 and worker.is_available())
    assert worker.pid != first_pid
    assert asyncio.run(worker.predict({}, timeout=30.0)) == worker.pid


def test_a_killed_worker_is_restarted(supervision_path, supervised):
    worker = supervise(supervised, "crashy", "cogito_crashy_predictor")
    first_pid = worker.pid

    os.kill(first_pid, signal.SIGKILL)

    wait_until(lambda: worker.restarts == 1 and worker.is_available())
    assert worker.pid != first_pid
    assert not pid_exists(first_pid)


def test_every_request_in_flight_fails_when_the_worker_crashes(
    supervision_path, supervised
):
    worker = supervise(supervised, "crashy", "cogito_crashy_predictor", concurrency=4)

    async def run():
        return await asyncio.gather(
            *(worker.predict({"delay": 0.5}, timeout=30.0) for _ in range(3)),
            worker.predict({"crash": True, "delay": 0.1}, timeout=30.0),
            return_exceptions=True,
        )

    results = asyncio.run(run())

    assert all(isinstance(result, WorkerUnavailableError) for result in results)


def test_requests_fail_at_once_while_the_worker_restarts(supervision_path, supervised):
    worker = supervise(
        supervised,
        "crashy",
        "cogito_crashy_predictor",
        policy=RestartPolicy(initial_delay=60.0, max_delay=60.0),
    )

    with pytest.raises(WorkerUnavailableError):
        asyncio.run(worker.predict({"crash": True}, timeout=30.0))
    wait_until(lambda: worker.pid is None)

    async def run():
        # Bounded, so a request that waits for the restart fails the test.
        return await asyncio.wait_for(worker.predict({}), timeout=5.0)

    with pytest.raises(WorkerUnavailableError) as error:
        asyncio.run(run())
    assert error.value.detail == "worker is restarting"


def test_failed_restarts_back_off_until_the_worker_loads(
    supervision_path, supervised, caplog, monkeypatch
):
    marker = supervision_path / "broken"
    monkeypatch.setenv("COGITO_TEST_MARKER", str(marker))
    caplog.set_level(logging.WARNING, logger="cogito.core.worker")
    worker = supervise(
        supervised,
        "flaky",
        "cogito_flaky_predictor",
        policy=RestartPolicy(initial_delay=0.01, multiplier=2.0, max_delay=1.0),
    )
    first_pid = worker.pid

    marker.touch()
    os.kill(first_pid, signal.SIGKILL)
    wait_until(lambda: len(restart_delays(caplog, "flaky")) >= 4)
    marker.unlink()

    wait_until(lambda: worker.is_available())
    delays = restart_delays(caplog, "flaky")
    assert delays[:4] == [0.01, 0.02, 0.04, 0.08]
    assert worker.restarts == 1
    assert any(
        record.getMessage() == "Unable to restart worker" for record in caplog.records
    )


def test_backoff_starts_again_after_a_stable_uptime(
    supervision_path, supervised, caplog
):
    caplog.set_level(logging.WARNING, logger="cogito.core.worker")
    worker = supervise(
        supervised,
        "crashy",
        "cogito_crashy_predictor",
        policy=RestartPolicy(initial_delay=0.01, max_delay=1.0, reset_after=0.0),
    )

    for restarts in (1, 2):
        os.kill(worker.pid, signal.SIGKILL)
        wait_until(lambda: worker.restarts == restarts and worker.is_available())

    assert restart_delays(caplog, "crashy") == [0.01, 0.01]


def test_backoff_keeps_growing_for_a_worker_that_crashes_right_away(
    supervision_path, supervised, caplog
):
    caplog.set_level(logging.WARNING, logger="cogito.core.worker")
    worker = supervise(
        supervised,
        "crashy",
        "cogito_crashy_predictor",
        policy=RestartPolicy(initial_delay=0.01, max_delay=1.0, reset_after=60.0),
    )

    for restarts in (1, 2):
        os.kill(worker.pid, signal.SIGKILL)
        wait_until(lambda: worker.restarts == restarts and worker.is_available())

    assert restart_delays(caplog, "crashy") == [0.01, 0.02]


def test_a_worker_is_given_up_after_max_restarts(
    supervision_path, supervised, monkeypatch
):
    marker = supervision_path / "broken"
    monkeypatch.setenv("COGITO_TEST_MARKER", str(marker))
    worker = supervise(
        supervised,
        "flaky",
        "cogito_flaky_predictor",
        policy=RestartPolicy(initial_delay=0.01, max_delay=0.01, max_restarts=2),
    )

    marker.touch()
    os.kill(worker.pid, signal.SIGKILL)
    wait_until(lambda: not worker._monitor.is_alive())

    with pytest.raises(WorkerUnavailableError) as error:
        asyncio.run(worker.predict({}))
    assert error.value.detail == "worker crashed and is no longer restarted"
    assert worker.restarts == 0
    assert not multiprocessing.active_children()


def test_stop_interrupts_the_backoff_and_starts_nothing(supervision_path, supervised):
    worker = supervise(
        supervised,
        "crashy",
        "cogito_crashy_predictor",
        policy=RestartPolicy(initial_delay=60.0, max_delay=60.0),
    )
    os.kill(worker.pid, signal.SIGKILL)
    wait_until(lambda: worker.pid is None)

    stopper = threading.Thread(target=worker.stop, args=(5.0,))
    stopper.start()
    stopper.join(10.0)

    assert not stopper.is_alive()
    assert worker.restarts == 0
    assert not multiprocessing.active_children()


def test_stop_abandons_a_restart_in_progress(supervision_path, supervised, monkeypatch):
    slow = supervision_path / "slow"
    monkeypatch.setenv("COGITO_TEST_SLOW", str(slow))
    worker = supervise(supervised, "flaky", "cogito_flaky_predictor")
    first_pid = worker.pid

    slow.touch()
    os.kill(first_pid, signal.SIGKILL)
    # The restarted worker is now importing its predictor, which takes 60s.
    wait_until(
        lambda: any(
            child.pid != first_pid for child in multiprocessing.active_children()
        )
    )
    stopper = threading.Thread(target=worker.stop, args=(5.0,))
    stopper.start()
    stopper.join(20.0)

    assert not stopper.is_alive()
    assert worker.restarts == 0
    assert not multiprocessing.active_children()


def test_stop_abandons_the_first_start(supervision_path, monkeypatch):
    slow = supervision_path / "slow"
    slow.touch()
    monkeypatch.setenv("COGITO_TEST_SLOW", str(slow))
    worker = SupervisedWorker(
        WorkerSpec(name="flaky", class_path="cogito_flaky_predictor:Predictor"),
        start_timeout=60.0,
    )
    errors = []

    def start():
        try:
            worker.start()
        except WorkerStartError as error:
            errors.append(error)

    starter = threading.Thread(target=start)
    starter.start()
    # The worker is now importing its predictor, which takes 60s.
    wait_until(lambda: multiprocessing.active_children())
    worker.stop(timeout=5.0)
    starter.join(20.0)

    assert not starter.is_alive()
    assert [error.detail for error in errors] == ["stopped while starting"]
    assert worker._monitor is None
    assert not multiprocessing.active_children()


def test_stop_leaves_no_live_worker_and_is_idempotent(supervision_path, supervised):
    worker = supervise(supervised, "crashy", "cogito_crashy_predictor")
    pid = worker.pid

    worker.stop(timeout=5.0)
    worker.stop(timeout=5.0)

    assert not pid_exists(pid)
    assert not multiprocessing.active_children()
    with pytest.raises(WorkerUnavailableError) as error:
        asyncio.run(worker.predict({}))
    assert error.value.detail == "worker is not running"


def test_workers_ignore_sigint(supervision_path, supervised):
    worker = supervise(supervised, "crashy", "cogito_crashy_predictor")
    pid = worker.pid

    os.kill(pid, signal.SIGINT)

    assert asyncio.run(worker.predict({"delay": 0.2}, timeout=30.0)) == pid
    assert worker.restarts == 0


def test_supervisor_rejects_duplicated_predictor_names():
    spec = WorkerSpec(name="toy", class_path="cogito_toy_predictor:Predictor")

    with pytest.raises(ValueError):
        WorkerSupervisor([spec, spec])


def test_supervisor_stops_the_started_workers_when_one_cannot_start(supervision_path):
    supervisor = WorkerSupervisor(
        [
            WorkerSpec(name="good", class_path="cogito_crashy_predictor:Predictor"),
            WorkerSpec(name="broken", class_path="cogito_broken_predictor:Predictor"),
        ],
        start_timeout=60.0,
    )

    with pytest.raises(WorkerStartError):
        supervisor.start()

    assert not multiprocessing.active_children()


def test_supervisor_stops_every_worker(supervision_path):
    supervisor = WorkerSupervisor(
        [
            WorkerSpec(name=name, class_path="cogito_crashy_predictor:Predictor")
            for name in ("first", "second")
        ],
        policy=FAST_RESTARTS,
        start_timeout=60.0,
    )
    supervisor.start()
    pids = [worker.pid for worker in supervisor]
    assert asyncio.run(supervisor["second"].predict({}, timeout=30.0)) == pids[1]

    supervisor.stop(timeout=5.0)

    assert not any(pid_exists(pid) for pid in pids)
    assert not multiprocessing.active_children()


@pytest.fixture
def launchers():
    """Kill every launcher and its workers, even when the test fails."""
    started = []
    yield started
    for process, pids in started:
        if process.poll() is None:
            process.kill()
            process.wait()
        for pid in pids:
            # A frozen worker never sees its pipe close.
            if pid_exists(pid):
                os.kill(pid, signal.SIGKILL)


def run_supervisor_launcher(launchers, path, mode):
    launcher = path / "supervisor_launcher.py"
    launcher.write_text(SUPERVISOR_LAUNCHER)
    out = path / "pids.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(path), env.get("PYTHONPATH")])
    )
    process = subprocess.Popen([sys.executable, str(launcher), str(out), mode], env=env)
    launchers.append((process, []))
    wait_until(lambda: out.exists() and out.read_text())
    pids = json.loads(out.read_text())
    launchers[-1] = (process, pids)
    return process, pids


def test_no_worker_is_left_when_the_gateway_returns_without_stopping(
    supervision_path, launchers
):
    process, pids = run_supervisor_launcher(launchers, supervision_path, "return")

    assert process.wait(timeout=60) == 0
    assert not any(pid_exists(pid) for pid in pids)


def test_no_worker_is_left_when_exiting_with_a_send_still_blocked(
    supervision_path, launchers
):
    # Without it the exit hook falls back to atexit, which runs too late for this
    # case: fail clearly instead of by a timeout.
    assert hasattr(
        threading, "_register_atexit"
    ), "threading._register_atexit is gone: exit with a blocked send would hang"
    process, pids = run_supervisor_launcher(launchers, supervision_path, "blocked")

    assert process.wait(timeout=60) == 0
    assert not any(pid_exists(pid) for pid in pids)


def test_no_process_is_left_when_a_forked_helper_holds_a_blocked_pipe(
    supervision_path, launchers
):
    process, pids = run_supervisor_launcher(
        launchers, supervision_path, "blocked-helper"
    )

    assert process.wait(timeout=60) == 0
    # The helper is reparented once its worker dies: wait for it to be reaped.
    wait_until(lambda: not any(pid_exists(pid) for pid in pids), timeout=10.0)


def test_no_worker_is_left_after_sigterm(supervision_path, launchers):
    process, pids = run_supervisor_launcher(launchers, supervision_path, "sigterm")
    assert all(pid_exists(pid) for pid in pids)

    process.send_signal(signal.SIGTERM)

    assert process.wait(timeout=60) == 0
    assert not any(pid_exists(pid) for pid in pids)


def test_workers_exit_when_the_gateway_is_killed(supervision_path, launchers):
    process, pids = run_supervisor_launcher(launchers, supervision_path, "wait")

    process.kill()
    process.wait(timeout=60)

    # Their parent is gone: whoever adopts them reaps them once they exit. Well
    # before the helper (the last pid) would exit on its own after 60s.
    wait_until(lambda: not any(pid_exists(pid) for pid in pids), timeout=15.0)


def test_unavailable_worker_surfaces_as_503_with_a_clear_message(
    supervision_path, supervised
):
    worker = supervise(supervised, "crashy", "cogito_crashy_predictor")
    worker.stop(timeout=5.0)

    async def run():
        try:
            await worker.predict({})
        except WorkerUnavailableError as error:
            return await worker_unavailable_exception_handler(None, error)

    response = asyncio.run(run())

    assert response.status_code == 503
    assert json.loads(response.body) == {
        "detail": "Worker for predictor crashy is unavailable: worker is not running"
    }


def test_an_unexpected_error_while_restarting_is_retried(
    supervision_path, supervised, monkeypatch
):
    import cogito.core.worker as worker_module

    real_spawn = worker_module._spawn
    calls = []

    def flaky_spawn(*args, **kwargs):
        # The first call is the first start, which must succeed.
        calls.append(None)
        if len(calls) == 2:
            raise OSError("Too many open files")
        return real_spawn(*args, **kwargs)

    monkeypatch.setattr(worker_module, "_spawn", flaky_spawn)
    worker = supervise(supervised, "crashy", "cogito_crashy_predictor")

    os.kill(worker.pid, signal.SIGKILL)

    wait_until(lambda: worker.restarts == 1 and worker.is_available())
    assert len(calls) == 3


def test_exit_hook_falls_back_to_atexit_with_a_warning(monkeypatch, caplog):
    import atexit

    import cogito.core.worker as worker_module

    registered = []
    monkeypatch.delattr(threading, "_register_atexit")
    monkeypatch.setattr(atexit, "register", registered.append)
    monkeypatch.setattr(worker_module, "_EXIT_HOOK_REGISTERED", False)
    caplog.set_level(logging.WARNING, logger="cogito.core.worker")

    worker_module._register_exit_hook()

    assert registered == [worker_module._terminate_running_supervisors]
    assert any(
        "threading._register_atexit is not available" in record.getMessage()
        for record in caplog.records
    )


def test_a_worker_that_cannot_be_tracked_after_its_handshake_is_reaped(
    supervision_path, monkeypatch
):
    import cogito.core.worker as worker_module

    def broken_handle(*args, **kwargs):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(worker_module, "PredictorWorker", broken_handle)

    with pytest.raises(RuntimeError):
        spawn_worker(
            WorkerSpec(name="crashy", class_path="cogito_crashy_predictor:Predictor"),
            start_timeout=60.0,
        )

    assert not multiprocessing.active_children()


def test_a_worker_whose_monitor_cannot_start_is_terminated(
    supervision_path, monkeypatch
):
    real_start = threading.Thread.start

    def start(thread):
        if thread.name.startswith("cogito-supervisor-"):
            raise RuntimeError("can't start new thread")
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    worker = SupervisedWorker(
        WorkerSpec(name="crashy", class_path="cogito_crashy_predictor:Predictor"),
        start_timeout=60.0,
    )

    with pytest.raises(RuntimeError):
        worker.start()

    assert worker._monitor is None
    assert worker.pid is None
    assert not multiprocessing.active_children()
    worker.stop(timeout=5.0)


def test_a_worker_is_terminated_when_its_polite_stop_cannot_start(
    supervision_path, supervised, monkeypatch
):
    worker = supervise(supervised, "crashy", "cogito_crashy_predictor")
    pid = worker.pid
    real_start = threading.Thread.start

    def start(thread):
        if thread.name.startswith("cogito-stop-"):
            raise RuntimeError("can't start new thread")
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)

    with pytest.raises(RuntimeError):
        worker.stop(timeout=5.0)

    assert not pid_exists(pid)
    assert not multiprocessing.active_children()
    assert worker._wake_reader.closed


def test_supervisor_terminates_workers_it_cannot_stop_in_parallel(
    supervision_path, monkeypatch
):
    supervisor = WorkerSupervisor(
        [
            WorkerSpec(name=name, class_path="cogito_crashy_predictor:Predictor")
            for name in ("first", "second")
        ],
        start_timeout=60.0,
    )
    supervisor.start()
    pids = [worker.pid for worker in supervisor]
    real_start = threading.Thread.start

    def start(thread):
        if thread.name == "cogito-stop-supervisor-second":
            raise RuntimeError("can't start new thread")
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)

    supervisor.stop(timeout=5.0)

    assert not any(pid_exists(pid) for pid in pids)
    assert not multiprocessing.active_children()


def test_requests_fail_when_a_forked_helper_keeps_the_pipe_open(
    supervision_path, supervised
):
    worker = supervise(
        supervised,
        "forking",
        "cogito_forking_predictor",
        policy=RestartPolicy(initial_delay=60.0, max_delay=60.0),
    )
    out = supervision_path / "helper.pid"

    async def run():
        # No request timeout: only the bound below keeps a hang from blocking.
        return await asyncio.wait_for(worker.predict({"out": str(out)}), timeout=20.0)

    try:
        with pytest.raises(WorkerUnavailableError):
            asyncio.run(run())
    finally:
        if out.exists() and out.read_text():
            helper = int(out.read_text())
            if pid_exists(helper):
                os.kill(helper, signal.SIGKILL)


def test_zero_initial_delay_stays_zero_for_any_attempt():
    policy = RestartPolicy(initial_delay=0.0)

    assert policy.delay(0) == 0.0
    assert policy.delay(10_000) == 0.0


def test_stop_kills_the_processes_the_predictor_started(supervision_path, supervised):
    worker = supervise(supervised, "helper", "cogito_helper_predictor")
    helper = asyncio.run(worker.predict({"spawn_helper": True}, timeout=30.0))
    assert pid_exists(helper)

    try:
        started = time.monotonic()
        worker.stop(timeout=30.0)
        # The helper inherited the worker's sentinel: a join that only waited on it
        # would take the whole timeout.
        assert time.monotonic() - started < 10.0

        wait_until(lambda: not pid_exists(helper), timeout=10.0)
    finally:
        if pid_exists(helper):
            os.kill(helper, signal.SIGKILL)


def test_a_crashed_worker_leaves_no_forked_helper(supervision_path, supervised):
    worker = supervise(supervised, "forking", "cogito_forking_predictor")
    out = supervision_path / "helper.pid"

    with pytest.raises(WorkerUnavailableError):
        asyncio.run(asyncio.wait_for(worker.predict({"out": str(out)}), timeout=20.0))

    helper = int(out.read_text())
    try:
        wait_until(lambda: not pid_exists(helper), timeout=10.0)
    finally:
        if pid_exists(helper):
            os.kill(helper, signal.SIGKILL)


def test_no_process_is_left_when_the_gateway_dies_while_a_worker_loads(
    supervision_path, launchers
):
    process, pids = run_supervisor_launcher(launchers, supervision_path, "loading")

    process.kill()
    process.wait(timeout=60)

    wait_until(lambda: not any(pid_exists(pid) for pid in pids), timeout=15.0)


def test_no_process_is_left_when_the_gateway_dies_while_stopping(
    supervision_path, launchers
):
    process, pids = run_supervisor_launcher(launchers, supervision_path, "stopping")
    # Let the launcher send STOP and start waiting for the running prediction.
    time.sleep(1.0)

    process.kill()
    process.wait(timeout=60)

    wait_until(lambda: not any(pid_exists(pid) for pid in pids), timeout=15.0)


def test_a_start_interrupted_before_publishing_the_worker_terminates_it(
    supervision_path, monkeypatch
):
    class Interrupted(threading.Thread):
        def __init__(self, *args, **kwargs):
            if kwargs.get("name", "").startswith("cogito-supervisor-"):
                raise KeyboardInterrupt()
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", Interrupted)
    worker = SupervisedWorker(
        WorkerSpec(name="crashy", class_path="cogito_crashy_predictor:Predictor"),
        start_timeout=60.0,
    )

    with pytest.raises(KeyboardInterrupt):
        worker.start()

    assert worker.pid is None
    assert not multiprocessing.active_children()
    worker.stop(timeout=5.0)


def test_no_process_is_left_when_the_gateway_dies_while_a_worker_bootstraps(
    supervision_path, launchers
):
    launcher = supervision_path / "supervisor_launcher.py"
    launcher.write_text(SUPERVISOR_LAUNCHER)
    out = supervision_path / "pids.json"
    boot = supervision_path / "pids.json.boot"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(supervision_path), env.get("PYTHONPATH")])
    )
    process = subprocess.Popen(
        [sys.executable, str(launcher), str(out), "bootstrap"], env=env
    )
    launchers.append((process, []))
    wait_until(lambda: boot.exists() and boot.read_text())

    worker = int(boot.read_text())
    launchers[-1] = (process, [worker])

    process.kill()
    process.wait(timeout=60)

    # Its gateway is gone before it even runs: it must not go on loading its
    # predictor, which would take a minute.
    wait_until(lambda: not pid_exists(worker), timeout=15.0)
    assert not out.exists()


PARTIAL_HANDSHAKE = WorkerSpec(
    name="partial", class_path="cogito_partial_handshake_predictor:Predictor"
)


def test_stop_abandons_a_start_frozen_halfway_through_its_handshake(
    supervision_path,
):
    worker = SupervisedWorker(PARTIAL_HANDSHAKE, start_timeout=600.0)
    errors = []

    def start():
        try:
            worker.start()
        except WorkerStartError as error:
            errors.append(error)

    starter = threading.Thread(target=start)
    starter.start()
    # The partial handshake is now blocking the start's recv().
    time.sleep(2.0)
    worker.stop(timeout=5.0)
    starter.join(20.0)

    assert not starter.is_alive()
    assert [error.detail for error in errors] == ["stopped while starting"]
    assert not multiprocessing.active_children()


def test_a_handshake_frozen_halfway_still_times_out(supervision_path):
    with pytest.raises(WorkerStartError) as error:
        spawn_worker(PARTIAL_HANDSHAKE, start_timeout=3.0)

    assert error.value.detail == "predictor not ready after 3.0s"
    assert not multiprocessing.active_children()


def test_a_frozen_handshake_times_out_even_with_a_forked_helper(
    supervision_path, monkeypatch
):
    out = supervision_path / "helper.pid"
    monkeypatch.setenv("COGITO_TEST_OUT", str(out))
    errors = []

    def start():
        try:
            spawn_worker(
                WorkerSpec(
                    name="partial",
                    class_path="cogito_partial_handshake_helper_predictor:Predictor",
                ),
                start_timeout=3.0,
            )
        except WorkerStartError as error:
            errors.append(error)

    # In a thread, so a start that hangs fails the test instead of blocking it.
    starter = threading.Thread(target=start, daemon=True)
    starter.start()
    starter.join(30.0)
    helper = int(out.read_text()) if out.exists() and out.read_text() else None
    try:
        assert not starter.is_alive()
        assert [error.detail for error in errors] == ["predictor not ready after 3.0s"]
        assert helper is not None
        wait_until(lambda: not pid_exists(helper), timeout=10.0)
    finally:
        if helper is not None and pid_exists(helper):
            os.kill(helper, signal.SIGKILL)


@pytest.mark.parametrize("stop_during_start", [False, True])
def test_worker_replacement_during_start_is_not_shutdown(
    supervision_path, supervised, stop_during_start
):
    worker = SupervisedWorker(
        WorkerSpec(name="crashy", class_path="cogito_crashy_predictor:Predictor"),
        policy=FAST_RESTARTS,
        start_timeout=60.0,
    )
    supervised.append(worker)
    starter = threading.get_ident()
    real_lock = worker._lock
    stopper = None

    class ReplaceBeforeStartReturns:
        entries = 0

        def __enter__(self):
            real_lock.acquire()
            if threading.get_ident() == starter:
                self.entries += 1

        def __exit__(self, *args):
            nonlocal stopper
            replace = threading.get_ident() == starter and self.entries == 2
            real_lock.release()
            if replace:
                # Force the monitor to replace the first worker after publication
                # but before start() checks its outcome, without relying on timing.
                first = worker._worker
                os.kill(first.pid, signal.SIGKILL)
                wait_until(lambda: worker.restarts == 1 and worker.is_available())
                assert worker._worker is not first
                if stop_during_start:
                    stopper = threading.Thread(target=worker.stop, args=(5.0,))
                    stopper.start()
                    wait_until(worker._stopping.is_set)

    worker._lock = ReplaceBeforeStartReturns()
    try:
        if stop_during_start:
            with pytest.raises(WorkerStartError, match="stopped while starting"):
                worker.start()
        else:
            worker.start()
            assert worker.restarts == 1
            assert worker.is_available()
    finally:
        if stopper is not None:
            stopper.join(20.0)
            assert not stopper.is_alive()
    if stop_during_start:
        assert not worker.is_available()
        assert not multiprocessing.active_children()


def test_load_error_handshake_failure_cleans_up_the_process_group(monkeypatch):
    from unittest.mock import Mock

    from cogito.core import worker as worker_module

    class GroupExited(BaseException):
        pass

    # Exercise the worker entry point in isolation: never change pytest's signals
    # or process group, nor start its parent watchdog.
    monkeypatch.setattr(signal, "signal", Mock())
    monkeypatch.setattr(os, "setpgrp", Mock())
    monkeypatch.setattr(threading.Thread, "start", Mock())
    monkeypatch.setattr(
        worker_module, "_load_predictor", Mock(side_effect=RuntimeError("load failed"))
    )
    cleanup = Mock(side_effect=GroupExited)
    monkeypatch.setattr(worker_module, "_exit_with_process_group", cleanup)
    conn = Mock()
    conn.send.side_effect = BrokenPipeError("gateway is gone")

    with pytest.raises(GroupExited):
        worker_module._worker_main(WorkerSpec("broken", "broken:Predictor"), {}, conn)

    assert conn.send.call_args.args[0]["status"] == "error"
    assert "load failed" in conn.send.call_args.args[0]["error"]
    cleanup.assert_called_once_with()


def test_interrupted_monitor_start_does_not_restart_an_untracked_worker(
    supervision_path, supervised, monkeypatch
):
    real_start = threading.Thread.start
    monitors = []

    def start(thread):
        real_start(thread)
        if thread.name.startswith("cogito-supervisor-"):
            monitors.append(thread)
            raise KeyboardInterrupt()

    worker = SupervisedWorker(
        WorkerSpec(name="crashy", class_path="cogito_crashy_predictor:Predictor"),
        policy=FAST_RESTARTS,
        start_timeout=60.0,
    )
    supervised.append(worker)
    monkeypatch.setattr(threading.Thread, "start", start)

    with pytest.raises(KeyboardInterrupt):
        worker.start()

    assert len(monitors) == 1
    assert not monitors[0].is_alive()
    assert worker._monitor is None
    assert worker.pid is None
    assert worker.restarts == 0
    assert worker._wake_reader.closed
    assert not multiprocessing.active_children()


def test_interrupted_shutdown_keeps_the_exit_cleanup_registered(
    supervision_path, monkeypatch
):
    from cogito.core import worker as worker_module

    supervisor = WorkerSupervisor(
        [WorkerSpec(name="crashy", class_path="cogito_crashy_predictor:Predictor")],
        start_timeout=60.0,
    )
    supervisor.start()
    pid = supervisor["crashy"].pid
    real_start = threading.Thread.start

    def start(thread):
        if thread.name.startswith("cogito-stop-supervisor-"):
            raise KeyboardInterrupt()
        return real_start(thread)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", start)
            with pytest.raises(KeyboardInterrupt):
                supervisor.stop()
        assert supervisor in worker_module._RUNNING_SUPERVISORS
        # Exercise the actual interpreter-exit path, which starts no threads.
        worker_module._terminate_running_supervisors()
        assert not pid_exists(pid)
        assert not multiprocessing.active_children()
    finally:
        supervisor.stop()


@pytest.mark.parametrize("resume", ["stop", "terminate"])
def test_interrupted_monitor_join_keeps_worker_for_shutdown_retry(
    supervision_path, supervised, monkeypatch, resume
):
    worker = supervise(supervised, "crashy", "cogito_crashy_predictor")
    pid = worker.pid
    real_join = worker._monitor.join

    def interrupted_join(*args, **kwargs):
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(worker._monitor, "join", interrupted_join)
        with pytest.raises(KeyboardInterrupt):
            worker.stop(timeout=5.0)

    assert pid_exists(pid)
    assert worker._stopping_worker.pid == pid
    assert not worker._shutdown_complete
    assert not worker.is_available()
    getattr(worker, resume)(timeout=5.0)
    real_join(timeout=5.0)
    assert not worker._monitor.is_alive()
    assert worker._shutdown_complete
    assert worker._stopping_worker is None
    assert worker._wake_reader.closed
    assert not pid_exists(pid)
    assert not multiprocessing.active_children()


def test_start_publication_interruption_signals_its_completion(supervision_path):
    worker = SupervisedWorker(
        WorkerSpec(name="crashy", class_path="cogito_crashy_predictor:Predictor")
    )
    real_lock = worker._lock

    class InterruptPublication:
        def __enter__(self):
            real_lock.acquire()

        def __exit__(self, *args):
            real_lock.release()
            raise KeyboardInterrupt()

    worker._lock = InterruptPublication()
    try:
        with pytest.raises(KeyboardInterrupt):
            worker.start()
        assert worker._started
        assert worker._start_done.is_set()
    finally:
        worker._lock = real_lock
        # Bound cleanup even if this regression fails.
        worker._start_done.set()
        worker.stop(timeout=5.0)
    assert not multiprocessing.active_children()


def test_rejected_start_does_not_complete_another_start():
    worker = SupervisedWorker(WorkerSpec(name="crashy", class_path="unused:Predictor"))
    # A concurrent caller observes the state published by the accepted start.
    worker._started = True
    try:
        with pytest.raises(RuntimeError, match="already started"):
            worker.start()
        assert not worker._start_done.is_set()
    finally:
        worker._start_done.set()
        worker.stop(timeout=5.0)


def test_real_interrupted_join_waits_for_monitor_body_before_closing_resources(
    supervision_path, supervised, monkeypatch
):
    worker = SupervisedWorker(
        WorkerSpec(name="crashy", class_path="cogito_crashy_predictor:Predictor"),
        start_timeout=60.0,
    )
    supervised.append(worker)
    entered = threading.Event()
    release = threading.Event()
    joining = threading.Event()
    resumed = threading.Event()
    real_watch = worker._watch

    def blocked_watch(initial):
        entered.set()
        release.wait()
        real_watch(initial)

    monkeypatch.setattr(worker, "_watch", blocked_watch)
    worker.start()
    assert entered.wait(5.0)
    real_join = worker._monitor.join

    def join(*args, **kwargs):
        joining.set()
        return real_join(*args, **kwargs)

    def interrupt_join():
        if joining.wait(5.0):
            time.sleep(0.1)
            os.kill(os.getpid(), signal.SIGINT)

    def resume():
        worker.terminate(timeout=5.0)
        resumed.set()

    interrupter = threading.Thread(target=interrupt_join)
    recovery = threading.Thread(target=resume)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(worker._monitor, "join", join)
            interrupter.start()
            with pytest.raises(KeyboardInterrupt):
                worker.stop(timeout=5.0)
        interrupter.join(5.0)
        recovery.start()
        # A falsely stopped Thread must not permit closing resources while its
        # target is still running. This also passes on CPython with join fixed.
        assert not resumed.wait(0.2)
        assert not worker._monitor_done.is_set()
        assert not worker._wake_reader.closed
        assert not worker._shutdown_complete
    finally:
        release.set()
        interrupter.join(5.0)
        if recovery.ident is not None:
            recovery.join(10.0)
    assert resumed.is_set()
    assert worker._monitor_done.is_set()
    assert worker._wake_reader.closed
    assert not multiprocessing.active_children()


def test_interruption_after_monitor_start_return_cancels_supervision(
    supervision_path, supervised
):
    worker = SupervisedWorker(
        WorkerSpec(name="crashy", class_path="cogito_crashy_predictor:Predictor"),
        policy=FAST_RESTARTS,
        start_timeout=60.0,
    )
    supervised.append(worker)
    launched = []
    previous_trace = sys.gettrace()

    def interrupt_after_thread_start(frame, event, arg):
        if frame.f_code is SupervisedWorker.start.__code__ and event == "line":
            monitor = frame.f_locals.get("monitor")
            if monitor is not None and monitor.ident is not None:
                # The first source line after real Thread.start() returned,
                # before the ownership assignment can run.
                launched.append(monitor)
                raise KeyboardInterrupt()
        return interrupt_after_thread_start

    try:
        sys.settrace(interrupt_after_thread_start)
        with pytest.raises(KeyboardInterrupt):
            worker.start()
    finally:
        sys.settrace(previous_trace)
    assert len(launched) == 1
    assert worker._monitor_done.is_set()
    assert not launched[0].is_alive()
    assert worker._monitor is None
    assert worker.restarts == 0
    assert worker._wake_reader.closed
    assert not multiprocessing.active_children()


SETUP_PREDICTORS = {
    # predict() returns what setup() stored: it fails if setup() never ran.
    "cogito_stateful_predictor": """
        import os


        class Predictor:
            def setup(self):
                self.state = f"set up in {os.getpid()}"

            def predict(self) -> str:
                return self.state
    """,
    # setup() creates the file at COGITO_TEST_STARTED, then blocks until the file
    # at COGITO_TEST_GATE exists.
    "cogito_gated_predictor": """
        import os
        import time


        class Predictor:
            def setup(self):
                open(os.environ["COGITO_TEST_STARTED"], "w").close()
                while not os.path.exists(os.environ["COGITO_TEST_GATE"]):
                    time.sleep(0.02)

            def predict(self) -> str:
                return "ok"
    """,
    "cogito_failing_setup_predictor": """
        class Predictor:
            def setup(self):
                raise RuntimeError("model weights missing")

            def predict(self) -> str:
                return "unreachable"
    """,
    # setup() fails while the file at COGITO_TEST_MARKER exists.
    "cogito_flaky_setup_predictor": """
        import os


        class Predictor:
            def setup(self):
                if os.path.exists(os.environ["COGITO_TEST_MARKER"]):
                    raise RuntimeError("weights not downloaded yet")

            def predict(self) -> int:
                return os.getpid()
    """,
    # predict() runs on the loop setup() ran on, and uses what setup() bound to it.
    "cogito_async_setup_predictor": """
        import asyncio


        class Predictor:
            async def setup(self):
                self.loop = asyncio.get_running_loop()
                self.queue = asyncio.Queue()
                await self.queue.put("from setup")

            async def predict(self) -> dict:
                return {
                    "same_loop": asyncio.get_running_loop() is self.loop,
                    "value": await self.queue.get(),
                }
    """,
    # A sync setup() that needs a running loop, as it has in-process.
    "cogito_sync_setup_async_predict_predictor": """
        import asyncio


        class Predictor:
            def setup(self):
                self.loop = asyncio.get_running_loop()

            async def predict(self) -> bool:
                return asyncio.get_running_loop() is self.loop
    """,
    "cogito_async_setup_sync_predict_predictor": """
        import asyncio


        class Predictor:
            async def setup(self):
                await asyncio.sleep(0)
                self.value = "set up asynchronously"

            def predict(self) -> str:
                return self.value
    """,
}


@pytest.fixture
def setup_path(supervision_path):
    for module, source in SETUP_PREDICTORS.items():
        (supervision_path / f"{module}.py").write_text(textwrap.dedent(source))
    return supervision_path


@contextlib.contextmanager
def gateway(supervisor, readiness_file):
    """
    What a gateway does around its workers: it serves /health-check while they
    start, and writes the readiness file once WorkerSupervisor.start() returns.

    Yields the /health-check status code getter and the start errors.
    """
    handler = create_health_check_handler(readiness_file)
    resources = contextlib.ExitStack()
    start_errors = []

    def start_workers():
        try:
            supervisor.start()
        except WorkerStartError as e:
            start_errors.append(e)
            return
        resources.enter_context(readiness_context(readiness_file))

    def health_check():
        return asyncio.run(handler(None)).status_code

    starter = threading.Thread(target=start_workers)
    starter.start()
    try:
        yield health_check, start_errors
    finally:
        starter.join()
        resources.close()
        supervisor.stop()


def test_setup_runs_in_the_worker_before_it_reports_ready(setup_path, workers):
    worker = start(workers, "stateful", "cogito_stateful_predictor")

    assert asyncio.run(worker.predict({})) == f"set up in {worker.pid}"


def test_async_setup_shares_the_event_loop_of_predict(setup_path, workers):
    worker = start(workers, "async", "cogito_async_setup_predictor")

    assert asyncio.run(worker.predict({})) == {
        "same_loop": True,
        "value": "from setup",
    }


def test_sync_setup_runs_on_the_event_loop_of_an_async_predict(setup_path, workers):
    worker = start(workers, "sync_setup", "cogito_sync_setup_async_predict_predictor")

    assert asyncio.run(worker.predict({})) is True


def test_async_setup_with_a_sync_predict(setup_path, workers):
    worker = start(workers, "mixed", "cogito_async_setup_sync_predict_predictor")

    assert asyncio.run(worker.predict({})) == "set up asynchronously"


def test_health_check_is_503_until_every_worker_is_set_up(
    setup_path, tmp_path, monkeypatch
):
    gate = tmp_path / "gate"
    setup_started = tmp_path / "setup-started"
    readiness_file = str(tmp_path / "readiness.lock")
    monkeypatch.setenv("COGITO_TEST_GATE", str(gate))
    monkeypatch.setenv("COGITO_TEST_STARTED", str(setup_started))
    supervisor = WorkerSupervisor(
        [
            WorkerSpec(name="fast", class_path="cogito_stateful_predictor:Predictor"),
            WorkerSpec(name="slow", class_path="cogito_gated_predictor:Predictor"),
        ],
        start_timeout=60.0,
    )
    with gateway(supervisor, readiness_file) as (health_check, start_errors):
        # The fast worker is set up, the slow one is still in its setup().
        wait_until(setup_started.exists)
        assert supervisor["fast"].is_available()
        assert not supervisor["slow"].is_available()
        assert health_check() == 503
        assert not os.path.exists(readiness_file)

        gate.touch()
        wait_until(lambda: health_check() == 200)
        assert supervisor["fast"].is_available()
        assert supervisor["slow"].is_available()

    assert start_errors == []
    assert not os.path.exists(readiness_file)
    assert not multiprocessing.active_children()


def test_setup_failure_is_logged_with_the_predictor_name(setup_path, tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="cogito.core.worker")
    readiness_file = str(tmp_path / "readiness.lock")
    supervisor = WorkerSupervisor(
        [
            WorkerSpec(name="good", class_path="cogito_stateful_predictor:Predictor"),
            WorkerSpec(
                name="broken", class_path="cogito_failing_setup_predictor:Predictor"
            ),
        ],
        start_timeout=60.0,
    )
    with gateway(supervisor, readiness_file) as (health_check, start_errors):
        wait_until(lambda: start_errors)
        assert health_check() == 503
        assert not os.path.exists(readiness_file)

    [error] = start_errors
    assert error.predictor == "broken"
    assert error.detail.startswith("setup() failed: ")
    assert "RuntimeError: model weights missing" in error.detail
    [record] = [
        record
        for record in caplog.records
        if record.getMessage() == "Unable to set up predictor"
    ]
    assert record.predictor == "broken"
    assert "model weights missing" in record.error
    # The worker that was set up is stopped with the rest.
    assert not multiprocessing.active_children()


def test_a_setup_failure_on_restart_is_logged_and_retried(
    setup_path, supervised, caplog, monkeypatch
):
    marker = setup_path / "not-downloaded"
    monkeypatch.setenv("COGITO_TEST_MARKER", str(marker))
    caplog.set_level(logging.WARNING, logger="cogito.core.worker")
    worker = supervise(supervised, "flaky", "cogito_flaky_setup_predictor")
    first_pid = worker.pid

    marker.touch()
    os.kill(first_pid, signal.SIGKILL)
    wait_until(
        lambda: any(
            record.getMessage() == "Unable to set up predictor"
            and record.predictor == "flaky"
            for record in caplog.records
        )
    )
    marker.unlink()

    wait_until(lambda: worker.is_available())
    assert worker.pid != first_pid
    assert asyncio.run(worker.predict({})) == worker.pid
