import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
from types import MappingProxyType

import pytest
from pydantic import BaseModel

from cogito.core.exceptioin_handlers import worker_timeout_exception_handler
from cogito.core.exceptions import (
    BadRequestError,
    WorkerPredictionError,
    WorkerStartError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from cogito.core.worker import WorkerSpec, build_worker_env, spawn_worker

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
