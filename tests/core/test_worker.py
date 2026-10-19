import json
import os
import sys
import textwrap
from types import MappingProxyType

import pytest

from cogito.core.exceptions import WorkerStartError
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
