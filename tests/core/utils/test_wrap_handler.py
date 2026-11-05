import asyncio
import json
import threading
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cogito.api.responses import ResultResponse
from cogito.core.exceptions import BadRequestError, NoThreadsAvailableError
from cogito.core.utils import create_routes_semaphores, wrap_handler

from pydantic._internal._model_construction import ModelMetaclass


def _config_with_traceback(enabled: bool):
    config = MagicMock()
    config.get_cogito_param.side_effect = lambda key: {
        "server.return_input_on_response": True,
        "server.return_traceback_on_response": enabled,
    }.get(key)
    return config


def test_wrap_handler_str_output():
    class MockPredictor:
        def predict(self, input: str) -> str:
            return f"Hello, {input}"

    original_handler = MockPredictor().predict
    wrapped_handler = wrap_handler(
        "predict:MockPredictor", original_handler, ResultResponse
    )

    class InputModel(BaseModel):
        input: str

    input_data = InputModel(input="World")
    response = wrapped_handler(input_data)
    assert response.result == "Hello, World"


def test_wrap_handler_float_input_int_output():
    class MockPredictor:
        def predict(self, input: float) -> int:
            return int(input)

    original_handler = MockPredictor().predict
    wrapped_handler = wrap_handler(
        "predict:MockPredictor", original_handler, ResultResponse
    )

    class InputModel(BaseModel):
        input: float

    input_data = InputModel(input=3.14)
    response = wrapped_handler(input_data)
    assert response.result == 3


def test_wrap_handler_base_model_input_str_output():

    class MyModel(BaseModel):
        input: str

    class MockPredict:
        def predict(self, input: MyModel) -> str:
            return f"Hello, {input}"

    original_handler = MockPredict().predict
    wrapped_handler = wrap_handler(
        "predict:MockPredictor", original_handler, ResultResponse
    )
    input_data = MyModel(input="World")
    response = wrapped_handler(input_data)
    assert response.result == "Hello, World"

    wrapped_handler_annotations = wrapped_handler.__annotations__

    assert issubclass(wrapped_handler_annotations["input"], BaseModel)
    assert issubclass(wrapped_handler_annotations["return"], ResultResponse)


def test_wrap_handler_returns_traceback_when_enabled():
    class BoomPredictor:
        def predict(self, input: str) -> str:
            raise RuntimeError("boom from third party integration")

    wrapped_handler = wrap_handler(
        "predict:BoomPredictor",
        BoomPredictor().predict,
        ResultResponse,
        config=_config_with_traceback(True),
    )

    class InputModel(BaseModel):
        input: str

    response = wrapped_handler(InputModel(input="World"))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 500

    payload = json.loads(response.body)
    assert payload["message"] == "boom from third party integration"
    assert "Traceback (most recent call last)" in payload["traceback"]
    assert "RuntimeError: boom from third party integration" in payload["traceback"]
    assert "in predict" in payload["traceback"]


def test_wrap_handler_omits_traceback_when_disabled():
    class BoomPredictor:
        def predict(self, input: str) -> str:
            raise RuntimeError("boom")

    wrapped_handler = wrap_handler(
        "predict:BoomPredictor",
        BoomPredictor().predict,
        ResultResponse,
        config=_config_with_traceback(False),
    )

    class InputModel(BaseModel):
        input: str

    response = wrapped_handler(InputModel(input="World"))
    payload = json.loads(response.body)
    assert payload == {"message": "boom"}


def test_wrap_handler_async_returns_traceback_when_enabled():
    class BoomPredictor:
        async def predict(self, input: str) -> str:
            raise RuntimeError("async boom")

    wrapped_handler = wrap_handler(
        "predict:BoomPredictor",
        BoomPredictor().predict,
        ResultResponse,
        config=_config_with_traceback(True),
    )

    class InputModel(BaseModel):
        input: str

    response = asyncio.run(wrapped_handler(InputModel(input="World")))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 500

    payload = json.loads(response.body)
    assert payload["message"] == "async boom"
    assert "RuntimeError: async boom" in payload["traceback"]
    assert "in predict" in payload["traceback"]


def test_wrap_handler_async_omits_traceback_when_disabled():
    class BoomPredictor:
        async def predict(self, input: str) -> str:
            raise RuntimeError("async boom")

    wrapped_handler = wrap_handler(
        "predict:BoomPredictor",
        BoomPredictor().predict,
        ResultResponse,
        config=_config_with_traceback(False),
    )

    class InputModel(BaseModel):
        input: str

    response = asyncio.run(wrapped_handler(InputModel(input="World")))
    payload = json.loads(response.body)
    assert payload == {"message": "async boom"}


def _blocking_sync_handler(threads: int):
    """A sync predictor that blocks until `release` is set, behind `threads`."""
    started = threading.Semaphore(0)
    release = threading.Event()

    class BlockingPredictor:
        def predict(self, input: str) -> str:
            started.release()
            release.wait(timeout=10)
            return input

    semaphore = threading.BoundedSemaphore(threads)
    wrapped_handler = wrap_handler(
        "predict:BlockingPredictor",
        BlockingPredictor().predict,
        ResultResponse,
        semaphore=semaphore,
    )
    return wrapped_handler, started, release


class _SyncInput(BaseModel):
    input: str


@contextmanager
def _in_flight(wrapped_handler, started, release, count):
    """Keep `count` requests running in their own threads while the block runs.

    On exit, whether the block failed or not, the requests are let through and
    their threads joined, so a failed assertion leaves nothing behind.
    """
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(wrapped_handler(_SyncInput(input="x")))
        )
        for _ in range(count)
    ]
    try:
        for thread in threads:
            thread.start()
        for _ in range(count):
            assert started.acquire(timeout=5)
        yield results
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)


@pytest.mark.parametrize("threads", [1, 2])
def test_wrap_handler_sync_rejects_requests_above_threads(threads):
    wrapped_handler, started, release = _blocking_sync_handler(threads)

    with _in_flight(wrapped_handler, started, release, threads) as results:
        with pytest.raises(NoThreadsAvailableError):
            wrapped_handler(_SyncInput(input="x"))

    assert [r.result for r in results] == ["x"] * threads


def test_wrap_handler_sync_limit_does_not_drift():
    wrapped_handler, started, release = _blocking_sync_handler(1)
    release.set()
    for _ in range(5):
        assert wrapped_handler(_SyncInput(input="x")).result == "x"
        assert started.acquire(timeout=5)

    release.clear()
    with _in_flight(wrapped_handler, started, release, 1):
        with pytest.raises(NoThreadsAvailableError):
            wrapped_handler(_SyncInput(input="x"))


@pytest.mark.parametrize("error", [RuntimeError("boom"), BadRequestError("bad")])
def test_wrap_handler_sync_releases_thread_on_error(error):
    class FailingPredictor:
        def predict(self, input: str) -> str:
            raise error

    semaphore = threading.BoundedSemaphore(1)
    wrapped_handler = wrap_handler(
        "predict:FailingPredictor",
        FailingPredictor().predict,
        ResultResponse,
        semaphore=semaphore,
    )

    for _ in range(2):
        try:
            wrapped_handler(_SyncInput(input="x"))
        except BadRequestError:
            pass
    assert semaphore.acquire(blocking=False)


@pytest.mark.parametrize(
    "is_async, expected",
    [(True, asyncio.Semaphore), (False, threading.BoundedSemaphore)],
)
def test_create_routes_semaphores_matches_predictor_kind(is_async, expected):
    config = MagicMock()
    config.cogito.get_predictor = "predict:Predictor"
    config.cogito.get_server_threads = 3

    semaphore = create_routes_semaphores(config, is_async=is_async)["predict:Predictor"]
    assert isinstance(semaphore, expected)
