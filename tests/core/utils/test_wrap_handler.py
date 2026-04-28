import json

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cogito.api.responses import ResultResponse
from cogito.core.utils import wrap_handler

from pydantic._internal._model_construction import ModelMetaclass


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


def test_wrap_handler_returns_traceback_on_error():
    class BoomPredictor:
        def predict(self, input: str) -> str:
            raise RuntimeError("boom from third party integration")

    original_handler = BoomPredictor().predict
    wrapped_handler = wrap_handler(
        "predict:BoomPredictor", original_handler, ResultResponse
    )

    class InputModel(BaseModel):
        input: str

    response = wrapped_handler(InputModel(input="World"))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 500

    payload = json.loads(response.body)
    assert payload["message"] == "boom from third party integration"
    assert payload["traceback"] is not None
    assert "Traceback (most recent call last)" in payload["traceback"]
    assert "RuntimeError: boom from third party integration" in payload["traceback"]
    assert "in predict" in payload["traceback"]
