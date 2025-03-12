import asyncio
from contextlib import contextmanager
import importlib
import inspect
import logging
import os
import time
from inspect import Parameter, signature
from typing import Any, Callable, Dict, get_type_hints

try:
    # Pydantic v2
    from pydantic.fields import FieldInfo as Field
except ImportError:
    # Pydantic v1
    from pydantic.fields import Field

from pydantic import create_model

from cogito.api.responses import ErrorResponse, ResultResponse
from cogito.core.config.file import ConfigFile
from cogito.core.exceptions import (
    ModelDownloadError,
    NoThreadsAvailableError,
    BadRequestError,
)
from cogito.core.metrics import inference_duration_histogram
from cogito.core.model_store import download_gcp_model, download_huggingface_model
from cogito.core.models import BasePredictor


def instance_class(class_path) -> Any:
    """
    Instance a class from a string path
    """
    path, class_name = class_path.split(":")
    module = importlib.import_module(f"{path}")


def instance_class(class_path) -> Any:
    """
    Instance a class from a string path
    """
    if not class_path:
        raise ValueError(f"No class path {class_path} specified.")

    path, class_name = class_path.split(":")
    module = importlib.import_module(f"{path}")

    if not hasattr(module, class_name):
        raise AttributeError(f"Class {class_name} not found in module {path}")
    if not hasattr(module, class_name):
        raise AttributeError(f"Class {class_name} not found in module {path}")

    object_class = getattr(module, class_name)
    object_class = getattr(module, class_name)

    # Build an instance of the class
    instance = object_class()
    instance = object_class()

    # Instantiate and return the class
    return instance
    return instance


def get_predictor_handler_return_type(predictor: BasePredictor):
    """This method returns the type of the output of the predictor.predict method"""
    # Get the return type of the predictor.predict method
    return_type = predictor.predict.__annotations__.get("return", Any)

    # Create a new dynamic type based on ResultResponse, with the correct module and annotated field
    return_class = type(
        f"{predictor.__class__.__name__}Response",
        (ResultResponse,),
        {
            "__annotations__": {
                "result": return_type
            },  # Annotate the result field with the return type
            "__module__": ResultResponse.__module__,  # Ensure the module is set correctly for Pydantic
        },
    )

    return return_class


def wrap_handler(
    descriptor: str,
    original_handler: Callable,
    response_model: ResultResponse,
    semaphore: asyncio.Semaphore = None,
) -> Callable:
    class_name, input_model = create_request_model(descriptor, original_handler)

    # Check if the original handler is an async function
    # Fixme Unify handler after replacing status checking model with file based mode.
    if inspect.iscoroutinefunction(original_handler):

        async def handler(input: input_model):
            async def a_timed_handler(input):
                result = None
                try:
                    dict_input = input.model_dump()
                except:
                    dict_input = input.dict()

                try:
                    start_time = time.time()
                    result = await original_handler(**dict_input)
                    end_time = time.time() - start_time
                    inference_duration_histogram.record(
                        end_time * 1000, {"predictor": class_name, "async": True}
                    )
                    # todo Count successful requests
                except BadRequestError as e:
                    raise
                except Exception as e:
                    logging.exception(e)
                    # todo Count failed requests
                    return ErrorResponse(message=str(e)).to_json_response()

                return response_model(
                    inference_time_seconds=end_time,
                    input=dict_input,
                    result=result,
                )

            if not semaphore:
                return await a_timed_handler(input)
            else:
                if semaphore.locked():
                    raise NoThreadsAvailableError(descriptor)
                await semaphore.acquire()
                try:
                    return await a_timed_handler(input)
                finally:
                    semaphore.release()

    else:

        def handler(input: input_model):
            def timed_handler(input):
                result = None
                try:
                    dict_input = input.model_dump()
                except:
                    dict_input = input.dict()
                try:
                    start_time = time.time()
                    result = original_handler(**dict_input)
                    end_time = time.time() - start_time
                    inference_duration_histogram.record(
                        end_time * 1000, {"predictor": class_name, "async": False}
                    )
                    # todo Count successful requests
                except BadRequestError as e:
                    raise
                except Exception as e:
                    logging.exception(e)
                    # todo Count failed requests
                    return ErrorResponse(message=str(e)).to_json_response()

                return response_model(
                    inference_time_seconds=end_time,
                    input=dict_input,
                    result=result,
                )

            if not semaphore:
                return timed_handler(input)
            else:
                if semaphore.locked():
                    raise NoThreadsAvailableError(descriptor)
                semaphore.acquire()
                try:
                    return timed_handler(input)
                finally:
                    semaphore.release()

    handler.__annotations__ = {"input": input_model, "return": response_model}
    logging.debug(
        f"Handler of {original_handler.__name__} annotated with {handler.__annotations__}"
    )
    return handler


# TODO: Maybe the return is not correct: class_name,input_model
# It is only used to create the input model
# class_name must be resolved outside of this function
def create_request_model(descriptor, original_handler):
    sig = signature(original_handler)
    type_hints = get_type_hints(original_handler)

    _, class_name = descriptor.split(":")

    input_fields = {}
    for name, param in sig.parameters.items():
        param_type = type_hints.get(name, Any)
        default_value = param.default if param.default != Parameter.empty else ...
        if not isinstance(default_value, type(...)):  # Comprueba si no es Ellipsis
            if isinstance(default_value, Field):
                default_value = default_value.default
        input_fields[name] = (param_type, Field(default=default_value))
    input_model = create_model(f"{class_name}Request", **input_fields)
    return class_name, input_model


def model_download(model_path: str) -> str:
    """
    Download a model from various sources based on the model path format.
    Supported formats:
    - Google Cloud Storage: gs://bucket/path/to/model
    - Hugging Face: repo_owner/repo_name
    """
    cache_dir = os.getenv("COGITO_HOME")
    os.environ["HF_HOME"] = cache_dir

    try:
        if model_path.startswith("gs://"):
            return download_gcp_model(model_path, cache_dir)
        return download_huggingface_model(model_path, cache_dir)
    except Exception as e:
        raise ModelDownloadError(model_path, e)


def create_routes_semaphores(config: ConfigFile) -> Dict[str, asyncio.Semaphore]:
    semaphores = {}
    semaphores[config.cogito.get_predictor] = asyncio.Semaphore(
        config.cogito.get_server_threads
    )

    return semaphores


# Dependencia para limitar la concurrencia
async def limit_concurrent_requests(semaphore: asyncio.Semaphore):
    await semaphore.acquire()  # Bloquea si se alcanzó el límite

    try:
        yield  # Ejecuta la lógica de la ruta
    finally:
        semaphore.release()  # Libera el semáforo al finalizar


@contextmanager
def readiness_context(readiness_file: str) -> None:
    full_readiness_file = os.path.expandvars(os.path.expanduser(readiness_file))
    folder = os.path.dirname(full_readiness_file)
    os.makedirs(folder, exist_ok=True)

    with open(full_readiness_file, "w") as f:
        f.write("ready")
    yield
    os.remove(full_readiness_file)
