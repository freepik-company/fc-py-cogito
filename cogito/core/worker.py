import asyncio
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple, Union

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from cogito.core.exceptions import (
    BadRequestError,
    WorkerPredictionError,
    WorkerStartError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)

# Workers always use the spawn start method: a forked child would inherit any CUDA
# context the parent created, and the env must be in place before the predictor
# module (and the libraries it imports) is loaded.
_MP_CONTEXT = multiprocessing.get_context("spawn")

STOP_MESSAGE = "stop"

# Messages exchanged after the start handshake. A request carries a correlation id
# that its response echoes back, so responses can arrive in any order:
#   {"type": "predict", "id": str, "inputs": dict}
#   {"type": "cancel", "id": str}   the caller gave up: skip it if not started
#   {"type": "result", "id": str, "status": "ok", "result": Any}
#   {"type": "result", "id": str, "status": "error", "error_type": str,
#    "message": str, "traceback": str}
# Responses travel as JSON text, never pickled: a pickled value may reference a
# type from the predictor module (even a str/int/float subclass), and unpickling
# it would import that module in the gateway.
PREDICT_MESSAGE = "predict"
CANCEL_MESSAGE = "cancel"
RESULT_MESSAGE = "result"


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


# error_type is the qualified name of the exception class, so an unrelated
# exception that happens to be called BadRequestError is not taken for cogito's.
BAD_REQUEST_ERROR_TYPE = _qualified_name(BadRequestError)

_logger = logging.getLogger(__name__)

# Serialises the temporary changes to os.environ made while a worker starts.
_SPAWN_ENV_LOCK = threading.Lock()


@dataclass(frozen=True)
class WorkerSpec:
    """
    What a worker process runs and the environment it runs with.

    cuda_visible_devices is the final value for CUDA_VISIBLE_DEVICES: None inherits
    the parent's value, an empty string hides every GPU (CPU-only placement).
    env holds extra resource variables, applied after the ones cogito sets.
    concurrency is how many requests the worker runs at the same time.
    """

    name: str
    class_path: str
    cuda_visible_devices: Optional[str] = None
    env: Mapping[str, str] = field(default_factory=dict)
    concurrency: int = 1

    def __post_init__(self):
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {self.concurrency}")


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
    # The worker already starts with this env (see _spawn_environ); applying it
    # again keeps it in place before the predictor import however it was started.
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

    _serve(spec, predictor, conn)
    conn.close()


class _EventLoopThread:
    """
    An event loop running in its own thread, for async predictors
    """

    def __init__(self, name: str):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.loop.run_forever()
        except BaseException:
            # A SystemExit or KeyboardInterrupt raised in a task the predictor
            # spawned escapes run_forever and stops the loop, so nothing can
            # finish the requests waiting on it. Exit the worker: the gateway sees
            # EOF and fails them with WorkerUnavailableError.
            _logger.critical(
                "Event loop of the worker stopped unexpectedly", exc_info=True
            )
            os._exit(1)

    def run(self, coroutine) -> Any:
        failed, value = asyncio.run_coroutine_threadsafe(
            _capture(coroutine), self.loop
        ).result()
        if failed:
            raise value
        return value

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


async def _capture(coroutine) -> Tuple[bool, Any]:
    """
    Await coroutine, returning what it raised instead of raising it.

    A BaseException such as SystemExit escaping a task would stop the event loop
    and leave the request waiting forever.
    """
    try:
        return False, await coroutine
    except BaseException as e:
        return True, e


def _error_response(request_id: str, error: BaseException) -> Dict[str, Any]:
    is_bad_request = isinstance(error, BadRequestError)
    try:
        message = (getattr(error, "message", None) if is_bad_request else None) or str(
            error
        )
    except BaseException:
        message = f"{_qualified_name(type(error))} (message unavailable)"
    return {
        "type": RESULT_MESSAGE,
        "id": request_id,
        "status": "error",
        "error_type": (
            BAD_REQUEST_ERROR_TYPE if is_bad_request else _qualified_name(type(error))
        ),
        "message": message,
        "traceback": traceback.format_exc(),
    }


def _run_request(call: Callable[..., Any], message: Dict[str, Any]) -> str:
    """
    Run one request and return its response as JSON text
    """
    try:
        result = jsonable_encoder(call(**message["inputs"]))
    except BaseException as e:
        # BaseException too: a SystemExit lost in the executor's future would leave
        # the request without a response.
        response = _error_response(message["id"], e)
    else:
        response = {
            "type": RESULT_MESSAGE,
            "id": message["id"],
            "status": "ok",
            "result": result,
        }
    try:
        return json.dumps(response)
    except (TypeError, ValueError) as e:
        error = _error_response(message["id"], e)
        error["message"] = f"Unable to serialise the prediction result: {e}"
        return json.dumps(error)


def _handle_request(
    call: Callable[..., Any],
    message: Dict[str, Any],
    claim: Callable[[str], bool],
    reply: Callable[[str], None],
) -> None:
    """
    Run one request, unless it was cancelled while queued, and send its response
    """
    if not claim(message["id"]):
        return
    try:
        payload = _run_request(call, message)
    except BaseException as e:
        # Nobody checks the executor's future: whatever goes wrong, answer.
        payload = json.dumps(_error_response(message["id"], e))
    reply(payload)


def _serve(spec: WorkerSpec, predictor: Any, conn: Connection) -> None:
    """
    Run prediction requests until the parent asks the worker to stop or goes away
    """
    send_lock = threading.Lock()

    def reply(response: str) -> None:
        # Connection is not thread-safe, and responses come from several threads.
        with send_lock:
            try:
                conn.send(response)
            except (BrokenPipeError, OSError):
                pass

    event_loop = None
    if inspect.iscoroutinefunction(predictor.predict):
        event_loop = _EventLoopThread(f"cogito-worker-{spec.name}-loop")

        def call(**inputs: Any) -> Any:
            return event_loop.run(predictor.predict(**inputs))

    else:
        call = predictor.predict

    # Requests submitted but not started yet, by id, and whether the caller
    # cancelled them. A request that already started runs to the end: a thread
    # cannot be interrupted.
    queued: Dict[str, bool] = {}
    queued_lock = threading.Lock()

    def claim(request_id: str) -> bool:
        with queued_lock:
            return not queued.pop(request_id, False)

    executor = ThreadPoolExecutor(
        max_workers=spec.concurrency,
        thread_name_prefix=f"cogito-worker-{spec.name}",
    )
    try:
        while True:
            try:
                message = conn.recv()
            except EOFError:
                break
            if message == STOP_MESSAGE:
                break
            if not isinstance(message, dict):
                continue
            if message.get("type") == PREDICT_MESSAGE:
                with queued_lock:
                    queued[message["id"]] = False
                executor.submit(_handle_request, call, message, claim, reply)
            elif message.get("type") == CANCEL_MESSAGE:
                with queued_lock:
                    if message.get("id") in queued:
                        queued[message["id"]] = True
    finally:
        # Requests already running still get their response.
        executor.shutdown(wait=True)
        if event_loop is not None:
            event_loop.close()


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
        self._send_lock = threading.Lock()
        # Requests are sent from this worker's own thread, never from a pool shared
        # with other workers: a send blocked on a frozen worker only holds this one.
        self._sender = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"cogito-ipc-{spec.name}-send"
        )
        # Requests waiting for a response, by correlation id. The reader thread and
        # the callers' event loops both use it.
        self._pending: Dict[str, Tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._pending_lock = threading.Lock()
        self._closed = False
        self._reader = threading.Thread(
            target=self._read_responses,
            name=f"cogito-ipc-{spec.name}",
            daemon=True,
        )
        self._reader.start()

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid

    def is_alive(self) -> bool:
        return self.process.is_alive()

    async def predict(
        self,
        inputs: Union[BaseModel, Mapping[str, Any]],
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Run a prediction in the worker and return its result as plain data.

        Raises WorkerTimeoutError if no response arrives within timeout seconds,
        BadRequestError if the predictor rejected the input, WorkerPredictionError
        if it failed, and WorkerUnavailableError if the worker is gone.
        """
        name = self.spec.name
        if isinstance(inputs, BaseModel):
            # Request models are created at runtime and cannot be pickled.
            dump = getattr(inputs, "model_dump", None) or inputs.dict
            inputs = dump()
        else:
            inputs = dict(inputs)

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request_id = uuid.uuid4().hex
        with self._pending_lock:
            if self._closed:
                raise WorkerUnavailableError(name, "worker is not running")
            self._pending[request_id] = (loop, future)

        message = {"type": PREDICT_MESSAGE, "id": request_id, "inputs": inputs}
        # The send runs on its own and a failure is forwarded to the response
        # future, so the request only ever waits on that future: a worker exit
        # fails it even while the send is still queued behind another one.
        try:
            send = loop.run_in_executor(self._sender, self._send, message)
        except RuntimeError:
            # The worker was stopped in the meantime.
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise WorkerUnavailableError(name, "worker is not running") from None
        send.add_done_callback(functools.partial(_forward_send_failure, future, name))
        abandoned = False
        try:
            try:
                # One deadline for sending the request and waiting for its response.
                response = await asyncio.wait_for(future, timeout)
            except asyncio.TimeoutError:
                abandoned = True
                raise WorkerTimeoutError(name, timeout) from None
            except asyncio.CancelledError:
                abandoned = True
                raise
        finally:
            # A send still queued never runs. One already sent is cancelled in the
            # worker, which skips it unless it already started; a late response
            # is dropped by the reader.
            send.cancel()
            if abandoned and not send.cancelled():
                self._send_in_background({"type": CANCEL_MESSAGE, "id": request_id})
            with self._pending_lock:
                self._pending.pop(request_id, None)

        if response["status"] == "ok":
            return response["result"]
        if response["error_type"] == BAD_REQUEST_ERROR_TYPE:
            raise BadRequestError(response["message"])
        raise WorkerPredictionError(name, response["message"], response["traceback"])

    def _send(self, message: Any) -> None:
        with self._send_lock:
            self.conn.send(message)

    def _send_quietly(self, message: Any) -> None:
        try:
            self._send(message)
        except (BrokenPipeError, OSError):
            pass

    def _send_in_background(self, message: Any) -> None:
        try:
            self._sender.submit(self._send_quietly, message)
        except RuntimeError:
            # The worker was stopped: there is nobody to tell.
            pass

    def _read_responses(self) -> None:
        """
        Hand each response to the request with the same correlation id
        """
        try:
            while True:
                try:
                    message = self.conn.recv()
                except (EOFError, OSError):
                    break
                try:
                    message = json.loads(message)
                except (TypeError, ValueError):
                    continue
                if not isinstance(message, dict) or message.get("type") != (
                    RESULT_MESSAGE
                ):
                    continue
                with self._pending_lock:
                    entry = self._pending.pop(message.get("id"), None)
                if entry is None:
                    _logger.debug(
                        "Dropping response for an expired request",
                        extra={"predictor": self.spec.name, "id": message.get("id")},
                    )
                    continue
                loop, future = entry
                _call_soon(loop, _set_future_result, future, message)
        finally:
            # The worker is gone: fail what is still waiting instead of hanging.
            with self._pending_lock:
                self._closed = True
                pending = list(self._pending.values())
                self._pending.clear()
            for loop, future in pending:
                error = WorkerUnavailableError(
                    self.spec.name, "worker exited before responding"
                )
                _call_soon(loop, _set_future_exception, future, error)

    def stop(self, timeout: float = 10.0) -> None:
        """
        Ask the worker to stop, and terminate it if it does not exit in time
        """
        stopper = None
        if self.process.is_alive():
            # Sent from its own thread: a worker that stopped reading can block the
            # send (or another sender holding the lock), and _reap must still run.
            stopper = threading.Thread(
                target=self._send_quietly,
                args=(STOP_MESSAGE,),
                name=f"cogito-stop-{self.spec.name}",
            )
            stopper.daemon = True
            stopper.start()
        _reap(self.process, timeout)
        # Once the worker is gone, blocked sends fail and the reader sees EOF: the
        # worker held the only other end of the pipe.
        if stopper is not None:
            stopper.join(timeout)
        self._sender.shutdown(wait=False, cancel_futures=True)
        self._reader.join(timeout)
        self.conn.close()


def _call_soon(loop: asyncio.AbstractEventLoop, callback, *args) -> None:
    try:
        loop.call_soon_threadsafe(callback, *args)
    except RuntimeError:
        # The caller's event loop is already closed: nobody is waiting.
        pass


def _forward_send_failure(
    future: asyncio.Future, predictor: str, send: asyncio.Future
) -> None:
    if send.cancelled():
        return
    error = send.exception()
    if error is None or future.done():
        return
    if isinstance(error, (BrokenPipeError, OSError)):
        error = WorkerUnavailableError(predictor, str(error))
    future.set_exception(error)


def _set_future_result(future: asyncio.Future, result: Any) -> None:
    # The request may have timed out or been cancelled in the meantime.
    if not future.done():
        future.set_result(result)


def _set_future_exception(future: asyncio.Future, error: BaseException) -> None:
    if not future.done():
        future.set_exception(error)


def _reap(process: multiprocessing.process.BaseProcess, timeout: float) -> None:
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join()


@contextmanager
def _spawn_environ(env: Dict[str, str]) -> Iterator[None]:
    """
    Apply env to this process while a worker is started, then restore it.

    A spawned interpreter copies the parent's environment when it starts, before it
    re-imports the launcher's main module and unpickles the worker arguments. Any
    CUDA initialisation in that bootstrap then already sees the worker's devices.
    """
    with _SPAWN_ENV_LOCK:
        previous = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def spawn_worker(spec: WorkerSpec, start_timeout: float = 120.0) -> PredictorWorker:
    """
    Start a worker process for a predictor and wait until the predictor is loaded.

    The worker starts with its env already applied, and the predictor is imported
    only inside the worker.
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
    with _spawn_environ(env):
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
