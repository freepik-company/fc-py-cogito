import asyncio
import atexit
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection, wait
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

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
    # A Ctrl+C in a terminal reaches the whole process group. The gateway drives
    # the shutdown and stops its workers itself; a worker killed by it would be
    # taken for a crash and restarted while the gateway shuts down.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Its own process group, so the processes its predictor starts can be stopped
    # with it (see _kill_process_group). Left behind they would be orphans, and
    # keep the worker's pipes open after it is gone.
    if hasattr(os, "setpgrp"):
        os.setpgrp()
    # The pipe only tells a worker reading requests that its gateway is gone. One
    # loading its predictor, or draining requests after a stop, must notice too.
    # The pid multiprocessing recorded at spawn: os.getppid() would already be the
    # adopter if the gateway died while this worker bootstrapped.
    gateway = multiprocessing.parent_process()
    threading.Thread(
        target=_watch_gateway,
        args=(gateway.pid if gateway is not None else os.getppid(),),
        name="cogito-worker-gateway-watch",
        daemon=True,
    ).start()

    # The worker already starts with this env (see _spawn_environ); applying it
    # again keeps it in place before the predictor import however it was started.
    os.environ.update(env)

    try:
        predictor = _load_predictor(spec.class_path)
    except BaseException:
        try:
            conn.send(
                {
                    "status": "error",
                    "pid": os.getpid(),
                    "error": traceback.format_exc(),
                }
            )
        except OSError:
            # The gateway is gone already: nobody would stop what the failed load
            # may have started.
            _exit_with_process_group()
        conn.close()
        sys.exit(1)

    try:
        conn.send(
            {
                "status": "ready",
                "pid": os.getpid(),
                "env": {key: os.environ.get(key) for key in env},
                "predictor": f"{type(predictor).__module__}.{type(predictor).__qualname__}",
            }
        )
    except OSError:
        # The gateway is gone already.
        _exit_with_process_group()

    try:
        _serve(spec, predictor, conn)
    except BaseException:
        _logger.critical("Worker loop failed", exc_info=True)
        _exit_with_process_group()
    conn.close()
    _stop_own_process_group()


def _stop_own_process_group() -> None:
    """
    Stop the processes the predictor started, before the worker exits.

    The gateway does it once it reaps the worker, but a gateway that was killed
    never will: only the worker, seeing its pipe close, knows it must clean up.
    """
    if not hasattr(os, "killpg") or os.getpgrp() != os.getpid():
        return
    # The worker leads the group: it ignores the signal it sends to the rest.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.killpg(os.getpgrp(), signal.SIGTERM)


def _watch_gateway(gateway_pid: int) -> None:
    """
    Exit with the process group as soon as the gateway is gone: the worker is then
    adopted by another process.
    """
    while os.getppid() == gateway_pid:
        time.sleep(_GATEWAY_CHECK_INTERVAL)
    _exit_with_process_group()


# How often a worker checks that its gateway is still alive.
_GATEWAY_CHECK_INTERVAL = 1.0


def _exit_with_process_group() -> None:
    """
    Exit at once, together with every process the predictor started.

    For a worker whose gateway is gone: nobody can receive the responses of the
    requests still running, and nobody is left to stop what keeps running, so
    neither the requests nor processes that ignore SIGTERM are waited for.
    """
    if hasattr(os, "killpg") and os.getpgrp() == os.getpid():
        # This worker included.
        os.killpg(os.getpgrp(), signal.SIGKILL)
    os._exit(1)


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
            except (EOFError, OSError):
                # The gateway is gone, maybe halfway through a message: only an
                # explicit stop drains the requests still running.
                _exit_with_process_group()
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
            send_job = self._sender.submit(self._send, message)
        except RuntimeError:
            # The worker was stopped in the meantime.
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise WorkerUnavailableError(name, "worker is not running") from None
        send = asyncio.wrap_future(send_job, loop=loop)
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
            # Cancel the send job itself, not its asyncio wrapper: the wrapper can
            # be cancelled while the job is already running. A job still queued
            # never runs. One running or done reaches the worker, so an abandoned
            # request is cancelled there too: the single sender thread sends the
            # cancel after it, and the worker skips it unless it already started.
            # A late response is dropped by the reader.
            sent = not send_job.cancel()
            if abandoned and sent:
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
            self._fail_pending()

    def _fail_pending(self) -> None:
        """
        Fail every request still waiting, and refuse new ones.

        The reader does it on EOF. That EOF never comes while a process the
        predictor forked still holds the worker's end of the pipe, so stop() and
        terminate() do it too once the worker is gone.
        """
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
        self._fail_pending()
        self.conn.close()

    def terminate(self, timeout: float = 10.0) -> None:
        """
        Terminate the worker without asking it first, and without starting a thread.

        For interpreter exit, where Python 3.12 refuses to start threads and nobody
        waits for the requests in flight any more.
        """
        if self.process.is_alive():
            self.process.terminate()
        _join(self.process, timeout)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        _kill_process_group(self.process)
        self._sender.shutdown(wait=False, cancel_futures=True)
        self._reader.join(timeout)
        self._fail_pending()
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
    _join(process, timeout)
    if process.is_alive():
        process.terminate()
        _join(process, timeout)
    if process.is_alive():
        process.kill()
        process.join()
    _kill_process_group(process)


def _join(process: multiprocessing.process.BaseProcess, timeout: float) -> None:
    """
    process.join(timeout), without relying on the process sentinel alone.

    The sentinel is a pipe, and a process the predictor forked inherits it: it
    stays open after the worker exited, and join() would wait the whole timeout.
    """
    deadline = time.monotonic() + timeout
    while process.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        wait([process.sentinel], min(remaining, _JOIN_POLL_INTERVAL))
    process.join()


# How often a join checks whether the process exited, besides its sentinel.
_JOIN_POLL_INTERVAL = 0.05


def _kill_process_group(process: multiprocessing.process.BaseProcess) -> None:
    """
    Kill what is left of a stopped worker's process group: the processes its
    predictor started, which would otherwise outlive it holding its pipes open.

    The group id is the worker's pid. While a member is alive the group keeps that
    id from being reused, and with no member left there is nothing to kill.
    """
    if not hasattr(os, "killpg") or process.pid is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # No member left, or the worker died before creating its group.
        pass


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
    return _spawn(spec, start_timeout)


class _SpawnAborted(Exception):
    """The start of a worker was abandoned because its supervisor is stopping"""


def _spawn(
    spec: WorkerSpec, start_timeout: float, abort: Optional[Connection] = None
) -> PredictorWorker:
    """
    spawn_worker, giving up as soon as abort becomes readable.

    Loading a predictor can take minutes; a supervisor that is stopping must not
    wait for it, nor leave the half-started process behind.
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
        deadline = time.monotonic() + start_timeout
        ready = wait(
            [parent_conn] + ([abort] if abort is not None else []), start_timeout
        )
        if parent_conn not in ready and ready:
            raise _SpawnAborted()
        if not ready:
            raise WorkerStartError(
                spec.name, f"predictor not loaded after {start_timeout}s"
            )
        handshake = _recv_handshake(
            spec, parent_conn, process, abort, deadline, start_timeout
        )
    except EOFError:
        _reap(process, timeout=10.0)
        parent_conn.close()
        raise WorkerStartError(
            spec.name, f"worker exited with code {process.exitcode}"
        ) from None
    except _SpawnAborted:
        _reap(process, timeout=1.0)
        parent_conn.close()
        raise
    except BaseException:
        _reap(process, timeout=0)
        parent_conn.close()
        raise

    if handshake.get("status") != "ready":
        _reap(process, timeout=10.0)
        parent_conn.close()
        raise WorkerStartError(spec.name, handshake.get("error", "unknown error"))

    try:
        return PredictorWorker(spec, env, process, parent_conn, handshake)
    except BaseException:
        # Nothing tracks the process yet: a supervisor retrying the start would
        # pile up live workers that neither stop() nor the exit hook can see.
        _reap(process, timeout=0)
        parent_conn.close()
        raise


def _recv_handshake(
    spec: WorkerSpec,
    conn: Connection,
    process: multiprocessing.process.BaseProcess,
    abort: Optional[Connection],
    deadline: float,
    start_timeout: float,
) -> Any:
    """
    conn.recv(), still giving up on abort or at the deadline.

    wait() only says that the handshake started to arrive: a worker that froze
    halfway through sending it would block recv() for good. A guard kills it in
    that case, so recv() sees EOF.
    """
    done_reader, done_writer = _MP_CONTEXT.Pipe(duplex=False)
    outcome: List[str] = []

    def guard() -> None:
        remaining = max(deadline - time.monotonic(), 0)
        ready = wait([done_reader] + ([abort] if abort is not None else []), remaining)
        if done_reader not in ready:
            outcome.append("aborted" if ready else "timed out")
            # The whole group: a helper the predictor forked holds the pipe too,
            # and recv() would wait for it. The worker is not reaped yet, so its
            # pid still names its group.
            _kill_process_group(process)
            process.kill()

    thread = threading.Thread(
        target=guard, name=f"cogito-handshake-{spec.name}", daemon=True
    )
    thread.start()
    try:
        try:
            handshake = conn.recv()
        except (EOFError, OSError):
            # OSError when the worker died halfway through the message.
            handshake = None
    finally:
        done_writer.send(None)
        thread.join()
        done_writer.close()
        done_reader.close()
    # Decided after the guard is done: it may have killed the worker right after
    # a complete handshake arrived.
    if outcome == ["aborted"]:
        raise _SpawnAborted()
    if outcome:
        raise WorkerStartError(
            spec.name, f"predictor not loaded after {start_timeout}s"
        )
    if handshake is None:
        raise EOFError
    return handshake


@dataclass(frozen=True)
class RestartPolicy:
    """
    When a supervisor restarts a worker that exited on its own.

    The n-th consecutive restart waits initial_delay * multiplier**n seconds, capped
    at max_delay. A worker that stayed up for reset_after seconds before exiting
    starts the sequence again. After max_restarts consecutive restarts the worker
    is given up on; None never gives up.
    """

    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    reset_after: float = 60.0
    max_restarts: Optional[int] = None

    def __post_init__(self):
        if self.initial_delay < 0:
            raise ValueError(
                f"initial_delay must not be negative, got {self.initial_delay}"
            )
        if self.max_delay < self.initial_delay:
            raise ValueError(
                f"max_delay must be at least initial_delay, got {self.max_delay}"
            )
        if self.multiplier < 1:
            raise ValueError(f"multiplier must be at least 1, got {self.multiplier}")
        if self.reset_after < 0:
            raise ValueError(
                f"reset_after must not be negative, got {self.reset_after}"
            )
        if self.max_restarts is not None and self.max_restarts < 0:
            raise ValueError(
                f"max_restarts must not be negative, got {self.max_restarts}"
            )

    def delay(self, attempt: int) -> float:
        """
        Seconds to wait before the restart number attempt, counting from 0
        """
        if self.initial_delay == 0:
            # Before the power, which can overflow for a large attempt.
            return 0.0
        try:
            delay = self.initial_delay * self.multiplier**attempt
        except OverflowError:
            return self.max_delay
        return min(delay, self.max_delay)


# How often a supervisor checks that its worker is still alive, besides waking up
# on its exit.
_EXIT_CHECK_INTERVAL = 1.0


class SupervisedWorker:
    """
    Keeps the worker of one predictor running.

    A monitor thread waits for the worker process to exit. When it exits on its own,
    the requests it had in flight fail with WorkerUnavailableError (see
    PredictorWorker), and the worker is started again following the restart
    policy. Requests made while it is down fail at once instead of waiting.
    """

    def __init__(
        self,
        spec: WorkerSpec,
        policy: Optional[RestartPolicy] = None,
        start_timeout: float = 120.0,
    ):
        self.spec = spec
        self.policy = policy or RestartPolicy()
        self.start_timeout = start_timeout
        self.restarts = 0
        self._worker: Optional[PredictorWorker] = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._gave_up = False
        # Written once by stop() and never read: once readable, it wakes the
        # monitor from whatever it waits on (the worker's exit or a start).
        self._wake_reader, self._wake_writer = _MP_CONTEXT.Pipe(duplex=False)
        self._monitor: Optional[threading.Thread] = None
        self._started = False
        self._start_done = threading.Event()

    @property
    def pid(self) -> Optional[int]:
        with self._lock:
            return self._worker.pid if self._worker is not None else None

    def is_available(self) -> bool:
        with self._lock:
            return self._worker is not None and self._worker.is_alive()

    def start(self) -> None:
        """
        Start the worker and its monitor.

        Raises WorkerStartError if the first start fails: that is a broken
        predictor, not a crash, and is not retried. A stop() meanwhile abandons it.
        """
        with self._lock:
            if self._started or self._stopping.is_set():
                raise RuntimeError(
                    f"Worker for predictor {self.spec.name} already started"
                )
            self._started = True
        try:
            try:
                worker = _spawn(self.spec, self.start_timeout, abort=self._wake_reader)
            except _SpawnAborted:
                raise WorkerStartError(
                    self.spec.name, "stopped while starting"
                ) from None
            supervised = False
            try:
                with self._lock:
                    if not self._stopping.is_set():
                        monitor = threading.Thread(
                            target=self._watch,
                            args=(worker,),
                            name=f"cogito-supervisor-{self.spec.name}",
                            daemon=True,
                        )
                        # Published first, so that from here on stop() finds it.
                        self._worker, self._monitor = worker, monitor
                        try:
                            monitor.start()
                        except BaseException:
                            self._worker, self._monitor = None, None
                            raise
                        # From here on the monitor owns the worker, even if it
                        # exits and is replaced before start() returns.
                        supervised = True
            finally:
                if not supervised:
                    # Stopped meanwhile, or interrupted (no thread left, a signal
                    # handler raising): nothing else would ever stop this worker.
                    worker.terminate(timeout=1.0)
            if not supervised or self._stopping.is_set():
                raise WorkerStartError(self.spec.name, "stopped while starting")
        finally:
            # stop() waits for this before it joins the monitor and closes the pipe
            # the start waits on.
            self._start_done.set()

    async def predict(
        self,
        inputs: Union[BaseModel, Mapping[str, Any]],
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Run a prediction in the current worker, see PredictorWorker.predict.

        Raises WorkerUnavailableError at once while the worker is being restarted.
        """
        with self._lock:
            worker = self._worker
            if worker is None:
                if self._stopping.is_set() or self._monitor is None:
                    detail = "worker is not running"
                elif self._gave_up:
                    detail = "worker crashed and is no longer restarted"
                else:
                    detail = "worker is restarting"
                raise WorkerUnavailableError(self.spec.name, detail)
        return await worker.predict(inputs, timeout)

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the worker for good: it is not restarted, and a start in progress is
        abandoned. Safe to call more than once.
        """
        stopping, worker = self._begin_stop()
        if not stopping:
            return
        try:
            if worker is not None:
                try:
                    worker.stop(timeout)
                except BaseException:
                    # E.g. no thread left for the polite stop: nothing could reach
                    # this worker after it.
                    worker.terminate(timeout)
                    raise
        finally:
            self._close_wake_pipe()

    def terminate(self, timeout: float = 10.0) -> None:
        """
        stop(), terminating the worker without asking it first and without starting
        a thread: see PredictorWorker.terminate.
        """
        stopping, worker = self._begin_stop()
        if not stopping:
            return
        if worker is not None:
            worker.terminate(timeout)
        self._close_wake_pipe()

    def _begin_stop(self) -> Tuple[bool, Optional[PredictorWorker]]:
        """
        Stop restarting the worker and wait for the monitor and any start to end.

        Returns whether this call is the one stopping it, and the worker to stop.
        """
        with self._lock:
            if self._stopping.is_set():
                return False, None
            # Set under the lock: start() and the monitor check it under the same
            # lock before publishing a worker, so none is left running after this.
            self._stopping.set()
            worker, self._worker = self._worker, None
            started = self._started
        try:
            self._wake_writer.send(None)
        except (BrokenPipeError, OSError):
            pass
        if started:
            # A first start in progress sees the wake-up and gives up.
            self._start_done.wait()
        if self._monitor is not None:
            self._monitor.join()
        return True, worker

    def _close_wake_pipe(self) -> None:
        self._wake_writer.close()
        self._wake_reader.close()

    def _watch(self, worker: PredictorWorker) -> None:
        attempt = 0
        while True:
            started_at = time.monotonic()
            # The sentinel alone is not enough: a process the predictor forked
            # inherits the pipe behind it and keeps it open after the worker died.
            while not self._stopping.is_set() and worker.process.is_alive():
                wait(
                    [worker.process.sentinel, self._wake_reader],
                    _EXIT_CHECK_INTERVAL,
                )
            if self._stopping.is_set():
                return
            uptime = time.monotonic() - started_at
            with self._lock:
                if self._worker is worker:
                    self._worker = None
            # The process is gone: this only reaps it and lets the reader thread
            # fail the requests it had in flight.
            worker.stop(timeout=1.0)
            _logger.error(
                "Worker exited unexpectedly",
                extra={
                    "predictor": self.spec.name,
                    "pid": worker.pid,
                    "exitcode": worker.process.exitcode,
                    "uptime": round(uptime, 3),
                },
            )
            if uptime >= self.policy.reset_after:
                attempt = 0

            worker, attempt = self._restart(attempt)
            if worker is None:
                return

    def _restart(self, attempt: int) -> Tuple[Optional[PredictorWorker], int]:
        """
        Start the worker again, backing off between failed starts.

        Returns the new worker, or None if the supervisor stopped or gave up, and
        the number of consecutive restarts so far.
        """
        while True:
            max_restarts = self.policy.max_restarts
            if max_restarts is not None and attempt >= max_restarts:
                with self._lock:
                    self._gave_up = True
                _logger.critical(
                    "Worker is no longer restarted",
                    extra={"predictor": self.spec.name, "restarts": attempt},
                )
                return None, attempt
            delay = self.policy.delay(attempt)
            attempt += 1
            _logger.warning(
                "Restarting worker",
                extra={
                    "predictor": self.spec.name,
                    "attempt": attempt,
                    "delay": delay,
                },
            )
            if self._stopping.wait(delay):
                return None, attempt
            try:
                worker = _spawn(self.spec, self.start_timeout, abort=self._wake_reader)
            except _SpawnAborted:
                return None, attempt
            except Exception as e:
                # Anything else (e.g. no file descriptors left) is retried too: a
                # monitor that died here would leave the worker down for good.
                is_start_error = isinstance(e, WorkerStartError)
                _logger.error(
                    "Unable to restart worker",
                    extra={
                        "predictor": self.spec.name,
                        "error": e.detail if is_start_error else str(e),
                    },
                    exc_info=not is_start_error,
                )
                continue
            with self._lock:
                if self._stopping.is_set():
                    stopped = True
                else:
                    stopped = False
                    self._worker = worker
                    self.restarts += 1
            if stopped:
                # Without a thread: this may run while the interpreter exits.
                worker.terminate(timeout=1.0)
                return None, attempt
            return worker, attempt


class WorkerSupervisor:
    """
    The supervised workers of every predictor, started and stopped together.

    Once started, the workers are also stopped when the interpreter exits, so a
    gateway that returns without calling stop() (e.g. uvicorn after SIGTERM) leaves
    no worker behind.
    """

    def __init__(
        self,
        specs: Iterable[WorkerSpec],
        policy: Optional[RestartPolicy] = None,
        start_timeout: float = 120.0,
        stop_timeout: float = 10.0,
    ):
        self.stop_timeout = stop_timeout
        self._workers: Dict[str, SupervisedWorker] = {}
        for spec in specs:
            if spec.name in self._workers:
                raise ValueError(f"Duplicated predictor name: {spec.name}")
            self._workers[spec.name] = SupervisedWorker(spec, policy, start_timeout)

    def __getitem__(self, name: str) -> SupervisedWorker:
        return self._workers[name]

    def __iter__(self) -> Iterator[SupervisedWorker]:
        return iter(list(self._workers.values()))

    def start(self) -> None:
        """
        Start every worker. If one cannot start, the ones already started are
        stopped and its WorkerStartError is raised.
        """
        _register_exit_hook()
        with _RUNNING_LOCK:
            _RUNNING_SUPERVISORS.add(self)
        try:
            for worker in self._workers.values():
                worker.start()
        except BaseException:
            # Stopping a worker that never started only releases its resources.
            self.stop()
            raise

    def stop(self, timeout: Optional[float] = None) -> None:
        """
        Stop every worker, in parallel. Safe to call more than once.
        """
        with _RUNNING_LOCK:
            _RUNNING_SUPERVISORS.discard(self)
        _stop_all(
            list(self._workers.values()),
            self.stop_timeout if timeout is None else timeout,
        )

    def _terminate(self) -> None:
        for worker in self._workers.values():
            worker.terminate(self.stop_timeout)


_RUNNING_SUPERVISORS: "set[WorkerSupervisor]" = set()
_RUNNING_LOCK = threading.Lock()
_EXIT_HOOK_REGISTERED = False


def _register_exit_hook() -> None:
    """
    Terminate the workers of every supervisor still running when the interpreter
    exits.

    Without it, exit would wait forever: ThreadPoolExecutor's exit hook joins a
    sender blocked on a frozen worker, and multiprocessing's joins the non-daemon
    workers, which only exit once asked to or once their pipe closes. threading's
    exit hooks run before both: before the interpreter joins its threads, and in
    reverse order, so after the ThreadPoolExecutor hook registered on import.
    """
    global _EXIT_HOOK_REGISTERED
    with _RUNNING_LOCK:
        if _EXIT_HOOK_REGISTERED:
            return
        # threading._register_atexit is private CPython API (3.9+). Its hooks run
        # inside threading._shutdown, before it joins the non-daemon threads,
        # ThreadPoolExecutor's senders among them. A plain atexit hook only runs
        # after that join: with a send blocked on a frozen worker, exit would hang
        # before ever reaching it. The fallback still covers idle workers, so it
        # is kept, but it is not silent.
        register = getattr(threading, "_register_atexit", None)
        if register is None:
            _logger.warning(
                "threading._register_atexit is not available: workers are stopped "
                "from atexit, and exit may hang while a send to a frozen worker "
                "is blocked"
            )
            register = atexit.register
        register(_terminate_running_supervisors)
        _EXIT_HOOK_REGISTERED = True


def _terminate_running_supervisors() -> None:
    # No thread is started from here: Python 3.12 refuses them at exit.
    with _RUNNING_LOCK:
        supervisors = list(_RUNNING_SUPERVISORS)
        _RUNNING_SUPERVISORS.clear()
    for supervisor in supervisors:
        supervisor._terminate()


def _stop_all(workers: List[SupervisedWorker], timeout: float) -> None:
    threads = [
        threading.Thread(
            target=worker.stop,
            args=(timeout,),
            name=f"cogito-stop-supervisor-{worker.spec.name}",
        )
        for worker in workers
    ]
    started = []
    for worker, thread in zip(workers, threads):
        try:
            thread.start()
        except RuntimeError:
            # E.g. no threads left: stop this one here, the hard way, rather than
            # leave it running.
            worker.terminate(timeout)
        else:
            started.append(thread)
    for thread in started:
        thread.join()
