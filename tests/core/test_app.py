import asyncio
import json
import textwrap

from cogito.core.app import Application
from cogito.core.config.file import ConfigFile

OOM_PREDICTOR = textwrap.dedent(
    """
    from cogito.core.exceptions import PredictorOutOfMemoryError


    class Predictor:
        def setup(self):
            pass

        def predict(self, value: int) -> int:
            raise PredictorOutOfMemoryError(
                "sdxl", 4096 * 1024**2, "CUDA out of memory. Tried to allocate 2.00 GiB"
            )
    """
)


async def post(app, path, payload):
    """
    POST payload to path through the whole ASGI app (middleware and exception
    handlers included), without an HTTP client: httpx is not a dependency.
    """
    body = json.dumps(payload).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    [start] = [m for m in sent if m["type"] == "http.response.start"]
    return start["status"], b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )


def test_out_of_memory_in_a_route_is_answered_with_the_predictor_and_its_cap(
    tmp_path, monkeypatch
):
    (tmp_path / "cogito_app_oom_predictor.py").write_text(OOM_PREDICTOR)
    monkeypatch.syspath_prepend(str(tmp_path))
    config = ConfigFile.default()
    config.cogito.predictor = "cogito_app_oom_predictor:Predictor"
    config.cogito.server.cache_dir = str(tmp_path)
    config.cogito.server.readiness_file = str(tmp_path / "readiness.lock")
    config_path = tmp_path / "cogito.yaml"
    config.save_to_file(str(config_path))

    status, body = asyncio.run(
        post(Application(str(config_path)).app, "/v1/predict", {"value": 1})
    )

    assert status == 500
    assert json.loads(body) == {
        "detail": "Predictor sdxl ran out of GPU memory (cap: 4096MB per device): "
        "CUDA out of memory. Tried to allocate 2.00 GiB"
    }
