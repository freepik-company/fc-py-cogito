import asyncio
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest

from cogito._version import __version__
from cogito.core.utils import is_ready


def create_health_check_handler(readiness_file: str) -> Callable:
    """Build the /health-check handler bound to the configured readiness file."""

    async def health_check_handler(request: Request) -> JSONResponse:
        # is_ready() does blocking file I/O; run it off the event loop thread.
        if await asyncio.to_thread(is_ready, readiness_file):
            return JSONResponse({"status": "OK"})

        return JSONResponse(
            {"status": "ERROR", "message": "Service is not ready"},
            status_code=503,
        )

    return health_check_handler


async def metrics_handler(request: Request) -> Response:
    return Response(content=generate_latest(), media_type="text/plain")


async def version_handler(request: Request) -> JSONResponse:
    return JSONResponse({"version": __version__})
