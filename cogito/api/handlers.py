from fastapi import Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest


async def health_check_handler(request: Request) -> JSONResponse:
    return JSONResponse({"status": "OK"})


async def metrics_handler(request: Request) -> Response:
    return Response(content=generate_latest(), media_type="text/plain")
