from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from cogito.core.exceptions import NoThreadsAvailableError, BadRequestError


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(
            {
                "detail": "There is an error with the request parameters.",
                "errors": exc.errors(),
                "body": exc.body,
            }
        ),
    )


async def too_many_requests_exception_handler(
    request: Request, exc: NoThreadsAvailableError
):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=jsonable_encoder(
            {
                "detail": "There are no threads available to process the request.",
            }
        ),
    )


async def bad_request_exception_handler(
    request: Request, exc: BadRequestError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(
            {
                "detail": exc.message,
            }
        ),
    )
