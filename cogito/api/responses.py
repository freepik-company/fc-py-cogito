from typing import Any

from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ResultResponse(BaseModel):
    inference_time_seconds: float
    input: dict[str, Any]
    result: Any


class ErrorResponse(BaseModel):
    message: str

    def to_json_response(self) -> JSONResponse:
        return JSONResponse(status_code=500, content=self.model_dump())


class BadRequestResponse(BaseModel):
    message: str

    def to_json_response(self) -> JSONResponse:
        return JSONResponse(status_code=400, content=self.model_dump())
