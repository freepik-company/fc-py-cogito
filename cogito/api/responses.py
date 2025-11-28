from typing import Any, Optional

from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ResultResponse(BaseModel):
    model_config = {"exclude_none": True}
    
    inference_time_seconds: float
    input: Optional[dict[str, Any]] = None
    result: Any


class ErrorResponse(BaseModel):
    message: str

    def to_json_response(self) -> JSONResponse:
        return JSONResponse(status_code=500, content=self.model_dump())


class BadRequestResponse(BaseModel):
    message: str

    def to_json_response(self) -> JSONResponse:
        return JSONResponse(status_code=400, content=self.model_dump())
