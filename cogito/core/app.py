import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Union

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from cogito.api.handlers import (
    health_check_handler,
    metrics_handler,
)
from cogito.api.responses import (
    ErrorResponse,
    BadRequestResponse,
)
from cogito.core.config import ConfigFile
from cogito.core.exceptioin_handlers import (
    bad_request_exception_handler,
    too_many_requests_exception_handler,
    validation_exception_handler,
)
from cogito.core.exceptions import (
    BadRequestError,
    ConfigFileNotFoundError,
    NoThreadsAvailableError,
    SetupError,
)
from cogito.core.logging import get_logger
from cogito.core.models import BasePredictor
from cogito.core.utils import (
    create_routes_semaphores,
    get_predictor_handler_return_type,
    instance_class,
    wrap_handler,
    readiness_context,
)


class Application:
    _logger: logging.Logger
    ready: bool

    def __init__(
        self,
        config_file_path: str = "./cogito.yaml",
        logger: Union[Any, logging.Logger] = None,
    ):

        self._logger = logger or Application._get_default_logger()

        try:
            self.config = ConfigFile.load_from_file(os.path.join(f"{config_file_path}"))
        except ConfigFileNotFoundError as e:
            self._logger.warning(
                "config file does not exist. Using default configuration.",
                extra={"error": str(e), "config_file_path": config_file_path},
            )
            self.config = ConfigFile.default()

        if self.config.cogito.get_server_cache_dir:
            os.environ["HF_HOME"] = self.config.cogito.get_server_cache_dir
            os.environ["COGITO_HOME"] = self.config.cogito.get_server_cache_dir
        else:
            os.environ["HF_HOME"] = os.path.expanduser("/.cogito/models")
            os.environ["COGITO_HOME"] = os.path.expanduser("/.cogito/models")

        @asynccontextmanager
        async def lifespan(app: FastAPI):

            try:
                await self.setup(app)
            except SetupError as e:
                self._logger.critical(
                    "Unable to start application",
                    extra={"error": e},
                )
                sys.exit(1)

            with readiness_context(self.config.cogito.get_server_readiness_file):
                yield

        self.app = FastAPI(
            title=self.config.cogito.get_server_name,
            version=self.config.cogito.get_server_version,
            description=self.config.cogito.get_server_description,
            access_log=self.config.cogito.get_fastapi_access_log,
            debug=self.config.cogito.get_fastapi_debug,
            lifespan=lifespan,
        )

        # FastAPIInstrumentor.instrument_app(self.app, excluded_urls=",".join(
        #        ["/health-check", "/metrics", "/docs", "/openapi.json"]))

        self.app.logger = self._logger

        self._set_default_routes()

        map_route_to_model: Dict[str, str] = {}
        self.map_model_to_instance: Dict[str, BasePredictor] = {}
        semaphores = create_routes_semaphores(self.config)

        route = self.config.cogito.get_route
        route_path = self.config.cogito.get_route_path
        predictor_string = self.config.cogito.get_predictor

        self._logger.info("Adding route", extra={"route": route})
        map_route_to_model[route_path] = predictor_string
        if predictor_string not in self.map_model_to_instance:
            predictor = instance_class(predictor_string)
            self.map_model_to_instance[predictor_string] = predictor
        else:
            self._logger.info(
                "Predictor class already loaded",
                extra={"predictor": predictor_string},
            )

        model = self.map_model_to_instance.get(predictor_string)
        response_model = get_predictor_handler_return_type(model)

        handler = wrap_handler(
            descriptor=predictor_string,
            original_handler=getattr(
                self.map_model_to_instance.get(predictor_string), "predict"
            ),
            semaphore=semaphores[predictor_string],
            response_model=response_model,
        )

        self.app.add_api_route(
            route_path,
            handler,
            methods=["POST"],
            name=self.config.cogito.get_route_name,
            description=self.config.cogito.get_route_description,
            tags=self.config.cogito.get_route_tags,
            response_model=response_model,
            responses={
                500: {"model": ErrorResponse},
                400: {"model": BadRequestResponse},
            },
        )

        self.app.add_exception_handler(BadRequestError, bad_request_exception_handler)
        self.app.add_exception_handler(
            RequestValidationError, validation_exception_handler
        )
        self.app.add_exception_handler(
            NoThreadsAvailableError, too_many_requests_exception_handler
        )

    def _set_default_routes(self) -> None:
        """Include default routes"""
        self.app.add_api_route(
            "/health-check",
            health_check_handler,
            methods=["GET"],
            name="health_check",
            description="Health check endpoint",
            tags=["health"],
        )

        self.app.add_api_route(
            "/metrics",
            metrics_handler,
            methods=["GET"],
            name="metrics",
            description="Metrics endpoint",
            tags=["metrics"],
        )

    async def setup(self, app: FastAPI):
        self._logger.info("Setting up application", extra={})
        for predictor in self.map_model_to_instance.values():
            try:
                self._logger.debug(
                    "Setting up predictor",
                    extra={"predictor": predictor.__class__.__name__},
                )
                # if is courutine
                if asyncio.iscoroutinefunction(predictor.setup):
                    await predictor.setup()
                else:
                    predictor.setup()
            except Exception as e:
                self._logger.critical(
                    "Unable to setting up predictor",
                    extra={"predictor": predictor.__class__.__name__, "error": e},
                )
                raise SetupError(predictor.__class__.__name__, e)

    def run(self):
        uvicorn.run(
            self.app,
            host=self.config.cogito.get_fastapi_host,
            port=self.config.cogito.get_fastapi_port,
        )

    @classmethod
    def _get_default_logger(cls):
        return get_logger("cogito.app")
