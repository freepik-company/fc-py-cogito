from typing import Optional

from cogito.core.config.v1.server import ServerConfig as v1
from cogito.core.config.v2.fastapi import FastAPIConfig
from cogito.core.config.v2.route import RouteConfig


class ServerConfig(v1):
    """
    Server configuration. The routes are declared by each predictor.
    """

    fastapi: FastAPIConfig
    route: Optional[RouteConfig] = None

    @classmethod
    def default(cls):
        config = super().default()

        return __class__(
            name=config.name,
            description=config.description,
            version=config.version,
            fastapi=FastAPIConfig.default(),
            cache_dir=config.cache_dir,
            threads=config.threads,
            readiness_file=config.readiness_file,
            return_input_on_response=config.return_input_on_response,
            return_traceback_on_response=config.return_traceback_on_response,
        )
