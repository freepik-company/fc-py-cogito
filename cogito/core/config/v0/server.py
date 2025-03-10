from typing import Optional
from cogito.core.config.v0.fastapi import FastAPIConfig
from cogito.core.config.v0.route import RouteConfig
from pydantic import BaseModel


class ServerConfig(BaseModel):
    """
    Server configuration.
    """

    name: str
    description: Optional[str]
    version: Optional[str] = "0.1.0"
    fastapi: FastAPIConfig
    cache_dir: str = None
    route: Optional[RouteConfig]
    threads: Optional[int] = 1
    readiness_file: str = "$HOME/readiness.lock"

    @classmethod
    def default(cls):
        return __class__(
            name="Cogito ergo sum",
            description="Inference server",
            version="0.1.0",
            fastapi=FastAPIConfig.default(),
            cache_dir="/tmp",
            route=RouteConfig.default(),
            threads=1,
            readiness_file="$HOME/readiness.lock",
        )
