from typing import Optional
from cogito.core.config.v0.server import ServerConfig as v0
from cogito.core.config.v1.route import RouteConfig


class ServerConfig(v0):
    """
    Server configuration.
    """

    route: Optional[RouteConfig]

    @classmethod
    def default(cls):
        config = super().default()

        return __class__(
            name=config.name,
            description=config.description,
            version=config.version,
            fastapi=config.fastapi,
            route=RouteConfig.default(),
            cache_dir=config.cache_dir,
            threads=config.threads,
            readiness_file=config.readiness_file,
        )
