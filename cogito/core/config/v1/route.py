from typing import Optional
from cogito.core.config.v0.route import RouteConfig as v0


class RouteConfig(v0):
    """
    Route configuration.
    """

    predictor: Optional[str] = None

    @classmethod
    def default(cls):
        config = super().default()

        return __class__(
            name=config.name,
            description=config.description,
            path=config.path,
            tags=config.tags,
        )
