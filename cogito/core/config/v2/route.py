from typing import List, Optional

from pydantic import Field

from cogito.core.config.v1.route import RouteConfig as v1


class RouteConfig(v1):
    """
    Route configuration of a predictor.
    """

    name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    @classmethod
    def default(cls):
        config = super().default()

        return cls(
            name=config.name,
            description=config.description,
            path=config.path,
            tags=config.tags,
        )
