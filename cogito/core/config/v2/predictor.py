from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from cogito.core.config.v2.batching import BatchingConfig
from cogito.core.config.v2.resources import ResourcesConfig
from cogito.core.config.v2.route import RouteConfig


class PredictorConfig(BaseModel):
    """
    Predictor configuration.

    The predictor class is declared as `class` in the configuration file.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    class_path: str = Field(alias="class")
    route: Optional[RouteConfig] = None
    threads: Optional[int] = 1
    resources: ResourcesConfig = Field(default_factory=ResourcesConfig)
    batching: Optional[BatchingConfig] = None

    @property
    def get_class(self) -> str:
        return self.class_path

    @property
    def get_route(self) -> Optional[RouteConfig]:
        return self.route

    @property
    def get_route_path(self) -> Optional[str]:
        return self.route.path if self.route else None

    @property
    def get_threads(self) -> int:
        return self.threads

    @property
    def get_resources(self) -> ResourcesConfig:
        return self.resources

    @property
    def get_batching(self) -> Optional[BatchingConfig]:
        return self.batching
