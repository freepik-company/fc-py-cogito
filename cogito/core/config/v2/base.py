from copy import deepcopy
from typing import List

from pydantic import Field

from cogito.core.config.v1.base import CogitoConfig as v1

from cogito.core.config.v2.fastapi import FastAPIConfig
from cogito.core.config.v2.predictor import PredictorConfig
from cogito.core.config.v2.resources import ResourcesConfig
from cogito.core.config.v2.route import RouteConfig
from cogito.core.config.v2.server import ServerConfig


class CogitoConfig(v1):
    """
    Cogito configuration.
    """

    server: ServerConfig
    predictors: List[PredictorConfig] = Field(min_length=1)
    # v1's single predictor, replaced by predictors: kept off the v2 file.
    predictor: str = Field("", exclude=True)

    @classmethod
    def default(cls):
        return cls(
            server=ServerConfig.default(),
            trainer="train:Trainer",
            predictors=[
                PredictorConfig(
                    name="predictor",
                    class_path="predict:Predictor",
                    route=RouteConfig(
                        name="Predict",
                        description="Make a single prediction",
                        path="/v1/predict",
                        tags=["predict"],
                    ),
                )
            ],
        )

    @classmethod
    def get_server_config_class(cls):
        return ServerConfig

    @classmethod
    def get_fastapi_config_class(cls):
        return FastAPIConfig

    @classmethod
    def get_route_config_class(cls):
        return RouteConfig

    @property
    def get_predictors(self) -> List[PredictorConfig]:
        return self.predictors

    @property
    def get_predictor_names(self) -> List[str]:
        return [predictor.name for predictor in self.predictors]

    def get_predictor_config(self, name: str) -> PredictorConfig:
        for predictor in self.predictors:
            if predictor.name == name:
                return predictor
        raise KeyError(
            f"Unknown predictor '{name}', available: {self.get_predictor_names}"
        )

    @property
    def get_predictor(self) -> str:
        if len(self.predictors) != 1:
            raise ValueError(
                f"get_predictor needs exactly one predictor, got "
                f"{len(self.predictors)}: use get_predictors instead"
            )
        return self.predictors[0].class_path

    def upgrade(self, version: int, config):
        """Upgrade the configuration to a newer version."""

        if config is None or version >= 2:
            return config

        # Work on a copy: the v1 upgrade mutates the configuration it receives
        previous_config = deepcopy(config)
        if version < 1:
            previous_config = v1.default().upgrade(version, previous_config)

        # The v1 single predictor becomes the only item of predictors
        route = previous_config.server.route
        if route is not None:
            route = RouteConfig(
                name=route.name,
                description=route.description,
                path=route.path,
                tags=route.tags if isinstance(route.tags, list) else [],
            )

        return __class__(
            server=ServerConfig(**previous_config.server.model_dump(exclude={"route"})),
            trainer=previous_config.trainer,
            predictors=[
                PredictorConfig(
                    name="predictor",
                    class_path=previous_config.predictor,
                    route=route,
                    threads=previous_config.server.threads or 1,
                    resources=ResourcesConfig.default(),
                )
            ],
        )
