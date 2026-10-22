from typing import List

from pydantic import Field, model_validator

from cogito.core.config.v1.base import CogitoConfig as v1

from cogito.core.config.v2.fastapi import FastAPIConfig
from cogito.core.config.v2.predictor import PredictorConfig
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

    @model_validator(mode="after")
    def _validate_unique_predictors(self):
        # Predictors without a route are not compared: they have no path yet.
        names = set()
        paths = {}
        for predictor in self.predictors:
            if predictor.name in names:
                raise ValueError(
                    f"predictor names must be unique, '{predictor.name}' is "
                    f"declared more than once"
                )
            names.add(predictor.name)

            if predictor.get_route_path is None:
                continue
            # '/v1/sdxl' and '/v1/sdxl/' reach the same handler.
            path = predictor.get_route_path.rstrip("/") or "/"
            if path in paths:
                raise ValueError(
                    f"route paths must be unique, '{path}' is declared by "
                    f"predictors '{paths[path]}' and '{predictor.name}'"
                )
            paths[path] = predictor.name
        return self

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
