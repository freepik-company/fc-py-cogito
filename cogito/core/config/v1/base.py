from typing import Type

from cogito.core.config.v0.base import CogitoConfig as v0

from cogito.core.config.v1.server import ServerConfig
from cogito.core.config.v1.fastapi import FastAPIConfig
from cogito.core.config.v1.route import RouteConfig


class CogitoConfig(v0):
    """
    Cogito configuration.
    """

    predictor: str = ""

    @classmethod
    def default(cls):
        return cls(
            server=ServerConfig.default(),
            trainer="train:Trainer",
            predictor="predict:Predictor",
        )

    @property
    def get_predictor(self):
        return self.predictor

    def upgrade(self, version: int, config: v0):
        """Upgrade the configuration to a newer version."""

        if config is not None and version < 1:
            previous_config = super().upgrade(version, config)

            # Copy the previous_config fields to self one by one except the server.route.predictor
            for field in previous_config.__dict__:
                if field != "server.route.predictor":
                    setattr(self, field, getattr(previous_config, field))

            # Set the predictor to the previous config
            self.predictor = previous_config.server.route.predictor
            self.server.route.predictor = None

        return self
