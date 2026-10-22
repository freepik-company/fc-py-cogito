"""V2 configuration models"""

from cogito.core.config.v2.base import CogitoConfig
from cogito.core.config.v2.batching import BatchingConfig
from cogito.core.config.v2.fastapi import FastAPIConfig
from cogito.core.config.v2.predictor import PredictorConfig
from cogito.core.config.v2.resources import ResourcesConfig
from cogito.core.config.v2.route import RouteConfig
from cogito.core.config.v2.server import ServerConfig

__all__ = [
    "BatchingConfig",
    "CogitoConfig",
    "FastAPIConfig",
    "PredictorConfig",
    "ResourcesConfig",
    "RouteConfig",
    "ServerConfig",
]
