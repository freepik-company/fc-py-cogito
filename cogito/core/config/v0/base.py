from abc import ABC, abstractmethod
from typing import Type

from cogito.core.config.v0.server import ServerConfig
from cogito.core.config.v0.route import RouteConfig
from cogito.core.config.v0.fastapi import FastAPIConfig

from pydantic import BaseModel


class ConfigInterface(ABC):
    """
    Interface for Cogito configuration classes.
    """

    @abstractmethod
    def upgrade(self):
        """Upgrade the configuration to a newer version."""
        pass

    @abstractmethod
    def get_cogito_config_class(self) -> Type["CogitoConfig"]:
        """Get the CogitoConfig class."""
        pass

    @abstractmethod
    def get_server_config_class(self) -> Type[ServerConfig]:
        """Get the ServerConfig class."""
        pass

    @abstractmethod
    def get_fastapi_config_class(self) -> Type[FastAPIConfig]:
        """Get the FastAPIConfig class."""
        pass

    @abstractmethod
    def get_route_config_class(self) -> Type[RouteConfig]:
        """Get the RouteConfig class."""
        pass

    @abstractmethod
    def get_route_path(self) -> str:
        """Get the route path."""
        pass

    @abstractmethod
    def get_server_threads(self) -> int:
        """Get the server threads."""
        pass

    @abstractmethod
    def get_server_name(self) -> str:
        """Get the server name."""
        pass

    @abstractmethod
    def get_server_description(self) -> str:
        """Get the server description."""
        pass

    @abstractmethod
    def get_server_version(self) -> str:
        """Get the server version."""
        pass

    @abstractmethod
    def get_server_readiness_file(self) -> str:
        """Get the server readiness file."""
        pass

    @abstractmethod
    def get_server_cache_dir(self) -> str:
        """Get the server cache directory."""
        pass

    @abstractmethod
    def get_fastapi(self) -> FastAPIConfig:
        """Get the fastapi configuration."""
        pass

    @abstractmethod
    def get_fastapi_access_log(self) -> bool:
        """Get the fastapi access log."""
        pass

    @abstractmethod
    def get_fastapi_debug(self) -> str:
        """Get the fastapi debug."""
        pass

    @abstractmethod
    def get_fastapi_host(self) -> str:
        """Get the fastapi host."""
        pass

    @abstractmethod
    def get_fastapi_port(self) -> int:
        """Get the fastapi port."""
        pass

    @abstractmethod
    def get_predictor(self) -> str:
        """Get the predictor."""
        pass

    def upgrade(self):
        """Upgrade the configuration to a newer version."""
        pass


class CogitoConfig(BaseModel, ConfigInterface):
    """
    Cogito configuration.
    """

    server: ServerConfig
    trainer: str = ""

    @classmethod
    def default(cls):
        return cls(server=ServerConfig.default(), trainer="train:Trainer")

    @classmethod
    def get_cogito_config_class(cls) -> Type["CogitoConfig"]:
        """Get the CogitoConfig class."""
        return cls

    @classmethod
    def get_server_config_class(cls) -> Type[ServerConfig]:
        """Get the ServerConfig class."""
        return ServerConfig

    @classmethod
    def get_fastapi_config_class(cls) -> Type[FastAPIConfig]:
        """Get the FastAPIConfig class."""
        return FastAPIConfig

    @classmethod
    def get_route_config_class(cls) -> Type[RouteConfig]:
        """Get the RouteConfig class."""
        return RouteConfig

    @property
    def get_predictor(self) -> str:
        return self.server.route.predictor

    @property
    def get_trainer(self) -> str:
        return self.trainer

    @property
    def get_route(self) -> str:
        return self.server.route

    @property
    def get_route_path(self) -> str:
        return self.server.route.path

    @property
    def get_route_name(self) -> str:
        return self.server.route.name

    @property
    def get_route_description(self) -> str:
        return self.server.route.description

    @property
    def get_route_tags(self) -> list[str]:
        return self.server.route.tags

    @property
    def get_server_threads(self) -> int:
        return self.server.threads

    @property
    def get_server_name(self) -> str:
        return self.server.name

    @property
    def get_server_description(self) -> str:
        return self.server.description

    @property
    def get_server_version(self) -> str:
        return self.server.version

    @property
    def get_server_readiness_file(self) -> str:
        return self.server.readiness_file

    @property
    def get_server_cache_dir(self) -> str:
        return self.server.cache_dir

    @property
    def get_fastapi(self) -> FastAPIConfig:
        return self.server.fastapi

    @property
    def get_fastapi_access_log(self) -> bool:
        return self.server.fastapi.access_log

    @property
    def get_fastapi_debug(self) -> str:
        return self.server.fastapi.debug

    @property
    def get_fastapi_host(self) -> str:
        return self.server.fastapi.host

    @property
    def get_fastapi_port(self) -> int:
        return self.server.fastapi.port

    def upgrade(self, version: int, config):
        """Upgrade the configuration to a newer version."""

        # No upgrade needed for v0
        self = config
        return self
