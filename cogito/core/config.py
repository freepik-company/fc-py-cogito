from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel

from cogito.core.exceptions import ConfigFileNotFoundError


class RouteConfig(BaseModel):
    """
    Route configuration.
    """

    name: str
    description: Optional[str] = None
    path: str
    predictor: str
    tags: List[str] = List

    @classmethod
    def default(cls):
        return cls(
            name="Predict",
            description="Make a single prediction",
            path="/v1/predict",
            predictor="predict:Predictor",
            tags=["predict"],
        )


class FastAPIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    access_log: bool = False

    @classmethod
    def default(cls):
        return cls()


class ServerConfig(BaseModel):
    """
    Server configuration.
    """

    name: str
    description: Optional[str]
    version: Optional[str] = "0.1.0"
    fastapi: FastAPIConfig
    route: Optional[RouteConfig]
    cache_dir: str = None
    threads: Optional[int] = 1
    readiness_file: str = "$HOME/readiness.lock"

    @classmethod
    def default(cls):
        return cls(
            name="Cogito ergo sum",
            description="Inference server",
            version="0.1.0",
            fastapi=FastAPIConfig.default(),
            route=RouteConfig.default(),
            cache_dir="/tmp",
            threads=1,
            readiness_file="$HOME/readiness.lock",
        )


class CogitoConfig(BaseModel):
    """
    Cogito configuration.
    """

    server: ServerConfig
    trainer: str = ""

    @classmethod
    def default(cls):
        return cls(server=ServerConfig.default(), trainer="train:Trainer")


class ConfigFile(BaseModel):
    """
    Configuration file.
    """

    cogito: CogitoConfig

    @classmethod
    def default(cls):
        return cls(cogito=CogitoConfig.default())

    @classmethod
    def exists(cls, file_path: str) -> bool:
        return Path(file_path).exists()

    @classmethod
    def load_from_file(cls, file_path: str) -> "ConfigFile":
        try:
            with open(file_path, "r") as file:
                yaml_data = yaml.safe_load(file)
            return cls(**yaml_data)
        except FileNotFoundError:
            raise ConfigFileNotFoundError(file_path)
        except Exception:
            raise ValueError(f"Error loading configuration file {file_path}")

    def save_to_file(self, file_path: str) -> None:
        path = Path(file_path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(self.model_dump(), file)
