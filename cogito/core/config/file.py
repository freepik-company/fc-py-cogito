import importlib
import os
from pathlib import Path
import sys
from typing import Dict, Optional, Type

import yaml
from pydantic import BaseModel

from cogito.core.exceptions import ConfigFileNotFoundError


# Constants
DEFAULT_CONFIG_VERSION = 1
NO_CONFIG_VERSION = 0

# Configuration version mapping
CONFIG_VERSION_MAP: Dict[int, str] = {
    0: "cogito.core.config.v0.base.CogitoConfig",
    1: "cogito.core.config.v1.base.CogitoConfig",
}


class ConfigFile(BaseModel):
    """
    Configuration file.
    """

    config_version: int = DEFAULT_CONFIG_VERSION
    cogito: Optional[BaseModel] = None

    @classmethod
    def latest_config_version(self) -> int:
        return DEFAULT_CONFIG_VERSION

    @classmethod
    def _get_config_class(cls, version: int) -> Type[BaseModel]:
        """Get the configuration class for the specified version."""
        if version not in CONFIG_VERSION_MAP:
            raise ValueError(f"Unsupported config version: {version}")

        module_path, class_name = CONFIG_VERSION_MAP[version].rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def get_latest_config_version_class(self) -> str:
        """Get the latest version of the configuration."""
        latest = max(CONFIG_VERSION_MAP.keys())
        return CONFIG_VERSION_MAP[latest]

    @classmethod
    def default(cls):
        config_class = cls._get_config_class(DEFAULT_CONFIG_VERSION)
        cogito_config = config_class.default()
        return cls(cogito=cogito_config, config_version=DEFAULT_CONFIG_VERSION)

    @classmethod
    def exists(cls, file_path: str) -> bool:
        return Path(file_path).exists()

    @classmethod
    def load_from_file(cls, file_path: str) -> "ConfigFile":
        try:
            with open(file_path, "r") as file:
                yaml_data = yaml.safe_load(file)

                if "config_version" not in yaml_data:
                    yaml_data["config_version"] = NO_CONFIG_VERSION

                cogito_data = yaml_data.get("cogito", {})
                config_version = yaml_data["config_version"]

                config_class = cls._get_config_class(config_version)
                cogito_config = config_class(**cogito_data)
                yaml_data["cogito"] = cogito_config

                return cls(**yaml_data)
        except FileNotFoundError:
            raise ConfigFileNotFoundError(file_path)
        except Exception as e:
            raise ValueError(f"Error loading configuration file {file_path}: {e}")

    def save_to_file(self, file_path: str) -> None:
        """Save the configuration to a file."""

        # Prepare the output dictionary
        output_dict = {"config_version": self.config_version}
        if self.cogito:
            cogito_dict = self.cogito.model_dump(exclude_none=True)
            output_dict["cogito"] = cogito_dict

        # Save the output dictionary to the file
        try:
            path = Path(file_path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as file:
                yaml.dump(output_dict, file, default_flow_style=False)
        except Exception as e:
            raise ValueError(f"Error saving configuration file {file_path}: {e}")

    def config_version(self) -> int:
        return self.config_version

    def upgrade(self) -> None:
        """Upgrade the configuration to a newer version."""

        if self.cogito is None:
            raise ValueError("Configuration is not loaded")

        # Load default config for the latest version
        latest_config_class = self._get_config_class(DEFAULT_CONFIG_VERSION)
        latest_config = latest_config_class.default()

        # Upgrade the configuration
        self.cogito = latest_config.upgrade(self.config_version, self.cogito)
        self.config_version = DEFAULT_CONFIG_VERSION

        return self


def build_config_file(config_path: str) -> ConfigFile:
    """
    Get the path to the configuration file
    """
    app_dir = os.path.dirname(os.path.abspath(config_path))
    sys.path.insert(0, app_dir)

    try:
        config = ConfigFile.load_from_file(f"{config_path}")
    except ConfigFileNotFoundError:
        raise

    return config
