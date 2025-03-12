import os
import sys
from typing import Any

from cogito.core.config.file import ConfigFile
from cogito.core.exceptions import ConfigFileNotFoundError
from cogito.core.utils import instance_class


def _config_file_path(config_path: str) -> ConfigFile:
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


def _get_instance_class(class_path: str) -> Any:
    if class_path == "":
        raise ValueError(f"No class path {class_path} specified.")

    class_instance = instance_class(class_path)

    return class_instance
