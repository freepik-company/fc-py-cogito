import asyncio
import os
import sys

from cogito.core.config import ConfigFile
from cogito.core.exceptions import ConfigFileNotFoundError
from cogito.core.utils import instance_class


def training(config_path, payload_data):
    """
    Train a model using the payload data
    """
    app_dir = os.path.dirname(os.path.abspath(config_path))
    sys.path.insert(0, app_dir)

    try:
        config = ConfigFile.load_from_file(f"{config_path}")
    except ConfigFileNotFoundError:
        raise ConfigFileNotFoundError(
            "No configuration file found. Please initialize the project first."
        )

    if config.cogito.trainer == "":
        raise ValueError("No trainer specified in the configuration file.")

    # Load training instance using the path to the cogito.yaml file
    trainer = instance_class(config.cogito.trainer)

    # Run setup method asynchronously
    asyncio.run(trainer.setup())

    # Call train method with payload data
    result = trainer.train(**payload_data)

    return result
