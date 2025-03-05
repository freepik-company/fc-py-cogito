import asyncio
import json
import os
import sys

from cogito.core.config import ConfigFile
from cogito.core.exceptions import ConfigFileNotFoundError
from cogito.core.utils import (
    create_request_model,
    get_predictor_handler_return_type,
    instance_class,
    wrap_handler,
)


def prediction(config_path, payload_data) -> dict:
    """
    Predict a model using the payload data
    """

    app_dir = os.path.dirname(os.path.abspath(config_path))

    try:
        config = ConfigFile.load_from_file(f"{config_path}")
    except ConfigFileNotFoundError:
        raise ConfigFileNotFoundError(
            "No configuration file found. Please initialize the project first."
        )

    # Load predictor instance using the path to the cogito.yaml file
    sys.path.insert(0, app_dir)
    predictor = config.cogito.server.route.predictor
    predictor_instance = instance_class(predictor)

    # Run setup method asynchronously
    asyncio.run(predictor_instance.setup())

    # Create input model from payload
    _, input_model_class = create_request_model(predictor, predictor_instance.predict)
    input_model = input_model_class(**payload_data)

    # Get response model type
    response_model = get_predictor_handler_return_type(predictor_instance)

    # Wrap handler with response model
    handler = wrap_handler(
        descriptor=predictor,
        original_handler=predictor_instance.predict,
        response_model=response_model,
    )

    # Call handler with input model
    response = handler(input_model)

    # Print response in JSON format
    return response
