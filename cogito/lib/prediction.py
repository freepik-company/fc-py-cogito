from cogito.core.config.file import build_config_file
from cogito.core.utils import (
    create_request_model,
    get_predictor_handler_return_type,
    instance_class,
    wrap_handler,
)


def setup(config_path) -> None:
    """
    Setup the prediction process
    """

    config = build_config_file(config_path)
    predictor = instance_class(config.cogito.get_predictor)

    try:
        predictor.setup()
    except Exception as e:
        raise Exception(f"Error setting up the predictor: {e}")


def run(config_path, payload_data) -> dict:
    """
    Predict a model using the payload data
    """

    config = build_config_file(config_path)
    predictor_path = config.cogito.get_predictor
    predictor_instance = instance_class(config.cogito.get_predictor)

    # Create input model from payload
    _, input_model_class = create_request_model(
        predictor_path, predictor_instance.predict
    )
    input_model = input_model_class(**payload_data)

    # Get response model type
    response_model = get_predictor_handler_return_type(predictor_instance)

    # Wrap handler with response model
    handler = wrap_handler(
        descriptor=predictor_path,
        original_handler=predictor_instance.predict,
        response_model=response_model,
    )

    # Call handler with input model
    response = handler(input_model)

    # Print response in JSON format
    return response
