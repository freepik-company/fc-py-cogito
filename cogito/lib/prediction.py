from cogito.core.config.file import build_config_file
from cogito.core.utils import (
    create_request_model,
    get_predictor_handler_return_type,
    instance_class,
    wrap_handler,
)


def run(config_path, payload_data, run_setup=True) -> dict:
    """
    Predict a model using the payload data
    """

    config = build_config_file(config_path)
    predictor_path = config.cogito.get_predictor
    predictor_instance = instance_class(config.cogito.get_predictor)

    # Run setup if needed
    try:
        if (
            hasattr(predictor_instance, "setup")
            and callable(getattr(predictor_instance, "setup"))
            and run_setup
        ):
            predictor_instance.setup()
    except Exception as e:
        raise Exception(f"Error setting up the predictor: {e}")

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
