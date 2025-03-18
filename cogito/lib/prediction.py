from cogito.core.config.file import build_config_file
from cogito.core.utils import (
    create_request_model,
    get_predictor_handler_return_type,
    instance_class,
    wrap_handler,
)
from cogito.core.exceptions import NoSetupMethodError


# SDK Predictor class
class Predict:

    # Initialize predictor
    def __init__(self, config_path):
        # Build config
        config = build_config_file(config_path)

        # Get predictor path and instance
        self.predictor_path = config.cogito.get_predictor
        self.predictor_instance = instance_class(config.cogito.get_predictor)

        # Create input model from payload
        _, self.input_model_class = create_request_model(
            self.predictor_path, self.predictor_instance.predict
        )

        # Get response model type
        self.response_model = get_predictor_handler_return_type(self.predictor_instance)

    # Setup predictor calling setup method in the user's code
    def setup(self):
        try:
            if hasattr(self.predictor_instance, "setup") and callable(
                getattr(self.predictor_instance, "setup")
            ):
                self.predictor_instance.setup()
            else:
                raise NoSetupMethodError(self.predictor_instance.__class__.__name__)
        except Exception as e:
            raise Exception(f"Error setting up the predictor: {e}")

    # Run predictor using the input model in the user's code
    def run(self, payload_data: dict) -> dict:
        input_model = self.input_model_class(**payload_data)

        # Wrap handler with response model
        handler = wrap_handler(
            descriptor=self.predictor_path,
            original_handler=self.predictor_instance.predict,
            response_model=self.response_model,
        )

        # Call handler with input model
        response = handler(input_model)

        # Print response in JSON format
        return response
