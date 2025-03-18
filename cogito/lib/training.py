from cogito.core.config.file import build_config_file
from cogito.core.utils import instance_class
from cogito.core.exceptions import NoSetupMethodError


# SDK Trainer class
class Trainer:

    # Initialize trainer
    def __init__(self, config_path):
        # Build config
        config = build_config_file(config_path)

        # Get trainer path and instance
        self.config_path = config_path
        self.trainer = instance_class(config.cogito.get_trainer)

    # Setup trainer calling setup method in the user's code
    def setup(self):
        # Run setup if needed
        try:
            if hasattr(self.trainer, "setup") and callable(
                getattr(self.trainer, "setup")
            ):
                self.trainer.setup()
            else:
                raise NoSetupMethodError(self.trainer.__class__.__name__)
        except Exception as e:
            raise Exception(f"Error setting up the trainer: {e}")

    # Run training calling train method in the user's code
    def run(self, payload_data, run_setup=True):
        # Call train method with payload data
        try:
            result = self.trainer.train(**payload_data)
        except Exception as e:
            raise Exception(f"Error training the model: {e}")

        return result
