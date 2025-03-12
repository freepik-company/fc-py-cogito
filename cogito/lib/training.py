from cogito.core.config.file import build_config_file
from cogito.core.utils import instance_class


def setup(config_path) -> None:
    """
    Setup the training process
    """

    config = build_config_file(config_path)
    trainer = instance_class(config.cogito.get_trainer)

    # Run the setup
    try:
        trainer.setup()
    except Exception as e:
        raise Exception(f"Error setting up the trainer: {e}")


def run(config_path, payload_data):
    """
    Train a model using the payload data
    """

    config = build_config_file(config_path)
    trainer = instance_class(config.cogito.get_trainer)

    # Call train method with payload data
    try:
        result = trainer.train(**payload_data)
    except Exception as e:
        raise Exception(f"Error training the model: {e}")

    return result
