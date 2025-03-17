from cogito.core.config.file import build_config_file
from cogito.core.utils import instance_class


def run(config_path, payload_data, run_setup=True):
    """
    Train a model using the payload data
    """

    config = build_config_file(config_path)
    trainer = instance_class(config.cogito.get_trainer)

    # Run setup if needed
    try:
        if (
            hasattr(trainer, "setup")
            and callable(getattr(trainer, "setup"))
            and run_setup
        ):
            trainer.setup()
    except Exception as e:
        raise Exception(f"Error setting up the trainer: {e}")

    # Call train method with payload data
    try:
        result = trainer.train(**payload_data)
    except Exception as e:
        raise Exception(f"Error training the model: {e}")

    return result
