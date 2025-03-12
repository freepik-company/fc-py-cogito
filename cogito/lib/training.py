from cogito.lib.common import _config_file_path, _get_instance_class


def setup(config_path) -> None:
    """
    Setup the training process
    """

    config = _config_file_path(config_path)
    trainer = _get_instance_class(config.cogito.get_trainer)

    # Run the setup
    try:
        trainer.setup()
    except Exception as e:
        raise Exception(f"Error setting up the trainer: {e}")


def run(config_path, payload_data):
    """
    Train a model using the payload data
    """

    config = _config_file_path(config_path)
    trainer = _get_instance_class(config.cogito.get_trainer)

    # Call train method with payload data
    try:
        result = trainer.train(**payload_data)
    except Exception as e:
        raise Exception(f"Error training the model: {e}")

    return result
