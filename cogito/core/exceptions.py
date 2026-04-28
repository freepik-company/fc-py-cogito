class ConfigFileNotFoundError(Exception):
    def __init__(self, file_path: str):
        super().__init__(f"Config file not found: {file_path}")


class SetupError(Exception):
    def __init__(self, predictor: str, error: Exception):
        super().__init__(f"Unable to setup predictor {predictor}: {error}")


class InvalidHandlerSignature(Exception):
    def __init__(self, class_name: str):
        super().__init__(class_name)


class ModelDownloadError(Exception):
    def __init__(self, model_path: str, error: Exception):
        super().__init__(f"Unable to download model {model_path}: {error}")


class NoThreadsAvailableError(Exception):
    def __init__(self, class_name: str):
        super().__init__(f"No threads available for {class_name}")


class BadRequestError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NoSetupMethodError(Exception):
    def __init__(self, class_name: str):
        super().__init__(f"No setup method found for {class_name}")


class PredictorSetupError(Exception):
    def __init__(self, error: Exception):
        super().__init__(f"Error setting up the predictor: {error}")


class TrainerSetupError(Exception):
    def __init__(self, error: Exception):
        super().__init__(f"Error setting up the trainer: {error}")


class TrainerRunError(Exception):
    def __init__(self, error: Exception):
        super().__init__(f"Error training the model: {error}")
