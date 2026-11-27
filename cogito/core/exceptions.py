from typing import Optional

from cogito.gpu import describe_memory_cap


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


class WorkerStartError(Exception):
    def __init__(self, predictor: str, detail: str):
        super().__init__(f"Unable to start worker for predictor {predictor}: {detail}")
        self.predictor = predictor
        self.detail = detail


class WorkerTimeoutError(Exception):
    def __init__(self, predictor: str, timeout: float):
        super().__init__(f"Predictor {predictor} did not respond within {timeout:g}s")
        self.predictor = predictor
        self.timeout = timeout


class WorkerPredictionError(Exception):
    def __init__(self, predictor: str, message: str, traceback: str = None):
        super().__init__(f"Predictor {predictor} failed: {message}")
        self.predictor = predictor
        self.message = message
        self.traceback = traceback


class PredictorOutOfMemoryError(WorkerPredictionError):
    """
    A predictor ran out of GPU memory, under the cap of its worker (gpu_memory_limit,
    in bytes per visible device, None without a cap).
    """

    def __init__(
        self,
        predictor: str,
        gpu_memory_limit: Optional[int],
        message: str,
        traceback: str = None,
    ):
        super().__init__(predictor, message, traceback)
        self.args = (
            f"Predictor {predictor} ran out of GPU memory "
            f"({describe_memory_cap(gpu_memory_limit)}): {message}",
        )
        self.gpu_memory_limit = gpu_memory_limit


class WorkerUnavailableError(Exception):
    def __init__(self, predictor: str, detail: str):
        super().__init__(f"Worker for predictor {predictor} is unavailable: {detail}")
        self.predictor = predictor
        self.detail = detail
