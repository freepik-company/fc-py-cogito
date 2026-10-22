from typing import Optional

from pydantic import BaseModel


class ResourcesConfig(BaseModel):
    """
    GPU resources of a predictor.

    gpus is the number of GPUs the predictor needs, a fraction for a shared GPU
    (0 means no GPU constraint). gpu_memory is the fraction of the device memory
    (0.5 is half of it) and gpu_compute the share of the device compute, in percent
    (50 is half of it).
    """

    gpus: float = 0
    gpu_memory: Optional[float] = None
    gpu_compute: Optional[int] = None

    @classmethod
    def default(cls):
        return cls()
