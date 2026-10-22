from typing import Optional

from pydantic import BaseModel, field_validator


class ResourcesConfig(BaseModel):
    """
    GPU resources of a predictor.

    gpus is the number of GPUs the predictor needs, a fraction for a shared GPU
    (0 means no GPU constraint). gpu_memory is the fraction of the device memory
    (0.5 is half of it) and gpu_compute the share of the device compute, in percent
    (50 is half of it).

    gpu_memory and gpu_compute cap the predictor on every GPU it sees, so they
    also apply with gpus 0: the predictor is then not placed, but still capped
    on the GPUs it inherits.
    """

    gpus: float = 0
    gpu_memory: Optional[float] = None
    gpu_compute: Optional[int] = None

    @field_validator("gpus", "gpu_memory", "gpu_compute", mode="before")
    @classmethod
    def _reject_booleans(cls, value, info):
        # pydantic reads true/false as 1/0, which is never what a YAML meant here.
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a number, got {value}")
        return value

    @field_validator("gpus")
    @classmethod
    def _validate_gpus(cls, value: float) -> float:
        if value == 0 or 0 < value <= 1 or (value > 1 and value.is_integer()):
            return value
        raise ValueError(
            f"gpus must be a fraction in (0, 1] or a positive integer "
            f"(0 for no GPU), got {value}"
        )

    @field_validator("gpu_memory")
    @classmethod
    def _validate_gpu_memory(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not 0 < value <= 1:
            raise ValueError(f"gpu_memory must be a fraction in (0, 1], got {value}")
        return value

    @field_validator("gpu_compute")
    @classmethod
    def _validate_gpu_compute(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not 1 <= value <= 100:
            raise ValueError(f"gpu_compute must be a percentage in 1..100, got {value}")
        return value

    @classmethod
    def default(cls):
        return cls()
