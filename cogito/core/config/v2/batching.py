from typing import Optional

from pydantic import BaseModel


class BatchingConfig(BaseModel):
    """
    Dynamic batching of a predictor.
    """

    enabled: bool = False
    max_batch_size: Optional[int] = None
    max_wait_ms: Optional[int] = None

    @classmethod
    def default(cls):
        return cls()
