from typing import List, Optional
from pydantic import BaseModel


class RouteConfig(BaseModel):
    """
    Route configuration.
    """

    name: str
    description: Optional[str] = None
    path: str
    predictor: Optional[str] = None
    tags: List[str] = List

    @classmethod
    def default(cls):
        return __class__(
            name="Predict",
            description="Make a single prediction",
            path="/v1/predict",
            tags=["predict"],
            predictor="predict:Predictor",
        )
