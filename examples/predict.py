import logging
import time
from pydantic import BaseModel, Field

from cogito import BasePredictor


class PredictResponse(BaseModel):
    image: str
    text: str
    my_custom_variable: str


class Predictor(BasePredictor):
    def predict(
        self,
        prompt: str,
        temperature: float = Field(
            0.5,
            description="Temperature is the most important attribute for an inference",
            gt=0.0,
            lt=1.0,
        ),
    ) -> PredictResponse:
        return PredictResponse(
            image="https://example.com/image.jpg",
            text="Hello world",
            my_custom_variable=self.my_custom_variable,
        )

    def setup(self):
        time.sleep(3)
        logging.info("I'm ready")

        self.my_custom_variable = "Hello"
