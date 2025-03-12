import time
from cogito.core.models import BaseTrainer


class Trainer(BaseTrainer):
    def setup(self):
        pass

    def train(
        self,
        model_name: str,
        epochs: int,
        learning_rate: float,
        batch_size: int,
    ):

        print(
            f"Training {model_name} for {epochs} epochs with learning rate {learning_rate} and batch size {batch_size}"
        )
        time.sleep(5)

        return "Training completed"
