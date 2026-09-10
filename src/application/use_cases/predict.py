from dataclasses import dataclass
from typing import Sequence

from src.domain.entities.prediction import Prediction
from src.domain.ports.model_loader import ModelLoader


@dataclass(frozen=True)
class PredictInput:
    features: Sequence[float]


class PredictUseCase:
    def __init__(self, model_loader: ModelLoader) -> None:
        self._model_loader = model_loader

    def execute(self, request: PredictInput) -> Prediction:
        if not request.features:
            raise ValueError("Se requiere al menos una característica.")

        model = self._model_loader.load()
        value = model.predict([list(request.features)])[0]
        return Prediction(value=value)
