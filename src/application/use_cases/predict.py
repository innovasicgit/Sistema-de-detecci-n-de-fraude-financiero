from dataclasses import dataclass
from typing import Sequence
from src.domain.entities.prediction import Prediction
from src.domain.ports.model_loader import ModelLoader


@dataclass(frozen=True)
class PredictInput:
    """Entrada de una predicción individual en formato de características."""

    features: Sequence[float]


class PredictUseCase:
    """Caso de uso pequeño que demuestra la inferencia individual desacoplada."""

    def __init__(self, model_loader: ModelLoader) -> None:
        self._model_loader = model_loader

    def execute(self, request: PredictInput) -> Prediction:
        # Una lista vacía no representa un registro válido para el modelo.
        if not request.features:
            raise ValueError("Se requiere al menos una característica.")

        # El loader abstrae si el modelo proviene de joblib, una API u otra infraestructura.
        model = self._model_loader.load()
        value = model.predict([list(request.features)])[0]
        return Prediction(value=value)
