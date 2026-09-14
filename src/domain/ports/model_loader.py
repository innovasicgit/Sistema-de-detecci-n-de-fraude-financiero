from typing import Any, Protocol


class PredictiveModel(Protocol):
    """Contrato para modelos que reciben una matriz de características."""

    def predict(self, features: list[list[float]]) -> list[Any]:
        ...


class ModelLoader(Protocol):
    """Contrato de infraestructura para cargar un modelo individual."""

    def load(self) -> PredictiveModel:
        ...
