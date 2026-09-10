from typing import Any, Protocol


class PredictiveModel(Protocol):
    def predict(self, features: list[list[float]]) -> list[Any]:
        ...


class ModelLoader(Protocol):
    def load(self) -> PredictiveModel:
        ...
