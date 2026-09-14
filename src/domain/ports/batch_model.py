from typing import Protocol, Sequence


class BatchPredictiveModel(Protocol):
    """Contrato mínimo para un modelo que procesa varios registros."""

    def predict(self, rows: Sequence[object]) -> Sequence[str]:
        ...


class BatchModelLoader(Protocol):
    """Contrato que permite cargar un modelo batch por su nombre visible."""

    def load(self, model_name: str) -> BatchPredictiveModel:
        ...
