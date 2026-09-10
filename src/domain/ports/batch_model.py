from typing import Protocol, Sequence


class BatchPredictiveModel(Protocol):
    def predict(self, rows: Sequence[object]) -> Sequence[str]:
        ...


class BatchModelLoader(Protocol):
    def load(self) -> BatchPredictiveModel:
        ...
