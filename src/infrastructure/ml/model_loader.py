from dataclasses import dataclass


@dataclass
class DemoModel:
    """Modelo temporal para validar el flujo antes de conectar un artefacto real."""

    def predict(self, features: list[list[float]]) -> list[float]:
        return [sum(features[0])]


class DemoModelLoader:
    def load(self) -> DemoModel:
        return DemoModel()
