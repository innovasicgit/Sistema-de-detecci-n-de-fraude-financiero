from dataclasses import dataclass


@dataclass
class DemoModel:
    """Modelo temporal para validar el flujo antes de conectar un artefacto real."""

    def predict(self, features: list[list[float]]) -> list[float]:
        # La suma solo simula una respuesta para validar el flujo de predicción individual.
        return [sum(features[0])]


class DemoModelLoader:
    """Loader temporal utilizado por el ejemplo de predicción individual."""

    def load(self) -> DemoModel:
        return DemoModel()
