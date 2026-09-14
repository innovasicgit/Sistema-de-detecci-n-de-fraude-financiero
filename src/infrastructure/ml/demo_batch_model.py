from typing import Sequence


CLASSES = (
    "Normal",
    "Pitufeo",
    "Redondeo de cifras",
    "Fraude de nómina",
    "Registro de pasivos como ingresos",
    "Uso de cuentas inexistentes",
)


class DemoBatchModel:
    """Modelo falso para probar la interfaz sin cargar un pipeline real."""

    def predict(self, rows: Sequence[object]) -> list[str]:
        # Recorre las clases de forma determinista para facilitar pruebas de presentación.
        predictions = []
        for index, _ in enumerate(rows):
            predictions.append(CLASSES[index % len(CLASSES)])
        return predictions


class DemoBatchModelLoader:
    """Loader compatible con el contrato batch que devuelve el modelo demo."""

    def load(self, model_name: str) -> DemoBatchModel:
        return DemoBatchModel()
