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
    def predict(self, rows: Sequence[object]) -> list[str]:
        predictions = []
        for index, _ in enumerate(rows):
            predictions.append(CLASSES[index % len(CLASSES)])
        return predictions


class DemoBatchModelLoader:
    def load(self, model_name: str) -> DemoBatchModel:
        return DemoBatchModel()
