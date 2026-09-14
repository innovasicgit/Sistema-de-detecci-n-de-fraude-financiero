from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelEvaluation:
    """Resultados de comparar las etiquetas reales con las predicciones del modelo."""

    report: Any
    confusion_matrix: Any
    labels: tuple[int, ...]
    total: int


FRAUD_LABELS = {
    # Los modelos devuelven códigos; la interfaz utiliza estos nombres legibles.
    0: "Normal",
    1: "Pitufeo",
    2: "Redondeo de cifras",
    3: "Fraude de nómina",
    4: "Registro de pasivos como ingresos",
    5: "Uso de cuentas inexistentes",
}


@dataclass(frozen=True)
class AnalysisResult:
    """Resumen de una inferencia operativa sobre un archivo contable."""

    records: Any
    total: int
    normal: int
    alerts: int
    model_name: str

    @property
    def alert_rate(self) -> float:
        # Un archivo vacío no debe provocar una división entre cero.
        return (self.alerts / self.total * 100) if self.total else 0.0
