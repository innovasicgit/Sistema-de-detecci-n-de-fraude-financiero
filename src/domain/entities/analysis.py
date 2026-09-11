from dataclasses import dataclass
from typing import Any


FRAUD_LABELS = {
    0: "Normal",
    1: "Pitufeo",
    2: "Redondeo de cifras",
    3: "Fraude de nómina",
    4: "Registro de pasivos como ingresos",
    5: "Uso de cuentas inexistentes",
}


@dataclass(frozen=True)
class AnalysisResult:
    records: Any
    total: int
    normal: int
    alerts: int
    model_name: str

    @property
    def alert_rate(self) -> float:
        return (self.alerts / self.total * 100) if self.total else 0.0
