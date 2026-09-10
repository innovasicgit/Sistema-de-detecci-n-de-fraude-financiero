from dataclasses import dataclass
from typing import Any


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
