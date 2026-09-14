from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Prediction:
    """Objeto de dominio que encapsula el valor de una predicción individual."""

    value: Any
