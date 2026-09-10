from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Prediction:
    value: Any
