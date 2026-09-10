from pathlib import Path
from typing import Any

import joblib


class JoblibModelLoader:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path

    def load(self) -> Any:
        if not self._model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {self._model_path}")
        return joblib.load(self._model_path)
