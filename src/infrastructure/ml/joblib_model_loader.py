from pathlib import Path
from typing import Any
import pathlib

import joblib


class JoblibModelLoader:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path

    def load(self) -> Any:
        if not self._model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {self._model_path}")
        try:
            original_posix_path = pathlib.PosixPath
            if hasattr(pathlib, "WindowsPath"):
                pathlib.PosixPath = pathlib.WindowsPath
            try:
                return joblib.load(self._model_path)
            finally:
                pathlib.PosixPath = original_posix_path
        except Exception as error:
            raise RuntimeError(f"No fue posible cargar el modelo: {self._model_path.name}") from error
