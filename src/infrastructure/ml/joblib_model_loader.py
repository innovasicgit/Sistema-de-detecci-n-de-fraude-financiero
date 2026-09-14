from pathlib import Path
from typing import Any
import pathlib

import joblib


class JoblibModelLoader:
    """Adaptador que carga un artefacto joblib desde una ruta configurada."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path

    def load(self) -> Any:
        # Separar el error de archivo inexistente facilita diagnosticar la configuración.
        if not self._model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {self._model_path}")
        try:
            # Algunos pickles fueron creados en Linux y contienen PosixPath.
            # En Windows se mapea temporalmente a WindowsPath solo durante la carga.
            original_posix_path = pathlib.PosixPath
            if hasattr(pathlib, "WindowsPath"):
                pathlib.PosixPath = pathlib.WindowsPath
            try:
                return joblib.load(self._model_path)
            finally:
                pathlib.PosixPath = original_posix_path
        except Exception as error:
            # Se ocultan detalles internos al usuario, conservando la causa encadenada para depuración.
            raise RuntimeError(f"No fue posible cargar el modelo: {self._model_path.name}") from error
